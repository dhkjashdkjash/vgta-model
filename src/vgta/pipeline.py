from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from vgta.config import load_experiment
from vgta.gkoa import GKOA, SearchSpace, decode_gta_candidate
from vgta.gta_model import GTA, build_gta_inputs, masked_reconstruction_loss
from vgta.metrics import classification_metrics, reconstruction_metrics
from vgta.missingness import apply_block_mask, apply_mcar_mask
from vgta.prepare_data import validate_prepared_npz
from vgta.training import SymmetricScaler, predict_vim, seed_everything, train_vim
from vgta.vim_model import VisionMamba20


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _mask_matrix(values: np.ndarray, *, fraction: float, seed: int) -> np.ndarray:
    return np.stack(
        [apply_mcar_mask(np.isfinite(row), fraction=fraction, seed=seed + index)
         for index, row in enumerate(values)]
    )


def expand_daily_prior(prior: np.ndarray, *, steps_per_day: int) -> np.ndarray:
    prior = np.asarray(prior, dtype=float)
    if prior.ndim != 2:
        raise ValueError("daily ViM priors must have shape [lake, day]")
    if steps_per_day != 6:
        raise ValueError("the manuscript configuration requires six four-hour steps per day")
    return np.repeat(prior, steps_per_day, axis=1)


def _train_gta(
    series: np.ndarray,
    prior: np.ndarray,
    *,
    scaler: SymmetricScaler,
    config: Dict[str, object],
    epochs: int,
    seed: int,
    steps_per_day: int,
) -> GTA:
    seed_everything(seed)
    supervised = _mask_matrix(series, fraction=0.3, seed=seed)
    masked = series.copy()
    masked[supervised] = np.nan
    expanded_prior = expand_daily_prior(prior, steps_per_day=steps_per_day)
    if expanded_prior.shape != series.shape:
        raise ValueError("daily ViM priors do not align with the four-hour series")
    inputs = build_gta_inputs(scaler.transform(masked), scaler.transform(expanded_prior))
    model = GTA(
        input_features=inputs.shape[-1],
        filters=int(config["filters"]),
        blocks=2,
        kernel_size=int(config["kernel_size"]),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    tensor_inputs = torch.as_tensor(inputs, dtype=torch.float32)
    targets = torch.as_tensor(scaler.transform(series), dtype=torch.float32)
    mask = torch.as_tensor(supervised, dtype=torch.bool)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = masked_reconstruction_loss(model(tensor_inputs), targets, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return model


def _evaluate_gta(
    model: GTA,
    series: np.ndarray,
    prior: np.ndarray,
    *,
    scaler: SymmetricScaler,
    seed: int,
    steps_per_day: int,
    supervised: np.ndarray = None,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    if supervised is None:
        supervised = _mask_matrix(series, fraction=0.3, seed=seed)
    supervised = np.asarray(supervised, dtype=bool) & np.isfinite(series)
    if not supervised.any():
        raise ValueError("evaluation mask contains no observed target values")
    masked = series.copy()
    masked[supervised] = np.nan
    expanded_prior = expand_daily_prior(prior, steps_per_day=steps_per_day)
    if expanded_prior.shape != series.shape:
        raise ValueError("daily ViM priors do not align with the four-hour series")
    inputs = build_gta_inputs(scaler.transform(masked), scaler.transform(expanded_prior))
    model.eval()
    with torch.no_grad():
        scaled_prediction = model(torch.as_tensor(inputs, dtype=torch.float32)).numpy()
        prediction = scaler.inverse_transform(scaled_prediction)
    metrics = reconstruction_metrics(series[supervised], prediction[supervised])
    return metrics, prediction, supervised


def _block_matrix(series: np.ndarray, *, length: int, seed: int) -> np.ndarray:
    return np.stack(
        [apply_block_mask(np.isfinite(row), length=length, seed=seed + index)
         for index, row in enumerate(series)]
    )


def _evaluate_scenarios(
    model: GTA,
    series: np.ndarray,
    prior: np.ndarray,
    *,
    scaler: SymmetricScaler,
    seed: int,
    steps_per_day: int,
    recorded_real_gap: np.ndarray = None,
) -> Tuple[Dict[str, object], np.ndarray, np.ndarray]:
    masks = {
        "mcar_10": _mask_matrix(series, fraction=0.10, seed=seed + 10),
        "mcar_30": _mask_matrix(series, fraction=0.30, seed=seed + 30),
        "mcar_50": _mask_matrix(series, fraction=0.50, seed=seed + 50),
        "block_1_day": _block_matrix(series, length=steps_per_day, seed=seed + 106),
        "block_3_days": _block_matrix(series, length=3 * steps_per_day, seed=seed + 318),
        "block_7_days": _block_matrix(series, length=7 * steps_per_day, seed=seed + 742),
        "recorded_real_gap": recorded_real_gap,
    }
    scenarios: Dict[str, object] = {}
    reference_prediction = None
    reference_mask = None
    for name, mask in masks.items():
        if mask is None:
            scenarios[name] = {"status": "unavailable", "n": 0}
            continue
        mask = np.asarray(mask, dtype=bool) & np.isfinite(series)
        if not mask.any():
            scenarios[name] = {"status": "insufficient_observed_length", "n": 0}
            continue
        metrics, prediction, used_mask = _evaluate_gta(
            model,
            series,
            prior,
            scaler=scaler,
            seed=seed,
            steps_per_day=steps_per_day,
            supervised=mask,
        )
        scenarios[name] = {"status": "evaluated", **metrics}
        if name == "mcar_30":
            reference_prediction = prediction
            reference_mask = used_mask
    if reference_prediction is None or reference_mask is None:
        raise ValueError("MCAR 30% scenario did not produce an evaluation")
    return scenarios, reference_prediction, reference_mask


def run_pipeline(args) -> int:
    if args.input is None:
        raise ValueError("pipeline requires --input pointing to prepared NPZ data")
    input_path = Path(args.input)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config or "configs/experiment.yaml")
    experiment = load_experiment(config_path)
    prepared_report = validate_prepared_npz(
        input_path,
        require_paper_counts=True,
        steps_per_day=experiment.steps_per_day,
    )
    data = np.load(input_path)
    seed = experiment.seed
    vim_epochs = experiment.vim_epochs
    gta_epochs = experiment.gta_epochs
    satellite_masks = {
        domain: data[f"{domain}_satellite_mask"].astype(bool)
        for domain in ("source", "evolution", "target")
    }
    concentrations = {
        domain: data[f"{domain}_concentration"].astype(float)
        for domain in ("source", "evolution", "target")
    }
    class_arrays = {
        domain: data[f"{domain}_class"].astype(int)
        for domain in ("source", "evolution", "target")
    }
    num_classes = prepared_report["num_classes"]
    bloom_label = prepared_report["bloom_label"]
    source_concentration = concentrations["source"][satellite_masks["source"]]
    gta_scaler = SymmetricScaler.fit(data["source_series"])
    labels = {
        domain: class_arrays[domain][satellite_masks[domain]]
        for domain in ("source", "evolution", "target")
    }
    vim = VisionMamba20(channels=20, classes=num_classes)
    train_vim(
        vim,
        data["source_features"][satellite_masks["source"]],
        source_concentration,
        labels["source"],
        epochs=vim_epochs,
        seed=seed,
    )
    priors = {}
    probabilities = {}
    predicted_labels = {}
    for domain in ("source", "evolution", "target"):
        mask = satellite_masks[domain]
        prior = np.full(mask.shape, np.nan, dtype=float)
        probability = np.full(mask.shape + (num_classes,), np.nan, dtype=float)
        predicted = np.full(mask.shape, -1, dtype=int)
        valid_prior, valid_probability, valid_predicted = predict_vim(
            vim, data[f"{domain}_features"][mask]
        )
        prior[mask] = valid_prior
        probability[mask] = valid_probability
        predicted[mask] = valid_predicted
        priors[domain] = prior
        probabilities[domain] = probability
        predicted_labels[domain] = predicted
    vim_metrics = classification_metrics(
        labels["target"], predicted_labels["target"][satellite_masks["target"]],
        bloom_label=bloom_label, classes=num_classes
    )
    _write_json(output / "vim" / "metrics.json", vim_metrics)
    torch.save(
        {"state_dict": vim.state_dict(), "num_classes": num_classes,
         "bloom_label": bloom_label},
        output / "vim" / "model.pt",
    )

    search = SearchSpace(
        lower=np.array([0.001, 2.0, 16.0]), upper=np.array([0.010, 5.0, 32.0])
    )

    def objective(position: np.ndarray) -> float:
        config = decode_gta_candidate(position)
        model = _train_gta(
            data["source_series"], priors["source"],
            scaler=gta_scaler,
            config=config,
            epochs=gta_epochs,
            seed=seed,
            steps_per_day=experiment.steps_per_day,
        )
        metrics, _, _ = _evaluate_gta(
            model, data["evolution_series"], priors["evolution"],
            scaler=gta_scaler, seed=seed + 100, steps_per_day=experiment.steps_per_day,
        )
        return float(metrics["rmse"])

    result = GKOA(
        search,
        population=experiment.gkoa_population,
        iterations=experiment.gkoa_iterations,
        seed=seed,
    ).optimize(objective)
    best_config = decode_gta_candidate(result.best_position)
    _write_json(
        output / "gkoa" / "history.json",
        {"best_score": result.best_score, "best_config": best_config,
         "history": result.history, "evaluations": result.evaluations},
    )
    gta = _train_gta(
        data["source_series"], priors["source"],
        scaler=gta_scaler,
        config=best_config,
        epochs=gta_epochs,
        seed=seed,
        steps_per_day=experiment.steps_per_day,
    )
    recorded_real_gap = data["target_real_gap_mask"] if "target_real_gap_mask" in data else None
    gta_scenarios, prediction, supervised = _evaluate_scenarios(
        gta, data["target_series"], priors["target"],
        scaler=gta_scaler,
        seed=seed + 200,
        steps_per_day=experiment.steps_per_day,
        recorded_real_gap=recorded_real_gap,
    )
    _write_json(output / "gta" / "metrics.json", {"scenarios": gta_scenarios})
    np.savez(
        output / "gta" / "predictions.npz",
        truth=data["target_series"], prediction=prediction, supervised_mask=supervised,
        vim_prior=priors["target"], vim_probabilities=probabilities["target"],
    )
    torch.save(
        {"state_dict": gta.state_dict(), "input_min_mg_l": gta_scaler.minimum,
         "input_max_mg_l": gta_scaler.maximum, "config": best_config},
        output / "gta" / "model.pt",
    )
    _write_json(
        output / "manifest.json",
        {"input": str(input_path), "input_sha256": _hash(input_path),
         "seed": seed,
         "target_used_for_fitting": False,
         "lake_counts": prepared_report["lake_counts"]},
    )
    return 0


def run_vim_stage(args) -> int:
    return run_pipeline(args)


def run_gkoa_stage(args) -> int:
    return run_pipeline(args)


def run_gta_stage(args) -> int:
    return run_pipeline(args)


def run_paper_stage(args) -> int:
    return run_pipeline(args)
