from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np


Objective = Callable[[np.ndarray], float]
EPSILON = 1e-8


def _semi_major_axis(
    radius: float,
    speed: float,
    gravitational_parameter: float,
    maximum_axis: float,
    epsilon: float = EPSILON,
) -> float:
    safe_radius = max(float(radius), epsilon)
    safe_gravity = max(float(gravitational_parameter), epsilon)
    denominator = 2.0 / safe_radius - float(speed) ** 2 / safe_gravity
    axis = safe_radius if abs(denominator) < epsilon else abs(1.0 / denominator)
    return float(np.clip(axis, epsilon, maximum_axis))


def _comet_axis(
    axis: float,
    comet_factor: float,
    maximum_axis: float,
    epsilon: float = EPSILON,
) -> float:
    return float(np.clip(axis * (1.0 + comet_factor), epsilon, maximum_axis))


def _orbital_position(axis: float, eccentricity: float, eccentric_anomaly: float) -> np.ndarray:
    radius = axis * (1.0 - eccentricity * np.cos(eccentric_anomaly))
    true_anomaly = 2.0 * np.arctan2(
        np.sqrt(1.0 + eccentricity) * np.sin(eccentric_anomaly / 2.0),
        np.sqrt(1.0 - eccentricity) * np.cos(eccentric_anomaly / 2.0),
    )
    return np.array(
        [radius * np.cos(true_anomaly), radius * np.sin(true_anomaly)],
        dtype=float,
    )


def _normalized_diversity(
    positions: np.ndarray,
    best: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    epsilon: float = EPSILON,
) -> float:
    distances = np.linalg.norm(positions - best, axis=1)
    span = np.linalg.norm(upper - lower)
    return float(np.mean(distances) / (span + epsilon))


@dataclass(frozen=True)
class SearchSpace:
    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self) -> None:
        lower = np.asarray(self.lower, dtype=float)
        upper = np.asarray(self.upper, dtype=float)
        if lower.ndim != 1 or upper.shape != lower.shape:
            raise ValueError("lower and upper bounds must be one-dimensional and equal-sized")
        if np.any(lower >= upper):
            raise ValueError("every lower bound must be smaller than its upper bound")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    @property
    def dimension(self) -> int:
        return int(self.lower.size)

    def contains(self, position: np.ndarray) -> bool:
        value = np.asarray(position, dtype=float)
        return value.shape == self.lower.shape and bool(
            np.all(value >= self.lower) and np.all(value <= self.upper)
        )


@dataclass(frozen=True)
class GKOAResult:
    best_position: np.ndarray
    best_score: float
    history: List[float]
    evaluations: List[Dict[str, object]]


def decode_gta_candidate(position: np.ndarray) -> Dict[str, object]:
    value = np.asarray(position, dtype=float)
    if value.shape != (3,):
        raise ValueError("GTA candidate must have three dimensions")
    return {
        "learning_rate": float(value[0]),
        "kernel_size": int(np.rint(value[1])),
        "filters": int(np.rint(value[2])),
    }


class GKOA:
    def __init__(
        self,
        space: SearchSpace,
        *,
        population: int = 5,
        iterations: int = 10,
        seed: int = 42,
        initial_mass: float = 0.1,
        mass_decay: float = 15.0,
        spiral_decay1: float = 0.1,
        spiral_decay2: float = 0.1,
        comet_factor: float = 0.2,
        stagnation_threshold: float = 0.01,
        kepler_iterations: int = 10,
        kepler_tolerance: float = 1e-6,
    ) -> None:
        if population < 2 or iterations < 1:
            raise ValueError("population must be at least two and iterations positive")
        self.space = space
        self.population = population
        self.iterations = iterations
        self.seed = seed
        self.initial_mass = initial_mass
        self.mass_decay = mass_decay
        self.spiral_decay1 = spiral_decay1
        self.spiral_decay2 = spiral_decay2
        self.comet_factor = comet_factor
        self.stagnation_threshold = stagnation_threshold
        self.kepler_iterations = kepler_iterations
        self.kepler_tolerance = kepler_tolerance

    @staticmethod
    def solve_kepler_equation(
        mean_anomaly: float,
        eccentricity: float,
        *,
        max_iterations: int = 10,
        tolerance: float = 1e-6,
    ) -> float:
        eccentric_anomaly = float(mean_anomaly)
        for _ in range(max_iterations):
            residual = (
                eccentric_anomaly
                - eccentricity * np.sin(eccentric_anomaly)
                - mean_anomaly
            )
            derivative = 1.0 - eccentricity * np.cos(eccentric_anomaly)
            if abs(derivative) < 1e-15:
                break
            updated = eccentric_anomaly - residual / derivative
            if abs(updated - eccentric_anomaly) < tolerance:
                eccentric_anomaly = updated
                break
            eccentric_anomaly = updated
        return eccentric_anomaly

    def _evaluate(
        self,
        objective: Objective,
        position: np.ndarray,
        evaluations: List[Dict[str, object]],
    ) -> float:
        score = float(objective(position.copy()))
        if not np.isfinite(score):
            raise ValueError("objective must return a finite value")
        evaluations.append({"position": position.tolist(), "score": score})
        return score

    def _repair(self, position: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        repaired = position.copy()
        outside = (repaired < self.space.lower) | (repaired > self.space.upper)
        if np.any(outside):
            reset = rng.uniform(self.space.lower, self.space.upper)
            repaired[outside] = reset[outside]
        return repaired

    def optimize(self, objective: Objective) -> GKOAResult:
        rng = np.random.default_rng(self.seed)
        dimension = self.space.dimension
        subspaces = int(np.ceil(dimension / 2))
        positions = rng.uniform(
            self.space.lower, self.space.upper, (self.population, dimension)
        )
        velocities = np.zeros_like(positions)
        semi_major_axis = np.ones((self.population, subspaces), dtype=float)
        eccentricity = rng.random((self.population, subspaces)) * 0.8
        mean_anomaly = np.zeros((self.population, subspaces), dtype=float)
        maximum_axis = float(np.linalg.norm(self.space.upper - self.space.lower))
        evaluations: List[Dict[str, object]] = []
        scores = np.array([self._evaluate(objective, row, evaluations) for row in positions])
        best_index = int(np.argmin(scores))
        best = positions[best_index].copy()
        best_score = float(scores[best_index])
        history = [best_score]

        for iteration in range(self.iterations):
            gravitational_parameter = max(
                self.initial_mass
                * np.exp(-self.mass_decay * iteration / self.iterations),
                EPSILON,
            )
            diversity = _normalized_diversity(
                positions,
                best,
                self.space.lower,
                self.space.upper,
            )
            for candidate_index in range(self.population):
                old_position = positions[candidate_index].copy()
                proposal = old_position.copy()
                proposed_axes = semi_major_axis[candidate_index].copy()
                proposed_eccentricity = eccentricity[candidate_index].copy()
                for subspace_index in range(subspaces):
                    first = 2 * subspace_index
                    second = first + 1
                    dimensions = [first] if second >= dimension else [first, second]
                    relative = np.zeros(2, dtype=float)
                    velocity = np.zeros(2, dtype=float)
                    relative[: len(dimensions)] = (
                        positions[candidate_index, dimensions] - best[dimensions]
                    )
                    velocity[: len(dimensions)] = velocities[candidate_index, dimensions]
                    radius = max(float(np.linalg.norm(relative)), EPSILON)
                    speed = float(np.linalg.norm(velocity))
                    axis = _semi_major_axis(
                        radius,
                        speed,
                        gravitational_parameter,
                        maximum_axis,
                    )
                    eccentricity_vector = (
                        (speed ** 2 - gravitational_parameter / radius) * relative
                        - np.dot(relative, velocity) * velocity
                    ) / gravitational_parameter
                    current_eccentricity = float(
                        np.clip(np.linalg.norm(eccentricity_vector), 0.0, 0.99)
                    )
                    if diversity < self.stagnation_threshold or rng.random() < 0.3:
                        updated_eccentricity = float(
                            np.clip(0.9 + 0.1 * rng.random(), 0.0, 0.99)
                        )
                        updated_axis = _comet_axis(
                            axis,
                            self.comet_factor,
                            maximum_axis,
                        )
                    else:
                        updated_axis = float(
                            np.clip(
                                axis * (1.0 - self.spiral_decay1 * rng.random()),
                                EPSILON,
                                maximum_axis,
                            )
                        )
                        updated_eccentricity = float(
                            np.clip(
                                current_eccentricity
                                * (1.0 - self.spiral_decay2 * rng.random()),
                                0.0,
                                0.99,
                            )
                        )
                    proposed_axes[subspace_index] = updated_axis
                    proposed_eccentricity[subspace_index] = updated_eccentricity
                    mean_motion = np.sqrt(
                        gravitational_parameter / (updated_axis ** 3 + EPSILON)
                    )
                    mean_anomaly[candidate_index, subspace_index] += mean_motion * rng.random()
                    anomaly = self.solve_kepler_equation(
                        mean_anomaly[candidate_index, subspace_index],
                        updated_eccentricity,
                        max_iterations=self.kepler_iterations,
                        tolerance=self.kepler_tolerance,
                    )
                    mapped = _orbital_position(
                        updated_axis,
                        updated_eccentricity,
                        anomaly,
                    )
                    proposal[dimensions] = best[dimensions] + mapped[: len(dimensions)]
                proposal = self._repair(proposal, rng)
                proposal_score = self._evaluate(objective, proposal, evaluations)
                if proposal_score < scores[candidate_index]:
                    velocities[candidate_index] = proposal - old_position
                    positions[candidate_index] = proposal
                    scores[candidate_index] = proposal_score
                    semi_major_axis[candidate_index] = proposed_axes
                    eccentricity[candidate_index] = proposed_eccentricity
                    if proposal_score < best_score:
                        best_score = proposal_score
                        best = proposal.copy()
                else:
                    velocities[candidate_index] *= 0.5
            history.append(best_score)
        return GKOAResult(best, best_score, history, evaluations)


def run_gkoa(args) -> int:
    if args.input is None:
        raise ValueError("GKOA requires --input pointing to prepared evolutionary data")
    from vgta.pipeline import run_gkoa_stage

    return int(run_gkoa_stage(args) or 0)
