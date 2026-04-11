"""
POC — SGSMA 2026 Synchrophasor Anomaly Detection
=================================================

Single-file proof-of-concept of the clean API architecture.
All classes are stubbed but the end-to-end flow is executable.
Replace stubs with real implementations one by one, writing tests as you go.

Usage:
    python poc_sgsma2026.py --mode train
    python poc_sgsma2026.py --mode run

Author: Jorge
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("sgsma_poc")


# =============================================================================
# DOMAIN TYPES
# =============================================================================

class EventType(Enum):
    """Event types matching the SGSMA 2026 label scheme (spec §7)."""
    NORMAL = 0
    FAULT_3LG = 1          # three-phase-to-ground fault
    LINE_OUTAGE = 2        # transmission line tripped
    GEN_CHANGE = 3         # generator MW step or trip
    LOAD_CHANGE = 4        # load step or disconnect
    PMU_DROPOUT = 5        # cyber: PMU communication loss
    CYBER_PHYSICAL = 6     # concurrent dropout + physical event elsewhere
    BAD_DATA = 7           # corrupted measurements
    UNKNOWN = 8            # open-set anomaly


@dataclass
class Event:
    """Declarative description of a physical or cyber disturbance.

    Replaces the `fault(...)` factory from the sketch. One type for all
    9 label classes, parameterized by `EventType` and a free `params` dict.
    """
    node: int                    # affected bus number (1-39)
    type: EventType
    t0: float                    # start time in seconds
    tf: float                    # end time in seconds
    params: dict[str, Any] = field(default_factory=dict)

    def duration(self) -> float:
        return self.tf - self.t0


@dataclass
class NodeTimeSeries:
    """Time-indexed measurements for a single bus.

    For PMU buses, this is raw sensed data (V, I, f, ROCOF, all 3 phases).
    For non-PMU buses, it comes from the estimator.
    """
    node_id: int
    has_pmu: bool
    timestamps: np.ndarray                  # shape (T,)
    va_mag: np.ndarray                      # shape (T,)
    va_ang: np.ndarray                      # ...
    vb_mag: np.ndarray
    vb_ang: np.ndarray
    vc_mag: np.ndarray
    vc_ang: np.ndarray
    ia_mag: np.ndarray
    ia_ang: np.ndarray
    ib_mag: np.ndarray
    ib_ang: np.ndarray
    ic_mag: np.ndarray
    ic_ang: np.ndarray
    freq: np.ndarray
    rocof: np.ndarray
    data_present: np.ndarray                # 1 if valid frame, 0 if dropped

    def n_samples(self) -> int:
        return len(self.timestamps)


@dataclass
class Detection:
    """One anomaly window flagged by the detector."""
    t_start: float
    t_end: float
    score: float                            # detector confidence / statistic
    source_nodes: list[int]                 # which PMU buses contributed


@dataclass
class Classification:
    """Prediction of the event type for one detection window."""
    detection: Detection
    predicted_type: EventType
    probabilities: dict[EventType, float]


@dataclass
class Localization:
    """Origin-bus prediction for one detection window."""
    detection: Detection
    top1_bus: int
    top3_buses: list[int]
    confidence: float


# =============================================================================
# SIMULATION LAYER — synthetic scenario generation
# =============================================================================

class Simulation_IEEE_39_BUS_Andes:
    """Drives an ANDES time-domain simulation of the IEEE 39-bus system.

    Used both for synthetic training data generation and for ablation/ground
    truth comparison. For real competition data, bypass this class entirely
    and load CSVs directly into NodeTimeSeries via `load_real_data()`.
    """

    PMU_BUSES = [2, 5, 6, 10, 19, 22, 29, 39]
    ALL_BUSES = list(range(1, 40))

    def __init__(self, case_path: str | Path | None = None) -> None:
        self.case_path = case_path
        self._events: list[Event] = []
        self._results: dict[int, NodeTimeSeries] = {}
        self._has_run = False

    def set_faults(self, events: list[Event]) -> None:
        """Inject the event list into the simulation."""
        self._events = events
        self._has_run = False
        logger.info(f"Simulation configured with {len(events)} events")

    def run(self, t_end: float = 300.0, dt: float = 1 / 30) -> None:
        """Integrate the ANDES DAE with event injection.

        STUB: currently returns placeholder sinusoids. Replace with real
        ANDES calls via `andes.System(...)` + `.TDS.run()`.
        """
        logger.info(f"Running simulation for {t_end}s at dt={dt}")
        n = int(t_end / dt)
        t = np.linspace(0, t_end, n)
        for bus in self.ALL_BUSES:
            self._results[bus] = self._make_stub_timeseries(bus, t)
        self._has_run = True

    def _make_stub_timeseries(self, bus: int, t: np.ndarray) -> NodeTimeSeries:
        """Placeholder sinusoidal data until ANDES is wired up."""
        n = len(t)
        nominal_v = 199186.0  # 345 kV / sqrt(3)
        return NodeTimeSeries(
            node_id=bus,
            has_pmu=(bus in self.PMU_BUSES),
            timestamps=t,
            va_mag=np.full(n, nominal_v) + np.random.normal(0, 100, n),
            va_ang=np.zeros(n),
            vb_mag=np.full(n, nominal_v) + np.random.normal(0, 100, n),
            vb_ang=np.full(n, -120.0),
            vc_mag=np.full(n, nominal_v) + np.random.normal(0, 100, n),
            vc_ang=np.full(n, 120.0),
            ia_mag=np.full(n, 500.0) + np.random.normal(0, 2, n),
            ia_ang=np.zeros(n),
            ib_mag=np.full(n, 500.0),
            ib_ang=np.full(n, -120.0),
            ic_mag=np.full(n, 500.0),
            ic_ang=np.full(n, 120.0),
            freq=np.full(n, 60.0) + np.random.normal(0, 0.002, n),
            rocof=np.random.normal(0, 0.0003, n),
            data_present=np.ones(n, dtype=int),
        )

    def get_pmu_nodes(self) -> list[NodeTimeSeries]:
        self._assert_has_run()
        return [self._results[b] for b in self.PMU_BUSES]

    def get_non_pmu_nodes(self) -> list[NodeTimeSeries]:
        self._assert_has_run()
        return [self._results[b] for b in self.ALL_BUSES if b not in self.PMU_BUSES]

    def get_by_node_id(self, node_id: int) -> NodeTimeSeries:
        self._assert_has_run()
        if node_id not in self._results:
            raise KeyError(f"Bus {node_id} not found in simulation results")
        return self._results[node_id]

    def _assert_has_run(self) -> None:
        if not self._has_run:
            raise RuntimeError("Simulation.run() must be called first")


def load_real_data(data_dir: Path) -> dict[int, NodeTimeSeries]:
    """Load the 8 competition CSVs into NodeTimeSeries objects.

    STUB: integrate with your existing src/io/load_csv.py.
    """
    raise NotImplementedError("Hook up existing CSV loader here")


# =============================================================================
# PHYSICS MODEL — shared between simulation and estimation
# =============================================================================

class IEEE39_Model:
    """Pure physics: Ybus, swing equations, Kirchhoff, topology.

    Owns no state, no data — just the equations and the network parameters.
    Both the simulation and the estimator consume this model.
    """

    def __init__(self, raw_path: str | Path | None = None) -> None:
        self.raw_path = raw_path
        self.n_buses = 39
        self.n_gen = 10
        self._ybus: np.ndarray | None = None
        self._zbus: np.ndarray | None = None

    def load_from_raw(self) -> None:
        """Parse PSS/E RAW via pandapower to build Ybus, Zbus, branches."""
        raise NotImplementedError("Hook up src/grid/load_case.py")

    @property
    def ybus(self) -> np.ndarray:
        if self._ybus is None:
            raise RuntimeError("Call load_from_raw() first")
        return self._ybus

    @property
    def zbus(self) -> np.ndarray:
        if self._zbus is None:
            raise RuntimeError("Call load_from_raw() first")
        return self._zbus

    def swing_rhs(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Right-hand side of the classical 2nd-order swing equation."""
        raise NotImplementedError("Hook up src/dynamics/swing.py")

    def measurement_fn(self, x: np.ndarray) -> np.ndarray:
        """h(x) → 14 channels × 8 PMU buses = 112 measurements."""
        raise NotImplementedError("Hook up src/dynamics/measurement.py")

    def sensitivity_jk(self, k: int) -> np.ndarray:
        """∂h/∂P_k: how a perturbation at bus k propagates to the 8 PMUs."""
        raise NotImplementedError("Hook up src/grid/jacobians.py")

    def electrical_distance(self, i: int, j: int) -> float:
        """d_ij = |Z_ii + Z_jj − 2 Z_ij| for topology-aware scoring."""
        raise NotImplementedError("Hook up src/grid/electrical_distance.py")


# =============================================================================
# ESTIMATOR — infers full-system state from the 8 PMUs
# =============================================================================

class IEEE39_Estimator:
    """Basic estimator: algebraic reconstruction from the power-flow Jacobian.

    Given PMU measurements at 8 buses and the IEEE39_Model, infers the 31
    unobserved buses by solving the linearized network equations. No filtering,
    no temporal smoothing — one snapshot in, one snapshot out.
    """

    def __init__(self, model: IEEE39_Model) -> None:
        self.model = model
        self._pmu_data: dict[int, NodeTimeSeries] = {}
        self._estimated: dict[int, NodeTimeSeries] = {}
        self._has_estimated = False

    def set_pmu_data(self, pmu_data: dict[int, NodeTimeSeries]) -> None:
        """Inject sensed PMU data keyed by bus ID.

        Replaces the `setNode1(), setNode2(), ...` pattern from the sketch.
        """
        for bus_id, ts in pmu_data.items():
            if not ts.has_pmu:
                raise ValueError(f"Bus {bus_id} marked as non-PMU")
        self._pmu_data = pmu_data
        self._has_estimated = False

    # Backward-compat aliases (delete after refactor is stable)
    def setNode1(self, ts: NodeTimeSeries) -> None: self._pmu_data[1] = ts
    def setNode2(self, ts: NodeTimeSeries) -> None: self._pmu_data[2] = ts

    def estimate(self) -> None:
        """Run the full-system state reconstruction."""
        if not self._pmu_data:
            raise RuntimeError("set_pmu_data() must be called first")
        logger.info(f"Estimating from {len(self._pmu_data)} PMUs")
        # STUB: for POC, just copy PMU data and fabricate the rest
        self._estimated = dict(self._pmu_data)
        self._has_estimated = True

    def get_by_node_id(self, node_id: int) -> NodeTimeSeries:
        if not self._has_estimated:
            raise RuntimeError("estimate() must be called first")
        if node_id not in self._estimated:
            raise NotImplementedError(f"Non-PMU bus {node_id} estimation stub")
        return self._estimated[node_id]

    def get_full_state_trajectory(self) -> dict[int, NodeTimeSeries]:
        """Return all 39 buses as a dict — what the detector consumes."""
        if not self._has_estimated:
            raise RuntimeError("estimate() must be called first")
        return self._estimated


class IEEE39_Estimator_Kalman(IEEE39_Estimator):
    """UKF-based estimator: 30-state model with Q, R, stochastic handling.

    Subclass of the basic estimator. The interface is identical so the rest
    of the pipeline is unchanged when swapping between the two.
    """

    def __init__(self, model: IEEE39_Model, alpha: float = 1e-3) -> None:
        super().__init__(model)
        self.alpha = alpha

    def estimate(self) -> None:
        raise NotImplementedError("Hook up src/estimator/ukf.py")


# =============================================================================
# DETECTOR — system-wide anomaly detection
# =============================================================================

class Detector:
    """Detects anomaly windows from the estimated full-system trajectory.

    CRITICAL DESIGN NOTE: detection is system-wide, not per-node. A fault
    at one bus is visible across all 8 PMUs simultaneously with varying
    magnitude. Detecting per-node independently creates redundant alarms
    and breaks the physical assumption.
    """

    def __init__(self, window_size: int = 90, threshold: float = 5.0) -> None:
        self.window_size = window_size
        self.threshold = threshold
        self._trained = False

    def train(self, normal_trajectory: dict[int, NodeTimeSeries]) -> None:
        """Calibrate the detector on known-normal data (first 60 s)."""
        logger.info("Training detector on normal baseline")
        self._trained = True

    def detect(self, trajectory: dict[int, NodeTimeSeries]) -> list[Detection]:
        """Scan the estimated trajectory and return anomaly windows.

        Input: full-system estimated state (dict of bus_id → NodeTimeSeries).
        Output: list of Detection windows with timestamps and source nodes.
        """
        if not self._trained:
            raise RuntimeError("train() must be called first")
        # STUB: return empty list until real detector is wired up
        return []

    def save(self, path: Path) -> None:
        raise NotImplementedError

    def load(self, path: Path) -> None:
        self._trained = True


# =============================================================================
# CLASSIFIER — assigns EventType to each detection
# =============================================================================

class Classificator:
    """LightGBM classifier operating on residual features within detection windows."""

    def __init__(self, n_features: int = 40) -> None:
        self.n_features = n_features
        self._trained = False

    def train(
        self,
        trajectories: list[dict[int, NodeTimeSeries]],
        ground_truth_events: list[list[Event]],
    ) -> None:
        """Train on a batch of synthetic scenarios with known events."""
        logger.info(f"Training classifier on {len(trajectories)} scenarios")
        self._trained = True

    def classify(
        self,
        trajectory: dict[int, NodeTimeSeries],
        detections: list[Detection],
    ) -> list[Classification]:
        """Predict the event type for each detection window."""
        if not self._trained:
            raise RuntimeError("train() must be called first")
        # STUB
        return [
            Classification(
                detection=d,
                predicted_type=EventType.NORMAL,
                probabilities={t: 1.0 / len(EventType) for t in EventType},
            )
            for d in detections
        ]

    def save(self, path: Path) -> None:
        raise NotImplementedError

    def load(self, path: Path) -> None:
        self._trained = True


# =============================================================================
# LOCALIZER — topology-based origin inference
# =============================================================================

class Localizer:
    """Infers the origin bus of each detected event via Jacobian cosine matching.

    No training: uses the IEEE39_Model sensitivity columns J_k directly.
    """

    def __init__(self, model: IEEE39_Model) -> None:
        self.model = model

    def localize(
        self,
        trajectory: dict[int, NodeTimeSeries],
        detections: list[Detection],
    ) -> list[Localization]:
        """Pick the bus whose sensitivity column best matches the observed residual."""
        # STUB
        return [
            Localization(detection=d, top1_bus=1, top3_buses=[1, 2, 3], confidence=0.0)
            for d in detections
        ]


# =============================================================================
# METRICS — evaluation against ground truth
# =============================================================================

class Metrics:
    """Evaluates detection, classification, localization, and state RMSE."""

    @staticmethod
    def get_rmse(estimated: NodeTimeSeries, real: NodeTimeSeries) -> dict[str, float]:
        """Per-channel RMSE between estimated and ground-truth node time series."""
        return {
            "va_mag": float(np.sqrt(np.mean((estimated.va_mag - real.va_mag) ** 2))),
            "freq": float(np.sqrt(np.mean((estimated.freq - real.freq) ** 2))),
            "rocof": float(np.sqrt(np.mean((estimated.rocof - real.rocof) ** 2))),
        }

    @staticmethod
    def detection_metrics(
        predictions: list[Detection],
        ground_truth: list[Event],
        tolerance_s: float = 5.0,
    ) -> dict[str, float]:
        """Precision, recall, F1, FP/min, mean detection delay."""
        raise NotImplementedError("Hook up src/eval/metrics.py")

    @staticmethod
    def classification_metrics(
        predictions: list[Classification],
        ground_truth: list[Event],
    ) -> dict[str, Any]:
        """Macro-F1, weighted-F1, per-class, confusion matrix."""
        raise NotImplementedError

    @staticmethod
    def localization_metrics(
        predictions: list[Localization],
        ground_truth: list[Event],
        model: IEEE39_Model,
    ) -> dict[str, float]:
        """Top-1, Top-3, mean electrical distance."""
        raise NotImplementedError

    @staticmethod
    def competition_score(
        macro_f1: float,
        n_params: int,
        lam: float = 0.03,
    ) -> float:
        """SGSMA 2026 official scoring formula from spec §10.4."""
        return macro_f1 - lam * np.log10(max(n_params, 1))


# =============================================================================
# PLOTTER — visualization
# =============================================================================

class Plotter:
    """Visualization helpers. All methods return matplotlib Figure objects
    so they can be composed into paper panels or saved independently."""

    def __init__(self, output_dir: Path | None = None) -> None:
        self.output_dir = output_dir or Path("plots")
        self.output_dir.mkdir(exist_ok=True)

    def plot_diagram(self):
        """IEEE 39-bus one-line diagram with PMU and event locations highlighted."""
        raise NotImplementedError

    def plot_voltage_a_by_node_id(self, node_id: int, ts: NodeTimeSeries):
        """Phase-A voltage magnitude trajectory for one bus."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(ts.timestamps, ts.va_mag)
        ax.set_title(f"Bus {node_id} — VA magnitude")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("V (volts)")
        return fig

    def plot_voltage_3f_by_node_id(self, node_id: int, ts: NodeTimeSeries):
        """Three-phase voltage trajectory for one bus."""
        raise NotImplementedError

    def plot_confusion_matrix(self, cm: np.ndarray):
        raise NotImplementedError

    def plot_detection_timeline(
        self,
        detections: list[Detection],
        ground_truth: list[Event],
    ):
        raise NotImplementedError


# =============================================================================
# MAIN — orchestration
# =============================================================================

def build_training_scenarios() -> list[list[Event]]:
    """Build the synthetic training battery.

    Returns a list of scenarios, each containing a list of Events to inject.
    Target distribution per CLAUDE.md §9 / augmentation plan.
    """
    scenarios = []

    # 200 three-phase faults at varied buses, impedances, clearing times
    for bus in range(1, 40):
        for severity in [0.01, 0.05, 0.1]:
            scenarios.append([
                Event(node=bus, type=EventType.FAULT_3LG,
                      t0=10.0, tf=10.08,
                      params={"fault_impedance": severity, "clearing_cycles": 5})
            ])

    # 200 generation changes
    for gen in [30, 31, 32, 33, 34, 35, 36, 37, 38, 39]:
        for delta_p in [-50, -25, -10, 10, 25, 50]:
            scenarios.append([
                Event(node=gen, type=EventType.GEN_CHANGE,
                      t0=15.0, tf=16.0,
                      params={"delta_p_mw": delta_p})
            ])

    # Add load changes, outages, dropouts similarly...
    return scenarios


def train_pipeline(out_dir: Path) -> None:
    """Phase 1.5: train detector and classifier on synthetic data."""
    logger.info("=" * 60)
    logger.info("TRAINING PIPELINE")
    logger.info("=" * 60)

    model = IEEE39_Model()
    # model.load_from_raw()  # uncomment when wired

    scenarios = build_training_scenarios()
    logger.info(f"Built {len(scenarios)} training scenarios")

    trajectories = []
    ground_truth_events = []
    for i, events in enumerate(scenarios[:3]):   # only 3 for POC smoke test
        sim = Simulation_IEEE_39_BUS_Andes()
        sim.set_faults(events)
        sim.run(t_end=30.0)

        estimator = IEEE39_Estimator(model)
        pmu_dict = {ts.node_id: ts for ts in sim.get_pmu_nodes()}
        estimator.set_pmu_data(pmu_dict)
        estimator.estimate()

        trajectories.append(estimator.get_full_state_trajectory())
        ground_truth_events.append(events)

    detector = Detector()
    detector.train(trajectories[0])
    detector.save(out_dir / "detector.pkl") if False else None  # stub

    classifier = Classificator()
    classifier.train(trajectories, ground_truth_events)

    logger.info("Training complete (POC stub)")


def run_pipeline(data_dir: Path, out_dir: Path) -> None:
    """Phase 2 + 3: detect, classify, localize on real or synthetic data."""
    logger.info("=" * 60)
    logger.info("INFERENCE PIPELINE")
    logger.info("=" * 60)

    model = IEEE39_Model()
    # model.load_from_raw()

    # POC: use a stub simulation instead of real data
    sim = Simulation_IEEE_39_BUS_Andes()
    sim.set_faults([
        Event(node=39, type=EventType.FAULT_3LG, t0=20.0, tf=20.1)
    ])
    sim.run(t_end=60.0)

    estimator = IEEE39_Estimator(model)
    pmu_dict = {ts.node_id: ts for ts in sim.get_pmu_nodes()}
    estimator.set_pmu_data(pmu_dict)
    estimator.estimate()

    trajectory = estimator.get_full_state_trajectory()

    detector = Detector()
    detector.load(out_dir / "detector.pkl") if False else detector.train(trajectory)
    detections = detector.detect(trajectory)
    logger.info(f"Detected {len(detections)} anomaly windows")

    classifier = Classificator()
    classifier.load(out_dir / "classifier.pkl") if False else None
    classifier._trained = True  # POC shortcut
    classifications = classifier.classify(trajectory, detections)

    localizer = Localizer(model)
    localizations = localizer.localize(trajectory, detections)

    # Metrics against ground truth (available from the simulation)
    metrics = Metrics()
    score = metrics.competition_score(macro_f1=0.78, n_params=18)
    logger.info(f"Competition score (stub values): {score:.4f}")

    # Plotting
    plotter = Plotter(output_dir=out_dir / "plots")
    if detections:
        fig = plotter.plot_voltage_a_by_node_id(39, trajectory[39])
        fig.savefig(out_dir / "plots" / "bus39_va.png")
        logger.info(f"Saved plot to {out_dir}/plots/bus39_va.png")

    logger.info("Inference complete")


def main() -> None:
    parser = argparse.ArgumentParser(description="SGSMA 2026 POC pipeline")
    parser.add_argument("--mode", choices=["train", "run"], default="run")
    parser.add_argument("--data", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("poc_output"))
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(name)s] %(levelname)s - %(message)s",
    )
    args.out.mkdir(parents=True, exist_ok=True)

    if args.mode == "train":
        train_pipeline(args.out)
    else:
        run_pipeline(args.data, args.out)


if __name__ == "__main__":
    main()