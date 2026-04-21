from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.detectors.cyber.event5_detector import Event5Detector
from src.detectors.cyber.event7_detector import Event7Detector
from src.detectors.domain.models.detection_input import DetectionInput
from src.detectors.domain.models.detection_output import DetectionOutput
from src.detectors.domain.services.hybrid_event_detector import HybridEventDetector
from src.detectors.fusion.non0_fusion import Non0Fusion
from src.detectors.physical.physical_change_detector import PhysicalChangeDetector
from src.detectors.postprocessing.non0_chunker import Non0Chunker
from src.detectors.postprocessing.non0_state_machine import Non0StateMachine


@dataclass(slots=True)
class Non0DetectorV2:
    event5: Event5Detector = field(default_factory=Event5Detector)
    event7: Event7Detector = field(default_factory=Event7Detector)
    physical: PhysicalChangeDetector = field(default_factory=PhysicalChangeDetector)
    fusion: Non0Fusion = field(default_factory=Non0Fusion)
    state_machine: Non0StateMachine = field(default_factory=Non0StateMachine)
    chunker: Non0Chunker = field(default_factory=Non0Chunker)
    auxiliary_m10: HybridEventDetector | None = None

    def fit(self, normal_inputs: DetectionInput) -> None:
        self.event7.fit(normal_inputs)

    def predict(self, inputs: DetectionInput) -> DetectionOutput:
        event5_out = self.event5.detect(inputs)
        event7_out = self.event7.detect(inputs)
        aux_physical = None
        if self.auxiliary_m10 is not None:
            aux_output = self.auxiliary_m10.predict(inputs)
            aux_physical = aux_output.p_physical
        physical_out = self.physical.detect(inputs, aux_physical_prob=aux_physical)

        fused = self.fusion.fuse(
            p_event5=event5_out.p_event5,
            p_event7=event7_out.p_event7,
            p_physical=physical_out.p_physical,
        )
        y_stable, states = self.state_machine.run(
            fused.p_non0,
            p_event5=event5_out.p_event5,
            p_event7=event7_out.p_event7,
            p_physical=physical_out.p_physical,
        )
        chunks = self.chunker.build_chunks(y_stable, inputs.timestamps, fused.p_non0)

        evidence: list[dict[str, float | str]] = []
        for i in range(len(fused.p_non0)):
            evidence.append(
                {
                    "p_event5": float(event5_out.p_event5[i]),
                    "p_event7": float(event7_out.p_event7[i]),
                    "p_physical": float(physical_out.p_physical[i]),
                    "event5_state": str(event5_out.states[i]),
                    "event7_mode": str(event7_out.modes[i]),
                    "fusion_attribution": str(fused.attribution[i]),
                    "fusion_trigger_reason": str(fused.trigger_reason[i]),
                    "state_machine_state": str(states[i]),
                }
            )
        return DetectionOutput(
            p_abnormal=fused.p_non0,
            p_cyber=np.maximum(event5_out.p_event5, event7_out.p_event7),
            p_physical=physical_out.p_physical,
            y_pred_frame=fused.y_frame,
            y_pred_stable=y_stable,
            confidence=fused.confidence,
            evidence=evidence,
            chunks=chunks,
            diagnostics={
                "p_event5": event5_out.p_event5.tolist(),
                "p_event7": event7_out.p_event7.tolist(),
                "event5_states": event5_out.states,
                "event7_modes": event7_out.modes,
                "fusion_attribution": fused.attribution,
                "fusion_trigger_reason": fused.trigger_reason,
                "state_machine_states": states,
            },
        )
