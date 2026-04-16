
# -----------------------------------------------------------------------------
# Export helpers
# -----------------------------------------------------------------------------

import textwrap

from src.config.config import DEFAULT_EVENT_DESCRIPTIONS, DEFAULT_EVENT_LABELS, AnalysisConfig
from src.utils.serialization import save_json


def write_event_dictionary(config: AnalysisConfig) -> None:
    payload = {
        str(event_id): {
            "name": label,
            "description": DEFAULT_EVENT_DESCRIPTIONS.get(int(event_id), ""),
        }
        for event_id, label in DEFAULT_EVENT_LABELS.items()
    }
    save_json(config.output_dir / "event_label_dictionary.json", payload)



def write_outputs_readme(config: AnalysisConfig) -> None:
    readme = textwrap.dedent(
        f"""
        RAW PMU Scenario Analyzer outputs
        =================================

        Root output folder: {config.output_dir}

        - dataset/: scenario-wide integrity, event catalog, cross-bus profiles
        - general/: normal-operation-only analysis (Event==0, DATA_PRESENT==1)
        - buses/: full-timeline analysis per PMU bus
        - events/: per event instance, plus per event/bus detailed EDA

        Event dictionary:
        - {config.output_dir / 'event_label_dictionary.json'}
        """
    ).strip() + "\n"
    (config.output_dir / "README_outputs.txt").write_text(readme, encoding="utf-8")

