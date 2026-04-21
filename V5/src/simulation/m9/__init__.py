"""M9 scenario simulator and validation framework."""

from src.simulation.m9.generator import generate_scenario, generate_template_matrix
from src.simulation.m9.validation import validate_scenario, run_m9_validation_suite
from src.simulation.m9.templates import get_template, list_templates

__all__ = [
    "generate_scenario",
    "generate_template_matrix",
    "validate_scenario",
    "run_m9_validation_suite",
    "get_template",
    "list_templates",
]
