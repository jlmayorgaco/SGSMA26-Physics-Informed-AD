from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / 'powerdynamics_ieee39' / 'output' / 'load_event_identifiability_v1'
RES = ROOT / 'results'

def test_registry_is_model_table_backed_and_hidden_only():
    r = pd.read_csv(RES / 'load_candidate_registry.csv')
    assert len(r) == 16
    assert set(r.registry_status) == {'VALID_MODEL_TABLE'}
    assert set(r.pmu_status) == {'HIDDEN'}

def test_temporal_operator_is_not_mislabeled_as_validated_tangent():
    r = pd.read_csv(RES / 'event_visibility_load.csv')
    assert len(r) == 16
    assert r.operator.str.contains('CAUSAL_ENVELOPE_REFERENCE').all()
    assert (r.validated_against_tds.sum() == 1)

def test_inference_outputs_contain_no_hidden_truth_fields():
    r = pd.read_csv(RES / 'amplitude_inference_comparison.csv')
    assert not any(c.lower().startswith(('hidden', 'truth', 'full')) for c in r.columns)
    s = pd.read_csv(RES / 'campaign_summary.csv').iloc[0]
    assert s.no_hidden_truth_leakage == 'PASS'

def test_bank_shortfall_is_explicitly_rejected():
    m = pd.read_csv(RES / 'load_event_manifest.csv')
    q = pd.read_csv(RES / 'load_event_rejections.csv')
    assert len(m) == 76 and len(q) == 74
    assert q.reason.str.contains('not executed').all()
