from src.v2.app.composition import CompositionRoot


def test_mock_pipeline_runs():
    comp = CompositionRoot(mode="mock")
    outputs = list(comp.run_pipeline())
    assert outputs, "Pipeline should produce at least one rendered output"
