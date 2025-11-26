from src.v2.interfaces import WorkerOutput
from src.v2.presenter.presenter import PresenterProcess
from src.v2.presenter.services import SimpleResultFormatter
from src.v2.telemetry.client import InMemoryTelemetry


def test_presenter_formats_output():
    formatter = SimpleResultFormatter()
    telemetry = InMemoryTelemetry()
    presenter = PresenterProcess(formatter, telemetry)
    output = WorkerOutput(
        event_id="evt-1",
        text="hello world",
        prompt_id="prompt-1",
        score=0.9,
        whisper_latency_ms=10.0,
        intent_latency_ms=5.0,
        total_latency_ms=20.0,
        decision="SUCCESS",
        metadata={},
    )
    rendered = presenter.run([output])
    assert rendered and "hello world" in rendered[0]
