from typing import runtime_checkable

from src.v2.interfaces import AudioSource, SilencePolicy, STTEngine, IntentClassifier, GovernorService, TelemetryClient, ResultFormatter
from src.v2.mocks.mocks import MockAudioSource, ImmediateSilencePolicy, MockSTTEngine, MockIntentClassifier
from src.v2.worker.services import SimpleGovernor
from src.v2.telemetry.client import InMemoryTelemetry
from src.v2.presenter.services import SimpleResultFormatter


def test_mock_implements_interfaces():
    assert isinstance(MockAudioSource(), AudioSource)
    assert isinstance(ImmediateSilencePolicy(), SilencePolicy)
    assert isinstance(MockSTTEngine(), STTEngine)
    assert isinstance(MockIntentClassifier(), IntentClassifier)
    assert isinstance(SimpleGovernor(), GovernorService)
    assert isinstance(InMemoryTelemetry(), TelemetryClient)
    assert isinstance(SimpleResultFormatter(), ResultFormatter)
