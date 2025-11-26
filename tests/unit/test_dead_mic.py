from src.sentinel.dead_mic import DeadMicDetector


def test_dead_mic_triggers_after_silence():
    detector = DeadMicDetector(max_silence_sec=0.1, rms_floor=0.0)
    now = 0.0
    assert detector.update([0.0, 0.0], 0.0, now) is False
    assert detector.update([0.0, 0.0], 0.0, now + 0.05) is False
    assert detector.update([0.0, 0.0], 0.0, now + 0.2) is True
