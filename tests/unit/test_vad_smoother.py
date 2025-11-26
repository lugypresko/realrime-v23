from src.cache.vad_smoother import VADSmoother


def test_vad_smoother_hysteresis():
    sm = VADSmoother(window_ms=400, enter_threshold=0.6, exit_threshold=0.3)
    assert sm.is_speaking() is False
    assert sm.update(0.7, now=0.0) is True
    assert sm.update(0.65, now=0.1) is True
    sm.update(0.1, now=0.6)
    assert sm.update(0.1, now=1.0) is False
    assert sm.is_speaking() is False
