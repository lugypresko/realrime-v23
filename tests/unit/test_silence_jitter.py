from src.cache.silence_jitter import SilenceJitter


def test_silence_jitter_triggers_after_window():
    jitter = SilenceJitter(min_continuous_ms=600, window_ms=800)
    jitter.update_silence(400)
    assert jitter.is_trigger_ready() is False
    jitter.update_silence(200)
    assert jitter.is_trigger_ready() is False
    jitter.update_silence(400)
    assert jitter.is_trigger_ready() is True
    jitter.reset_on_speech()
    assert jitter.is_trigger_ready() is False
