from src.cache.anti_repeat import AntiRepeatCache


def test_anti_repeat_suppresses_and_records():
    cache = AntiRepeatCache(max_len=5)
    assert cache.should_suppress("p1", 0.2) is False
    cache.record("p1")
    assert cache.should_suppress("p1", 0.05) is True
    cache.record("p2")
    assert list(cache.history) == ["p1", "p2"]
