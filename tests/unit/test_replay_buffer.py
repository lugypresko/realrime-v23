try:
    import numpy as np
except Exception:  # pragma: no cover
    from src.mocks import mock_numpy as np

from src.cache.replay_buffer import ReplayBuffer


def test_replay_buffer_dump(tmp_path):
    buf = ReplayBuffer(sample_rate=4, duration_sec=1)
    buf.add([0.1, -0.1, 0.2, -0.2], 0.5)
    out_path = buf.dump_to_disk(tmp_path / "replay.npz", "evt1")
    data = np.load(out_path)
    assert "audio" in data and "vad" in data
    assert data["event_id"] == "evt1"
    assert len(data["audio"]) == len(data["vad"]) == 4
