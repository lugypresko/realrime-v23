class SilenceJitter:
    def __init__(self, min_continuous_ms: int = 600, window_ms: int = 800):
        self.min_continuous_ms = min_continuous_ms
        self.window_ms = window_ms
        self.silence_ms = 0.0
        self.window_silence_ms = 0.0

    def reset_on_speech(self) -> None:
        self.silence_ms = 0.0
        self.window_silence_ms = 0.0

    def update_silence(self, delta_ms: float) -> None:
        self.silence_ms += delta_ms
        self.window_silence_ms = min(self.window_ms, self.window_silence_ms + delta_ms)

    def is_trigger_ready(self) -> bool:
        return self.silence_ms >= self.min_continuous_ms and self.window_silence_ms >= self.window_ms
