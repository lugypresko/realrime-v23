import json
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List

from src.logging.structured_logger import log_event


class AudioLoader:
    def __init__(self, source: str) -> None:
        self.source = source

    def load(self) -> Iterable[List[float]]:
        """Yield 3-5s audio chunks for evaluation."""
        try:
            import soundfile as sf
            import numpy as np

            data, sr = sf.read(self.source)
            chunk = int(sr * 3)
            for idx in range(0, len(data), chunk):
                yield data[idx : idx + chunk].astype("float32")
        except Exception:
            yield []


class YouTubeLoader(AudioLoader):
    def __init__(self, url: str, download_dir: Path = Path(".cache")) -> None:
        super().__init__(url)
        self.download_dir = download_dir

    def load(self) -> Iterable[List[float]]:
        try:
            from pytube import YouTube
            import soundfile as sf
            import numpy as np

            self.download_dir.mkdir(parents=True, exist_ok=True)
            stream = YouTube(self.source).streams.filter(only_audio=True).first()
            if not stream:
                return []
            file_path = Path(stream.download(output_path=self.download_dir))
            data, sr = sf.read(file_path)
            chunk = int(sr * 3)
            for idx in range(0, len(data), chunk):
                yield data[idx : idx + chunk].astype("float32")
        except Exception:
            yield []


def evaluate_audio_chunks(chunks: Iterable, transcribe: Callable, classify: Callable) -> Dict:
    results = []
    for idx, chunk in enumerate(chunks):
        start = time.monotonic()
        try:
            text = transcribe(chunk)
            intent_id, score = classify(text)
            latency_ms = (time.monotonic() - start) * 1000
            results.append(
                {
                    "index": idx,
                    "text": text,
                    "intent": intent_id,
                    "score": score,
                    "latency_ms": latency_ms,
                }
            )
        except Exception as exc:
            results.append({"index": idx, "error": str(exc)})
    report = {"results": results}
    Path("evaluation.json").write_text(json.dumps(report, indent=2))
    log_event({"type": "EVAL_COMPLETE", "count": len(results)})
    return report
