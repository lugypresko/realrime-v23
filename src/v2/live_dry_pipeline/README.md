# Live Dry Pipeline Validation (v2)

This lightweight harness exercises the v2 Sentinel → Worker → Presenter pipeline with real multiprocessing and logs-only presentation to assess stability before full live microphone runs.

## What the checks do
- **dry_live_runner.py**: runs the pipeline for a configurable duration (default 3 minutes), counts triggers, measures latency, samples CPU/RAM, and reports duplicate/missed triggers.
- **health_report.py**: summarizes run metrics and determines `READY_FOR_LIVE` based on latency, error, and resource thresholds.

## How to run
```bash
python -m src.v2.live_dry_pipeline.dry_live_runner --minutes 3 --output-file dry_live_report.json --fail-on-warning --verbose
```

### Expected output
A summary such as:
```
--- DRY LIVE PIPELINE REPORT ---
Total triggers: <count>
Average end-to-end latency: <ms>
p95 latency: <ms>
Duplicate triggers: <count>
Missed triggers: <count>
Audio overruns: 0
STT failures: 0
Intent failures: 0
Memory drift: <mb>
CPU spikes: <count>
READY_FOR_LIVE: YES|NO
```

## Requirements
- Microphone/input device accessible to the OS
- Real STT/intent dependencies available for the chosen composition mode
- Optional: `psutil` for CPU/RAM sampling (skipped if unavailable)
- Generates `dry_live_report.json` unless `--output-file` overrides

> Note: This harness does **not** start UI elements and does **not** alter pipeline behavior; it only observes stability and readiness.
