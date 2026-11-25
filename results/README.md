# Results

This directory holds generated run artifacts. Git ignores everything here
except this README, `.gitkeep`, and the curated release folder `v0.1.0/`.

## Curated release artifacts

Only final, reviewed artifacts for a tagged release are tracked:

```
results/v0.1.0/metrics.json
results/v0.1.0/*.png
```

All other contents (training history, evaluation dumps, galleries, overlays)
are reproducible outputs and must not be committed.
