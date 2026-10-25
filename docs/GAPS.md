# V1 scope and remaining gaps

The current application uses the fine-tuned EfficientNet-B0 + DCT + pHash MLP
pipeline. AI-generated video detection is deferred. Full build details and
measured results are in [BUILD_AND_METRICS.md](BUILD_AND_METRICS.md).

| Area | Current state | Remaining work |
|---|---|---|
| Facial-manipulation classification | Four arms were compared on the same held-out selection. The 150-video run is recorded with per-video scores and three timing repeats per arm. | Validate on more independent identities, source videos, datasets, and compression conditions. The paired source-group interval includes zero for the observed accuracy gain. |
| Inference parity | Training and serving share preprocessing/features; the ONNX export has a recorded parity check. | Re-run parity when preprocessing, checkpoint, exporter, or runtime changes. |
| Video acceleration | Approximate pHash reuse is guarded by RGB/DCT checks and forced refresh. All frames are decoded and retain temporal weight. | Measure quality/speed tradeoffs on long, edited, low-light, and high-motion clips before broad deployment. |
| Calibration and decisions | Scores are uncalibrated; application review bands require human review. | Calibrate on representative deployment data and measure false-positive/false-negative costs before automated decisions. |
| AI-generated videos | Not included in v1. | Build a separate dataset with source/prompt-disjoint splits and publish an independent benchmark. |
| Blockchain receipts | File and commitment hashes can be recorded and checked against a configured registry. | Deploy a registry only on a chain and account controlled by the operator; chain anchoring proves timestamped hashes, not media truth or model correctness. |
| Operations | Container configuration includes a non-root user, read-only root filesystem, capability drop, and persistent receipt/signing-key volume. | Add deployment-specific monitoring, backups, rate limits, and independent security review before public production use. |

The v1 results are research evidence for this test selection. They are not a
guarantee for new media, unseen identities, generative systems, or high-stakes
moderation decisions.
