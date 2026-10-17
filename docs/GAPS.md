# Authenfy implementation and gap register

Audit baseline: upstream commit `b1e19d088177a0991ebbfaba89b820cd08baaa90`.

## Fine-tuned v1

The active development target is EfficientNet-B0 + DCT + pHash into an MLP.
AI-video work is deferred. The earlier pretrained engine's audit below is
historical evidence, not a result for this fine-tuned architecture.

| Gap | V1 work | Status |
|---|---|---|
| Lost training weights and no reproducible training | ImageNet-initialized EfficientNet fine-tuning; frozen-backbone deep/fused MLP controls; pinned source-pair manifest | T4 runtime reset during preparation; dataset restoration in progress; completed fine-tuning pending |
| Temporary Colab storage loses progress after runtime replacement | Persistent checked crop archives, epoch snapshots and completed-stage copies; empty-directory restore and corruption checks | Implemented and locally tested; active Drive mount awaiting Google authorization |
| No comparable accuracy/speed record | Four arms on identical held-out complete clips; three rotated timing repeats; raw predictions and all timing components | Full held-out measurements pending |
| Inference path can disagree with training features | Shared crop, DCT, pHash and normalization; CPU ONNX parity; trained export and benchmark hashes checked on installation | Implemented and code-tested; actual trained export pending |
| pHash collisions and hidden skipped frames | RGB/DCT residual guards, 12-frame refresh and explicit approximate-reuse evidence; reused low-score clips abstain | Implemented; accuracy/speed tradeoff must be measured |
| Confounded or inflated headline accuracy | Source-pair separation, both source IDs checked, fixed test subset, balanced metrics, 83.33% majority-class control, source-group intervals | Protocol implemented; identity/YouTube-source separation remains unverified |
| Incomplete result preservation | Epoch/config/runtime records, frame/video scores, per-method metrics, calibration, resource snapshots, CSV timing repeats, curves and provenance | Recording implemented; result bundle pending |
| Face localization and deployment calibration | Current engine explicitly discloses full-frame center-crop screening and uncalibrated review bands | Open; do not claim face localization or deployment confidence |

See [EFFICIENTNET_EXPERIMENT.md](EFFICIENTNET_EXPERIMENT.md) for exact scope.

## Earlier pretrained engine

| Priority | Original gap | Work and evidence | Status |
| --- | --- | --- | --- |
| P0 | Missing `models.detects` / `models.detect` implementation; models ignored by Git | Pinned checksummed ONNX model, reproducible downloader, explicit 503 for missing or corrupt weights | Fixed |
| P0 | Root route loads nonexistent `index.html`; navigation targets unregistered endpoints | Application factory and coherent template routes; route tests | Fixed |
| P0 | Two independent Flask apps expose divergent analysis implementations | One application factory; compatibility import in old route module | Fixed |
| P0 | Same-filename uploads race, overwrite and delete each other | Random temporary paths, one concurrent analysis, busy response, cleanup on every path | Fixed |
| P0 | Unbounded upload/decoder work and leaked exception details | 64 MiB cap, dimension/duration checks, isolated analysis worker, 90-second kill deadline, generic unexpected-error response | Fixed for local service; OS/container resource limits still needed for public exposure |
| P0 | Generic generated-image model cannot establish face-swap coverage | T4 audit found 0/20 manipulated examples detected at 0.5; preserve `face-audit-baseline.json`; coverage-aware abstention added | False reassurance fixed; specialist integration separately validated |
| P1 | Perceptual hash similarity treated as interchangeable content | pHash included as evidence only; only identical RGB bytes reuse inference; collision regression test | Fixed |
| P1 | No temporal sampling/evidence or honest confidence policy | Uniform 16-frame sampling with endpoints, component scores, explicit uncalibrated bands and disagreement/coverage abstention | Implemented; independent calibration open |
| P1 | Broken/unsafe browser result handling and no failure states | Text-only rendering, upload progress, JSON errors, busy state, mobile layout, receipt download and byte verification | Fixed; browser checks pass locally |
| P1 | No analysis provenance or verification implementation | HMAC over canonical JSON, SQLite receipts, random receipt access tokens, exact-byte SHA-256 verification, tamper tests | Fixed locally; key custody/retention remain operator responsibilities |
| P1 | Blockchain and immutability claims without a contract | Immutable publisher registry implemented and exercised with authorization, zero commitment, duplicate and altered-record tests | Contract tested on development chain; no public deployment or app RPC integration |
| P1 | Vulnerable/unreproducible dependency set | Patched Python pins; pip-audit report; npm lockfile; local tests and CI workflow | Python audit has zero known vulnerabilities at run time |
| P1 | No independent model evaluation or calibration | Pinned model/input hashes; source-graph/ONNX parity; small labeled regression audit kept distinct from generalization evidence | Independent evaluation open |
| P1 | Facial model expects dlib-aligned crops | YuNet five-landmark alignment approximates DeepfakeBench transform; expose missing, small and partial-face coverage | Actual alignment path checked on all 32 crops with zero missing faces: 19/20 manipulated crops detected, 5/12 real crops falsely flagged at 0.5. Broad raw-image/video validation still required |
| P2 | Audio/fusion/lip synchronization/rPPG claims | Explicitly disclose that audio is not analyzed | Not implemented |
| P2 | Grad-CAM promises without localization code | Real per-frame component scores; no fabricated heatmap | Localization not implemented |
| P2 | Claimed React frontend, account registration and deployment | Actual stack documented as Flask/Jinja/native CSS/JS; no account collection; loopback deployment | Local delivery only; public service/authentication open |

## Evidence and promotion rules

- A running notebook, a numerical parity test and a labeled model benchmark are different kinds of evidence.
- The 32 example crops are drawn from eight correlated video/method groups. FF++ samples may overlap the specialist's training data. Do not call their score independent accuracy.
- Do not tune thresholds or fit adapters on these audit examples. Reserve a new identity/video/source-disjoint dataset for calibration and final evaluation.
- Compare image, video and compressed-media results separately. Count missed faces and decode failures; do not discard them to improve metrics.
- Require a representative independent evaluation before presenting any deployment-level accuracy, false-positive rate or calibrated confidence.
- Public Ethereum integration needs a verified contract address, issuer identity, chain ID, finality/receipt checks and a clear retention policy. A chain record is not evidence that media content is true.
