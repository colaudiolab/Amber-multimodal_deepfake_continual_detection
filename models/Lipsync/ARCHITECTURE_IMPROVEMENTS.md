# Architecture Improvements: Before vs After & Verdict

This doc compares the **original** design to the **proposed** changes and states whether each change is actually better or has trade-offs.

---

## 1. Preserve Temporal Resolution

### Visual branch

| Aspect | Original | After |
|--------|----------|--------|
| Pooling | `out.mean(dim=[3, 4])` → (B, D, T') | `AdaptiveAvgPool3d((t_len, 1, 1))` then squeeze → (B, D, T') |
| Temporal tokens | T' = T (no temporal stride in convs) | Same: T' = T |

**Verdict: Functionally equivalent.** The visual encoder never reduced time (all strides are (1,·,·)). The new code only makes “pool over space, keep time” explicit and robust if someone later adds temporal stride. **Keep the change** for clarity.

---

### Audio branch

| Aspect | Original | After |
|--------|----------|--------|
| Strides | layer3 stride (2,2), layer4 (2,1) → T' ≈ T/8 | layer3 (2,1), layer4 (2,1) → T' ≈ T/4 |
| Pooling | `out.mean(dim=2)` | `AdaptiveAvgPool2d((1, t_len))` then squeeze |

**Verdict: Likely better for lip-sync.** More audio frames (≈2× tokens) help with 2–3 frame misalignments. **Trade-off:** more tokens → more cross-attention and temporal transformer cost (O(T²)). **Config:** `LipSyncModel(..., preserve_audio_temporal=True)` (default, T/4) vs `False` (original T/8) for ablation.

---

## 2. Gated Cross Attention

| Aspect | Original | After |
|--------|----------|--------|
| Fusion | `concat(v_out, a_out)` → `Linear(2*D, D)` | `gate = σ(MLP(concat(v_out, a_out)))`, then `gate*v_out + (1-gate)*a_out` → `Linear(D, D)` |

**Verdict: Theoretically better, no clear downside.** The gate lets the model trust video more when audio is noisy and audio when lips are occluded. If the gate collapses to ~0.5, behavior is similar to averaging. Extra cost: one small MLP. **Recommendation:** keep; validate with an ablation (train with gate vs fixed 0.5).

---

## 3. High-Frequency (Laplacian) Artifact Branch

| Aspect | Original | After |
|--------|----------|--------|
| Inputs | Raw feature map + delta map | Raw + delta + **Laplacian(raw_video)** → Conv3D → feature |
| Role | Temporal + delta only | + spatial high-freq (GAN smoothing, blending, boundaries) |

**Verdict: Theoretically better for the stated artifacts.** Laplacian is a fixed high-pass prior; the rest is learned. **Trade-offs:** (1) Artifact detector needs `raw_video` (now passed from `LipSyncModel`). (2) Slightly more parameters and compute. High-freq branch is optional (`use_high_freq=True` by default; set to `False` to match old behavior). **Recommendation:** keep; run ablation with `use_high_freq=False` to measure impact.

---

## Summary

| Change | Better? | Notes |
|--------|--------|--------|
| Visual AdaptiveAvgPool3d | Same result, clearer intent | Keep |
| Audio less temporal stride | Yes, for lip-sync | Keep; use `preserve_audio_temporal=True/False` to ablate |
| Audio AdaptiveAvgPool2d | Same result | Keep |
| Gated cross attention | Yes, in principle | Keep; ablate to confirm |
| High-freq Laplacian branch | Yes, for GAN/blend artifacts | Keep; ablate with `use_high_freq=False` |

**Conclusion:** All proposed changes are either equivalent or improvements in design. The only real trade-off is **compute vs quality** for the increased audio temporal resolution; the rest are either clarity or strictly more expressive. Run ablations on your data to confirm gains.

---

## Ablation: how to toggle improvements

Use these flags when constructing `LipSyncModel` to compare against the original design:

| Flag | Default | Effect |
|------|---------|--------|
| `preserve_audio_temporal=True` | True | More audio tokens (T/4). Set `False` for original T/8. |
| `use_high_freq_artifact=True` | True | Use Laplacian high-freq branch. Set `False` to disable. |
| Gated cross-attention | always on | No flag; compare by training an older checkpoint or a fork with concat-only fusion. |

Example (original-style audio + no high-freq branch):

```python
model = LipSyncModel(
    detect_artifacts=True,
    preserve_audio_temporal=False,   # original T/8
    use_high_freq_artifact=False,     # no Laplacian branch
)
```

---

## 4. Sync Contrastive Loss (Alignment)

| Aspect | Original | After |
|--------|----------|--------|
| Loss | BCE + batch contrastive (real vs fake) | + **sync contrastive**: (video, correct_audio) vs (video, shifted_audio ±5,±10,±15) |

**Verdict: Improvement.** Existing `cross_modal_contrastive_loss` is real-vs-fake. Sync loss forces temporal alignment on real pairs. **Implementation:** `sync_contrastive_loss()` in `losses.py`; `--sync-weight 0.2`, `--sync-shift-frames 5,10,15` in train/finetune. Set `--sync-weight 0` to disable.

---

## 5. Multi-Scale Temporal Convolutions

| Aspect | Original | After |
|--------|----------|--------|
| Pre-Transformer | Sequential Conv1d k=3 → k=5 | **Branches:** k=3, k=5, k=7 → concat → Linear |

**Verdict: Improvement.** Captures micro (k=3), phoneme (k=5), syllable (k=7) motion. **Implementation:** `TemporalTransformer(multi_scale_pre_conv=True)` by default. Set `multi_scale_pre_conv=False` for original sequential conv.

---

## 7. Training Stability (AdamW, LR, Scheduler)

| Aspect | Original | After |
|--------|----------|--------|
| Optimizer | train: Adam; finetune: AdamW (wd=1e-4) | AdamW with configurable `weight_decay` (e.g. 0.01). Optional Cosine LR + warmup. |
| LR | lr=1e-4, lr_encoder=1e-5 | Same defaults; proposal encoder 5e-5 is close (use `--lr-encoder 5e-5`). |

**Verdict:** Finetune already uses AdamW. Add `--weight-decay 0.01` and optional Cosine+warmup for stability. train.py still uses Adam; can add AdamW option.

---

## 8. Hard Negative Mining

| Proposal | Status |
|----------|--------|
| If pred confidence > 0.9 but label wrong → store sample, re-train later | **Not implemented.** Can be added as: per-epoch collect indices where (sigmoid(logit) > 0.9 and label wrong), then oversample those indices in a later epoch (e.g. weighted sampler or buffer). |

**Verdict:** Theoretically improves robustness. Defer to a follow-up (buffer + resample logic).

---

## 9. Temporal Jitter Augmentation

| Aspect | Original | After |
|--------|----------|--------|
| Frame sampling | Fixed window from video | **Already:** `_sample_aligned_contiguous_clip(..., train_mode=True)` uses **random start** when T > video_frames. |

**Verdict: Already implemented.** Dataset uses random start frame in train mode. Optional: random crop length (e.g. 24–32 frames) for extra jitter; not added by default.

---

## 10. Final Classifier (Calibration)

| Aspect | Original | After |
|--------|----------|--------|
| Head | Linear → ReLU → Dropout → Linear | Linear → **GELU** → Dropout → **LayerNorm** → Linear |

**Verdict: Improvement.** GELU + LayerNorm often improve calibration. **Implementation:** `ClassificationHead` in `classifier.py` updated.

---

## 11. Sliding-Window Inference

| Aspect | Proposal | Status |
|--------|----------|--------|
| Predict from multiple windows, score = mean(window_scores) | Long videos: **already implemented.** `LipSyncPredictor` uses `chunk_size`, `chunk_stride`, aggregates `window_confidences` (e.g. weighted by VAD). Short clips: single clip. |

**Verdict: Already in place for long videos.** For short clips, a single window is typical; optional multi-window for short clips can be added (e.g. 2 overlapping half-width windows) for extra stability.
