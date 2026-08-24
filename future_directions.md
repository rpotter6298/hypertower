# Future Directions

Deferred ideas that are out of scope for the current manuscript but worth revisiting once the paper is out. Nothing here changes headline numbers; the fused HyperTower already beats the Kovalyk-Borodyak reproduction comfortably. These are things we noticed while building `binocular_analog/` that could tighten the image-only or bilateral tower further.

## Image tower / bilateral fusion (from Kovalyk repro comparison)

Context: their end-to-end binocular ResNet-50 reproduction lands at ~0.86 bilateral AUC on PAPILA, which is above what our own `img_solo` bilateral image tower gets. The gap is not evidence that their architecture is better; it points at four training choices we haven't yet copied. Ranked by expected effect:

1. **Backbone: plain ImageNet V1 ResNet-50 instead of REFUGE-pretrained `refugelike`.**
   In the Kovalyk repro we saw V1 > V2 on PAPILA (opposite of ImageNet-accuracy prediction). Same logic could apply to REFUGE pretraining: the REFUGE checkpoint may over-specialize early features to a different fundus camera / cohort, and plain ImageNet may transfer more cleanly. Try `backbone: "resnet50"` on `img_solo` with everything else held fixed.

2. **Backpropagate the patient-level loss through the backbone.**
   In `img_solo`, the `hb` stage does not set `train_towers: true`, so the patient-level loss can only reshape the 4096→256 HyperBridge projection and the head. Kovalyk's BinoCNN is jointly trained end-to-end. Enable tower training on the `hb` stage (probably with a smaller LR to avoid destabilising the eye-level warmup) and see if the bilateral AUC moves.

3. **Wider patient-level fusion.**
   HyperBridge `embedding_mlp` compresses 4096→256. Kovalyk's head goes 4096→512→2 with a real hidden layer. Try `hidden_dim: 512` and stack a small MLP head; cheap ablation.

4. **Label smoothing + explicit class weighting in the head loss.**
   Kovalyk uses `CrossEntropyLoss(weight=inv_freq, label_smoothing=0.14)`. Ours is BCE-with-logits, no smoothing, no class weight. On a 3.5:1 imbalance at this dataset size, adding smoothing (~0.1) and a pos_weight tends to stabilise the minimizer. Cheap to try.

**NOT on the list:** disc-region ROI cropping (GT contour bbox or U-Net-derived bbox). Kovalyk pre-crops to a 299×299 square ROI, but our own crop-sweep experiments (`v4/results/experiments/backbone_replication/gtcrop_refugelike`, `unetcrop_refugelike`) showed cropping hurts on PAPILA. Do not add it back as an image-tower improvement.

## Compute / infrastructure (already documented)

See `memory/project_future_v4_compute_fixes.md` for the two deferred v4 speedups:
- Lazy BCD tower forward (~33% nt savings)
- Frozen-prefix embedding cache for hb/val (foundation for a future feature-cache service)

## Reproduction extensions (nice-to-have, not blocking)

- Fill in the remaining 6 CNN configs from Kovalyk-Borodyak (VGG16, InceptionV3, MobileNetV2, mono + bino each) if we ever want to report a fuller reproduction table rather than a single ResNet-50 headline.
- Try the paper's freeze-count under a proper Ray Tune HPO context; our 0.86-0.88 vs their 0.764 is the persistent unexplained residual, and the freeze-mode ablation did not close it.
