# HyperTower Phase Plan

## Phase 1 — PAPILA Baseline Reproduction
**Goal:** Reproduce the CNN results reported in the PAPILA paper and establish proper evaluation methodology.

**Covers:**
- Reproduce PAPILA paper AUC results for VGG16, MobileNetV2, ResNet50, DenseNet121, InceptionV3
- Present results with 95% CI and boxplot across repeated CV folds
- **Show effect of data leakage:** compare leaky CV (same patient in train and test) vs proper patient-stratified CV — motivates repeated CV methodology

**Status:** Complete

---

## Phase 2 — Image Preprocessing & Backbone Selection
**Goal:** Show the effect of image-level design choices on classification performance.

**Covers:**
- **Show the effect of pre-training:** Refugelike (REFUGE-pretrained) backbone vs standard ImageNet backbones
- **Show the effect of cropping:** U-Net optic disc crop vs no cropping, and crop scale sensitivity

**Status:** Complete — best config is refugelike backbone, no crop (proper CV)

---

## Phase 3 — Clinical Data Fusion (Single-Eye)
**Goal:** Show the effect of combining fundus image features with clinical metadata in the single-eye pipeline.

**Covers:**
- **Show the effect of combining image data with clinical data:** fused bridge vs image-only and clinical-only ablations
- Architecture search: loss function (BCD vs all-losses), SE attention, IOP correction strategy, feature ablations, network dimensions, dropout, learning rate, warmup strategy, augmentation, balanced sampling
- Epoch length sensitivity

**Status:** Complete — key findings: IOP ratio correction + drop raw (+1.7%), excluding axial length helps, age is most informative clinical feature, SE adds no benefit, LR very sensitive

---

## Phase 3.5 — Confirmation & Tuning
**Goal:** Confirm that the top phase 3 findings combine additively, and tune BCD probability with the best IOP preprocessing.

**Covers:**
- Combine `iop_ratio_drop_raw` (best preprocessing) with `bcd_p07` (best loss setting)
- Extend BCD probability sweep to p=0.8 and p=0.9 to find the optimum

**Planned approach:**
- Baseline is phase 3 `iop_ratio_drop_raw` (0.8685 ± 0.011)
- All runs use best single-eye settings: refugelike, no crop, ratio IOP + drop raw, no axial length

**Status:** Not started

---

## Phase 4 — Dual CNN Architecture (Image Only)
**Goal:** Show the effect of processing both eyes jointly, and compare bilateral architectures against the single-eye baseline.

**Covers:**
- **Show the effect of a dual CNN:** bilateral tower (both eyes) vs single-eye tower, image data only
- **Architecture comparison:** canonical HyperTower bilateral (siamese shared-weight backbone returning mean+delta features) vs independent per-eye processing with late fusion

**Planned approach:**
- Image-only, no clinical data — isolates the bilateral vision question cleanly
- Use best image settings from phase 2 (refugelike, no crop)
- Compare siamese tower, independent bilateral, and single-eye (phase 3 baseline) directly

**Status:** Not started

---

## Phase 5 — Full HyperTower: Bilateral + Clinical Data + Fusion Heads
**Goal:** Bring together the best bilateral architecture (phase 4) with clinical data fusion (phase 3), and compare prediction aggregation strategies.

**Covers:**
- **Show the effect of a dual CNN + clinical data:** bilateral tower with fused clinical bridge — the full HyperTower model
- **Ensemble vs fused head:** patient-level prediction via ensemble (average OD+OS eye-level scores) vs learned fused head trained on top of single-eye scores

**Planned approach:**
- Use best settings from all prior phases (refugelike, no crop, ratio IOP + drop raw, no axial length)
- Compare tower modes: single, bilateral, ensemble, fused-head
- Establish final best configuration as the HyperTower result

**Status:** Not started
