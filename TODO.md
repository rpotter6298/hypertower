# Project TODO

## Paper

- [ ] **Learning curve analysis** — train on 25/50/75/100% of training data, plot AUC vs n.
  Motivation: empirical evidence that the model is data-starved, which justifies the decision
  not to pursue attention-gating (transformer) extensions to the NTowerHT bridge.
  If the curve is still ascending at full data → supports the argument that a more expressive
  architecture would overfit at this sample size. Generates a figure for the paper.

- [ ] **GradCAM nasal-side analysis** — re-run GradCAM separately for OD and OS eyes rather
  than aggregated. The current aggregation mirrors the two eyes against each other, washing out
  any directional bias. Clinically, we would expect GradCAM attention offset from the disc center
  to trend toward the nasal side (where RNFL loss presents earliest in glaucoma). If the model
  has learned this, it would only be visible in per-side heatmaps — OD and OS are mirror images
  so the nasal direction is opposite for each. This could be a strong interpretability result
  for the paper if the bias is present.

- [ ] **Quantify attention-gating as future work** — use the learning curve result + parameter
  count ratio (Q/K/V projections over fusion_dim vs training n) to formally justify the choice.
  Frame in paper as: "we identify cross-attention inside the NTowerHT bridge as a promising
  extension, but our sample size (N≈400 training patients) is insufficient to avoid overfitting
  a more expressive interaction layer" — cite the learning curve figure as evidence.
