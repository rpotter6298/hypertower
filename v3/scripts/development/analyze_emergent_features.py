import sys
from pathlib import Path

# Make sure we can import from v3 classes when running directly
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from v3.classes.hypertower_models import SingleEyeHT
from v3.classes.papila_builders import build_papila_data
from v3.classes.profiles import build_papila_profile
from v3.classes.loader_factory import make_loader, filter_eye_samples
from v3.classes.transforms import build_eval_transform
from v3.classes.utils import choose_device


def extract_emergent_features(model, loader, device, num_classes=2):
    """
    Finds samples where the fused bridge is correct, but both individual
    towers are wrong, and extracts the driving features from the fusion_dim.

    Assumes `model` is a SingleEyeHT. Can be adapted for NTowerHT.
    """
    model.eval()

    emergent_samples = []

    # Access the final linear layer weights in the HTClassifier
    # HTClassifier head is: Sequential(ReLU(), Dropout(), Linear())
    # Index 2 is the Linear layer
    final_linear = model.bridge.classifier_fused.head[2]
    final_weights = (
        final_linear.weight.detach().cpu()
    )  # Shape: [num_classes, fusion_dim]
    final_bias = final_linear.bias.detach().cpu()

    with torch.no_grad():
        for batch in loader:
            x = batch.get("image_1")
            m = batch.get("matrix_1")
            y = batch.get("label_1")

            if not (torch.is_tensor(x) and torch.is_tensor(m) and torch.is_tensor(y)):
                continue

            x, m, y = x.to(device), m.to(device), y.to(device)

            # 1. Get Tower Embeddings
            img_feats = model.img_tower(x)
            md_feats = model.cd_tower(m)

            # 2. Get Independent Tower Predictions
            logits_i = model.bridge.aux_heads[0](img_feats)
            logits_m = model.bridge.aux_heads[1](md_feats)
            pi = logits_i.argmax(dim=1)
            pm = logits_m.argmax(dim=1)

            # 3. Get Fused Representation & Prediction
            # _compute_fused applies the Hadamard product and optional SE gate
            z_fused = model.bridge._compute_fused([img_feats, md_feats])
            logits_fused = model.bridge.classifier_fused(z_fused)
            pf = logits_fused.argmax(dim=1)

            # 4. Find the "Aha!" Moments and "Corrections"
            for i in range(len(y)):
                yi = y[i].cpu().item()
                is_correct = pf[i] == yi
                img_wrong = pi[i] != yi
                md_wrong = pm[i] != yi

                if not is_correct:
                    continue

                is_aha = img_wrong and md_wrong
                is_correction = img_wrong or md_wrong

                if is_correction:
                    category = (
                        "Aha!"
                        if is_aha
                        else ("Corrected Image" if img_wrong else "Corrected Clinical")
                    )
                    # Apply the ReLU that happens inside HTClassifier before the Linear layer
                    z_act = F.relu(z_fused[i]).cpu()

                    # Calculate how much each feature contributed to the correct class logit
                    feature_contributions = z_act * final_weights[yi]

                    emergent_samples.append(
                        {
                            "patient_id": (
                                batch.get("id_1", ["Unknown"])[i]
                                if "id_1" in batch
                                else "Unknown"
                            ),
                            "target_class": yi,
                            "category": category,
                            "z_activated": z_act.numpy(),
                            "contributions": feature_contributions.numpy(),
                            "total_logit": logits_fused[i, yi].cpu().item(),
                        }
                    )

    return emergent_samples


def plot_top_emergent_features(emergent_samples, top_k=10):
    """
    Plots the top K contributing dimensions across all emergent success samples.
    """
    if not emergent_samples:
        print("No emergent success or correction samples found in this pass.")
        return

    ahas = [s for s in emergent_samples if s["category"] == "Aha!"]
    corrected_img = [s for s in emergent_samples if s["category"] == "Corrected Image"]
    corrected_clin = [
        s for s in emergent_samples if s["category"] == "Corrected Clinical"
    ]

    print(f"\nFound {len(emergent_samples)} total events of interest:")
    print(f"  - 'Aha!' Moments (Both wrong, Fused right): {len(ahas)}")
    print(f"  - Corrected Image (Image wrong, Fused right): {len(corrected_img)}")
    print(
        f"  - Corrected Clinical (Clinical wrong, Fused right): {len(corrected_clin)}"
    )

    # Prioritize true Aha moments if they exist, otherwise use corrections
    plot_samples = ahas if ahas else emergent_samples
    plot_title = (
        "Aha! Moments (Both Wrong, Fused Right)"
        if ahas
        else "Fusion Corrections (At least one wrong)"
    )

    # Average the feature contributions across samples
    all_contribs = np.stack([s["contributions"] for s in plot_samples])
    mean_contribs = all_contribs.mean(axis=0)

    # Get indices of the top K features with the highest absolute contribution
    top_indices = np.argsort(np.abs(mean_contribs))[-top_k:][::-1]

    top_values = mean_contribs[top_indices]
    labels = [f"Dim {idx}" for idx in top_indices]

    plt.figure(figsize=(10, 6))
    colors = ["green" if v > 0 else "red" for v in top_values]

    plt.barh(np.arange(top_k), top_values[::-1], color=colors[::-1])
    plt.yticks(np.arange(top_k), labels[::-1])
    plt.xlabel("Mean Contribution to Correct Logit")
    plt.title(f"Top {top_k} Features Driving {plot_title}")
    plt.tight_layout()
    plt.show()

    print(f"\nAnalyzed {len(plot_samples)} samples for this plot.")
    print("\nTop Feature Breakdown:")
    for idx, val in zip(top_indices, top_values):
        print(f"Dimension {idx:3d}: {val:+.4f} average logit push")


def main():
    device = choose_device("auto")
    print(f"Using device: {device}")

    repo_root = Path(__file__).resolve().parents[3]
    image_dir = repo_root / "Papila" / "FundusImages"
    clinical_dir = repo_root / "Papila" / "ClinicalData"

    print("Loading PAPILA data with Phase 5 settings...")
    data = build_papila_data(
        image_dir=str(image_dir),
        clinical_dir=str(clinical_dir),
        label_col="Diagnosis",
        cat_cols=["Gender", "Phakic/Pseudophakic"],
        iop_corr_method="ratio",
        iop_drop_raw=True,
        exclude_cols=["Axial_Length"],
    )
    # Filter to binary
    data.df = data.df[data.df["Diagnosis"].isin([0, 1])].reset_index(drop=True)

    print("Building dataloader (All samples)...")
    profile_eye = build_papila_profile(
        patient_col="Patient ID", label_col="Diagnosis", sample_mode="eye"
    )
    eye_samples = filter_eye_samples(
        profile_eye.build_samples(df=data.df, clinical=data)
    )

    loader = make_loader(
        eye_samples,
        profile_eye.slot_descriptors(),
        image_transform=build_eval_transform("refugelike"),
        batch_size=16,
        shuffle=False,
        num_workers=4,
    )

    print("Building SingleEyeHT model...")
    model = SingleEyeHT(
        backbone="refugelike",
        freeze_ratio=0.0,
        augment=False,
        clinical_data=data,
        num_classes=2,
        cd_hidden_dim=128,
        fusion_dim=256,
        bridge_mode="fused",
    ).to(device)

    base_ckpt_dir = repo_root / "v3" / "results" / "phase5" / "logit_mlp_head_ckpt"
    checkpoints = sorted(list(base_ckpt_dir.rglob("best_single.pt")))

    if not checkpoints:
        print(f"\n[!] No checkpoints found in {base_ckpt_dir}")
        print("Please ensure you ran the jobs with the --save-checkpoints flag.")
        return

    all_emergent_data = []

    for checkpoint_path in checkpoints:
        print(f"\nProcessing {checkpoint_path.relative_to(repo_root)}...")
        state_dict = torch.load(checkpoint_path, map_location=device)

        # Backward compatibility for checkpoints saved before the N-tower bridge refactor
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("bridge.W_img.", "bridge.W.0.")
            k = k.replace("bridge.W_md.", "bridge.W.1.")
            k = k.replace("bridge.ln_img.", "bridge.ln.0.")
            k = k.replace("bridge.ln_md.", "bridge.ln.1.")
            k = k.replace("bridge.classifier_img.", "bridge.aux_heads.0.")
            k = k.replace("bridge.classifier_cd.", "bridge.aux_heads.1.")
            k = k.replace(
                "bridge.classifier_fused.2.", "bridge.classifier_fused.head.2."
            )
            new_state_dict[k] = v

        model.load_state_dict(new_state_dict)

        emergent_data = extract_emergent_features(model, loader, device)
        all_emergent_data.extend(emergent_data)
        print(f"  -> Found {len(emergent_data)} events of interest.")

    print(
        f"\nTotal aggregated events across {len(checkpoints)} checkpoints: {len(all_emergent_data)}"
    )
    plot_top_emergent_features(all_emergent_data, top_k=15)


if __name__ == "__main__":
    main()
