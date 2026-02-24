from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch


class ClinicalDataset(Dataset):
    """Generic dataset wrapping a DataBundle-like instance.
    Returns (img_tensor, meta_tensor, label)."""

    def __init__(
        self,
        clinical_data,
        img_transform,
        meta_transform=None,
        image_preprocessor=None,
        geometry_provider=None,
        geometry_dim: int = 0,
    ):
        self.clinical = clinical_data
        self.transform_image = img_transform
        self.meta_transform = meta_transform or (lambda x: x)
        self.image_preprocessor = image_preprocessor
        self.geometry_provider = geometry_provider
        self.geometry_dim = geometry_dim if geometry_provider is not None else 0

    def __len__(self):
        return len(self.clinical.df)

    def __getitem__(self, idx: int):
        row = self.clinical.df.iloc[idx]
        # load & transform image
        img_path = self.clinical.get_image_path(row)
        orig_img = Image.open(img_path).convert("RGB")
        img = orig_img
        if self.image_preprocessor is not None:
            img = self.image_preprocessor(img, img_path)
        img_t = self.transform_image(img)
        # encode & transform metadata
        meta = self.clinical.encode_metadata(row)
        meta_t = self.meta_transform(meta)
        # label
        label = self.clinical.get_label(row)
        if self.geometry_dim > 0:
            features = None
            if self.geometry_provider is not None and hasattr(self.geometry_provider, "geometry_features"):
                features = self.geometry_provider.geometry_features(orig_img, img_path)
            if features is None:
                geom_vec = torch.zeros(self.geometry_dim, dtype=torch.float32)
            else:
                features = np.asarray(features, dtype=np.float32)
                if features.shape[0] != self.geometry_dim:
                    geom_vec = torch.zeros(self.geometry_dim, dtype=torch.float32)
                else:
                    geom_vec = torch.from_numpy(features)
            return img_t, meta_t, geom_vec, label
        return img_t, meta_t, label
