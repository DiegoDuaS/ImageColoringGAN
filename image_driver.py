import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from skimage.color import rgb2lab

class ColorizationDataset(Dataset):
    def __init__(self, root_dir, num_classes=8, image_size=128):
        self.root_dir = root_dir
        self.image_paths = [
            os.path.join(root_dir, fname)
            for fname in os.listdir(root_dir)
            if fname.endswith(".jpg")
        ]
        self.num_classes = num_classes
        self.resize = transforms.Resize((image_size, image_size))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        fname = os.path.basename(img_path)


        # Parse label from 'color_<id>_<label>.jpg'

        parts = fname.split("_")
        if len(parts) < 3:
            raise ValueError(f"Filename '{fname}' does not match pattern 'color_<id>_<label>.jpg'")
        label_name = parts[2].split(".")[0]


        if not hasattr(self, "label_to_idx"):
            label_names = sorted({os.path.basename(p).split("_")[2].split(".")[0] for p in self.image_paths})
            self.label_to_idx = {name: i for i, name in enumerate(label_names)}
            self.idx_to_label = {i: name for name, i in self.label_to_idx.items()}

        label_idx = self.label_to_idx[label_name]

        # One-hot encode the label
        label_onehot = torch.zeros(self.num_classes, dtype=torch.float32)
        label_onehot[label_idx] = 1.0

        img = Image.open(img_path).convert("RGB")
        img = self.resize(img)
        img_np = np.array(img) / 255.0

        # Convert to LAB
        lab = rgb2lab(img_np).astype(np.float32)
        lab_tensor = torch.from_numpy(lab.transpose((2, 0, 1)))  # (3, H, W)

        # Split channels
        L = lab_tensor[0:1, :, :] / 50.0 - 1.0 
        ab = lab_tensor[1:3, :, :] / 110.0

        return L, ab, label_onehot
