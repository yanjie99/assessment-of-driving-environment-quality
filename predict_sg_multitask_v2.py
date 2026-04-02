import os
import csv
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from transformers import SegformerModel


# 1. Label settings

IGNORE_INDEX = 255

SEG_ID2LABEL = {
    0: "road",
    1: "sidewalk",
    2: "building",
    3: "wall",
    4: "fence",
    5: "pole",
    6: "traffic light",
    7: "traffic sign",
    8: "vegetation",
    9: "terrain",
    10: "sky",
    11: "person",
    12: "rider",
    13: "car",
    14: "truck",
    15: "bus",
    16: "train",
    17: "motorcycle",
    18: "bicycle"
}

NUM_SEG_CLASSES = len(SEG_ID2LABEL)

SCENE_ID2LABEL = {
    0: "low",
    1: "moderate",
    2: "high"
}

NUM_SCENE_CLASSES = len(SCENE_ID2LABEL)


CITYSCAPES_COLORS = np.array([
    [128, 64, 128],   # road
    [244, 35, 232],   # sidewalk
    [70, 70, 70],     # building
    [102, 102, 156],  # wall
    [190, 153, 153],  # fence
    [153, 153, 153],  # pole
    [250, 170, 30],   # traffic light
    [220, 220, 0],    # traffic sign
    [107, 142, 35],   # vegetation
    [152, 251, 152],  # terrain
    [70, 130, 180],   # sky
    [220, 20, 60],    # person
    [255, 0, 0],      # rider
    [0, 0, 142],      # car
    [0, 0, 70],       # truck
    [0, 60, 100],     # bus
    [0, 80, 100],     # train
    [0, 0, 230],      # motorcycle
    [119, 11, 32],    # bicycle
], dtype=np.uint8)


# 2. Model definition

class SegFormerDecodeHead(nn.Module):
    def __init__(self, in_channels, embedding_dim, num_classes, dropout=0.1):
        super().__init__()

        self.proj_layers = nn.ModuleList([
            nn.Conv2d(ch, embedding_dim, kernel_size=1)
            for ch in in_channels
        ])

        self.fuse = nn.Sequential(
            nn.Conv2d(len(in_channels) * embedding_dim, embedding_dim, kernel_size=1),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        self.classifier = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)

    def forward(self, features):
        target_size = features[0].shape[-2:]
        projected = []

        for feat, proj in zip(features, self.proj_layers):
            x = proj(feat)
            if x.shape[-2:] != target_size:
                x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
            projected.append(x)

        x = torch.cat(projected, dim=1)
        x = self.fuse(x)
        logits = self.classifier(x)
        return logits


class SegFormerMultiTaskV2(nn.Module):
    def __init__(
        self,
        num_seg_classes,
        num_scene_classes,
        backbone_name="nvidia/mit-b0",
        decoder_embed_dim=256,
        cls_hidden_dim=256,
        cls_dropout=0.2
    ):
        super().__init__()

        self.encoder = SegformerModel.from_pretrained(backbone_name)
        hidden_sizes = self.encoder.config.hidden_sizes

        self.seg_decoder = SegFormerDecodeHead(
            in_channels=hidden_sizes,
            embedding_dim=decoder_embed_dim,
            num_classes=num_seg_classes,
            dropout=0.1
        )

        deep_channels = hidden_sizes[-1]

        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(deep_channels, cls_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(cls_dropout),
            nn.Linear(cls_hidden_dim, num_scene_classes)
        )

    def forward(self, pixel_values):
        outputs = self.encoder(pixel_values=pixel_values, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        seg_logits = self.seg_decoder(list(hidden_states))
        deep_feat = hidden_states[-1]
        cls_logits = self.cls_head(deep_feat)

        return {
            "seg_logits": seg_logits,
            "cls_logits": cls_logits
        }



# 3. Utility functions

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def decode_segmap(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id in range(NUM_SEG_CLASSES):
        color_mask[mask == class_id] = CITYSCAPES_COLORS[class_id]

    return color_mask


def overlay_mask_on_image(image: np.ndarray, color_mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    overlay = (1 - alpha) * image + alpha * (color_mask / 255.0)
    overlay = np.clip(overlay, 0, 1)
    return (overlay * 255).astype(np.uint8)


def compute_class_proportions(pred_mask: np.ndarray) -> dict:
    total_pixels = pred_mask.size
    result = {}

    for class_id, class_name in SEG_ID2LABEL.items():
        class_pixels = np.sum(pred_mask == class_id)
        result[f"prop_{class_name}"] = class_pixels / total_pixels

    return result


def load_image(image_path: Path, image_size=(512, 1024)):
    image = Image.open(image_path).convert("RGB")
    original_size = image.size  # (W, H)

    resized = TF.resize(image, image_size, interpolation=InterpolationMode.BILINEAR)
    tensor = TF.to_tensor(resized)
    tensor = TF.normalize(
        tensor,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    image_np = np.array(resized).astype(np.uint8)
    return tensor, image_np, original_size


def save_mask_png(mask: np.ndarray, output_path: Path) -> None:
    Image.fromarray(mask.astype(np.uint8)).save(output_path)


def save_color_png(color_mask: np.ndarray, output_path: Path) -> None:
    Image.fromarray(color_mask).save(output_path)


# 4. Prediction function

@torch.no_grad()
def predict_one_image(
    model: nn.Module,
    image_path: Path,
    device: torch.device,
    image_size=(512, 1024)
):
    image_tensor, image_np, original_size = load_image(image_path, image_size=image_size)

    pixel_values = image_tensor.unsqueeze(0).to(device)

    outputs = model(pixel_values)
    seg_logits = outputs["seg_logits"]
    cls_logits = outputs["cls_logits"]

    seg_logits = F.interpolate(
        seg_logits,
        size=(image_size[0], image_size[1]),
        mode="bilinear",
        align_corners=False
    )

    pred_mask = torch.argmax(seg_logits, dim=1).squeeze(0).cpu().numpy()

    cls_probs = torch.softmax(cls_logits, dim=1).squeeze(0).cpu().numpy()
    pred_scene_id = int(np.argmax(cls_probs))
    pred_scene_label = SCENE_ID2LABEL[pred_scene_id]

    color_mask = decode_segmap(pred_mask)
    overlay = overlay_mask_on_image(image_np / 255.0, color_mask, alpha=0.5)

    return {
        "image_name": image_path.name,
        "pred_mask": pred_mask,
        "color_mask": color_mask,
        "overlay": overlay,
        "scene_id": pred_scene_id,
        "scene_label": pred_scene_label,
        "scene_probs": cls_probs,
        "class_proportions": compute_class_proportions(pred_mask),
        "original_size": original_size,
    }


# 5. Main batch inference

def run_inference(
    input_dir: Path,
    output_dir: Path,
    checkpoint_path: Path,
    image_size=(512, 1024),
    backbone_name="nvidia/mit-b0"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Output folders
    mask_dir = output_dir / "pred_masks_raw"
    color_dir = output_dir / "pred_masks_color"
    overlay_dir = output_dir / "pred_overlays"

    ensure_dir(output_dir)
    ensure_dir(mask_dir)
    ensure_dir(color_dir)
    ensure_dir(overlay_dir)

    # Load model
    model = SegFormerMultiTaskV2(
        num_seg_classes=NUM_SEG_CLASSES,
        num_scene_classes=NUM_SCENE_CLASSES,
        backbone_name=backbone_name,
        decoder_embed_dim=256,
        cls_hidden_dim=256,
        cls_dropout=0.2
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Loaded checkpoint: {checkpoint_path}")

    # Find images
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    image_paths = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in valid_exts])

    if len(image_paths) == 0:
        raise FileNotFoundError(f"No supported images found in {input_dir}")

    print(f"Found {len(image_paths)} images.")

    # CSV output
    csv_path = output_dir / "prediction_summary.csv"
    rows = []

    for idx, image_path in enumerate(image_paths, start=1):
        print(f"[{idx}/{len(image_paths)}] Predicting: {image_path.name}")

        result = predict_one_image(
            model=model,
            image_path=image_path,
            device=device,
            image_size=image_size
        )

        stem = image_path.stem

        # Save outputs
        save_mask_png(result["pred_mask"], mask_dir / f"{stem}_pred_mask.png")
        save_color_png(result["color_mask"], color_dir / f"{stem}_pred_color.png")
        save_color_png(result["overlay"], overlay_dir / f"{stem}_overlay.png")

        row = {
            "image_name": result["image_name"],
            "scene_id": result["scene_id"],
            "scene_label": result["scene_label"],
            "prob_low": float(result["scene_probs"][0]),
            "prob_moderate": float(result["scene_probs"][1]),
            "prob_high": float(result["scene_probs"][2]),
        }

        row.update(result["class_proportions"])
        rows.append(row)

    # Save CSV
    if len(rows) > 0:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(f"\nDone. Results saved to: {output_dir}")
    print(f"CSV summary: {csv_path}")


# 6. CLI

def parse_args():
    parser = argparse.ArgumentParser(description="Predict Singapore images using trained SegFormer Multi-task V2 model.")
    parser.add_argument("--input_dir", type=str, required=True, help="Folder containing input images.")
    parser.add_argument("--output_dir", type=str, required=True, help="Folder to save prediction outputs.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained .pth checkpoint.")
    parser.add_argument("--img_h", type=int, default=512, help="Inference resize height.")
    parser.add_argument("--img_w", type=int, default=1024, help="Inference resize width.")
    parser.add_argument("--backbone", type=str, default="nvidia/mit-b0", help="SegFormer backbone name.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_inference(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        checkpoint_path=Path(args.checkpoint),
        image_size=(args.img_h, args.img_w),
        backbone_name=args.backbone
    )