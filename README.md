# Singapore Street-View Inference with SegFormer Multi-Task V2

This script `predict_singapore_multitask_v2.py` applies the trained **SegFormer Multi-Task V2** model to a new image dataset, such as Singapore street view imagery from driver perspective.

It predicts:

1. **Semantic segmentation**
   - road
   - sidewalk
   - building
   - vegetation
   - sky
   - vehicles
   - pedestrians
   - and other Cityscapes-style classes

2. **Image-level visual driving-environment quality**
   - `low`
   - `moderate`
   - `high`

It also saves:
- raw segmentation masks
- colorized segmentation masks
- overlay images
- a CSV summary containing scene-quality probabilities and class proportions

---

## Files required before running

Before running the script, make sure the following files are prepared.

### Required files

- `predict_singapore_multitask_v2.py`
- trained checkpoint file, for example:
  - `segformer_multitask_quality_best_v2.pth`
- a folder containing input images, for example:
  - `singapore_images/`

### Example structure

```text
project_folder/
├── predict_singapore_multitask_v2.py
├── segformer_multitask_quality_best_v2.pth
└── singapore_images/
    ├── xxx.jpg
    ├── xxx.jpg
    ├── xxx.png
    └── ...
```

## Environment setup

Python `3.10`

**Create a conda environment**

```bash
conda create -n geoai python=3.10 -y
conda activate geoai
```

**Install required packages**

```bash
pip install torch torchvision torchaudio
pip install transformers
pip install pillow numpy
```

## GPU/CPU note

The script can run on:
- GPU if CUDA is available
- CPU otherwise

GPU is recommended because inference will be much faster.

The script automatically detects the device with
```python
torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

