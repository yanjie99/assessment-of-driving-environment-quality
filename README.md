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

```
</> Bash
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

## Input image requirements

The input folder can contain common image formats:
- `.png`
- `.jpg`
- `.jpeg`
- `.bmp`
- `.webp`

**Image size**

The original image size does not need to `512x1024`.

The script will resize each image to the specified inference size, by default:
- height=`512`
- width=`1024`

However, is the aspect ratio is very different from the training images, the prediction quality may drop.

## How to run the script

**Example command**

```bash
python predict_singapore_multitask_v2.py \
    --input_dir singapore_images \
    --output_dir singapore_predictions \
    --checkpoint segformer_multitask_quality_best_v2.pth \
    --img_h 512 \
    --img_w 1024
```

## Expected outputs

The script creates the following folders automatically:

```
singapore_predictions/
├── pred_masks_raw/
├── pred_masks_color/
├── pred_overlays/
└── prediction_summary.csv
```

`pred_masks_raw/` contains raw segmentation masks.

- each pixel value is the predicted class ID
- useful for later analysis

`pred_masks_color/` contains colorized segmentation masks.

- useful for quick visualization

`pred_overlays/` contains overlay images.

- segmentation mask blended with the RGB image.

`prediction_summary.csv` contains one row per image with:

- image name
- predicted scene quality class
- propabilities for low/moderate/high quality
- class proportions from segmentation output

*This CSV is especially useful for later spatial analysis*

## Meaning of the classification labels

The script predicts one of three image-level classes: `0 low` `1 moderate` `2 high`.
These labels represent visual driving-environment quality proxies derived from the training design.

**Interpretation**

- low quality: visually harsher, less green, more crowded/complex road-user environment
- moderate quality: balanced condition
- high: greener, calmer, less visually complex

**These labels are model-derived visual proxies, not directly safety ratings or official road quality assessments.**

## Segmentation class mapping

The segmentation mask uses the following class IDs:

```
| ID | Label         |
| -: | ------------- |
|  0 | road          |
|  1 | sidewalk      |
|  2 | building      |
|  3 | wall          |
|  4 | fence         |
|  5 | pole          |
|  6 | traffic light |
|  7 | traffic sign  |
|  8 | vegetation    |
|  9 | terrain       |
| 10 | sky           |
| 11 | person        |
| 12 | rider         |
| 13 | car           |
| 14 | truck         |
| 15 | bus           |
| 16 | train         |
| 17 | motorcycle    |
| 18 | bicycle       |
```


