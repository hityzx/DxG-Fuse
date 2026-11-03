# DxG-Fuse

Official PyTorch implementation of **DxG-Fuse: Description-Guided Fusion for Multimodal Fake News Detection via MLLMs**.

## 1. Environment Setup

Run the following command from the project directory:

```bash
pip install -r requirements.txt
```

## 2. Pretrained Models

Download [bert-base-uncased](https://huggingface.co/bert-base-uncased) and [google/vit-base-patch16-224-in21k](https://huggingface.co/google/vit-base-patch16-224-in21k), including their weights and configuration files, into `pretrained/`:

```text
pretrained/
├── bert-base-uncased/
└── vit/
```

## 3. Train the Forgery Detector

Obtain the detector training data from [FakeShield](https://github.com/zhipeixu/FakeShield) and organize the images under `DxG-Fuse/data/fd/`:

```text
data/fd/
├── train/
│   ├── normal/
│   ├── deepfake/
│   └── photoshop/
└── val/
    ├── normal/
    ├── deepfake/
    └── photoshop/
```

Train the detector (the pretrained ResNet-34 weights are downloaded automatically):

```bash
python train_fd.py
```

Then use the saved `best_fd_model.pth` to add a `forgery` field (0 = original, 1 = forged) to each JSON file in place. Each record must contain `image_path`:

```bash
for split in train val test; do
    python train_fd.py --input_json "data/{dataset}/${split}.json"
done
```

## 4. Generate Multi-Perspective Descriptions

Take the [AMG](https://github.com/mazihan880/AMG-An-Attributing-Multi-modal-Fake-News-Dataset) dataset based on the Llama-3.2-Vision model as an example, place the dataset splits at `data/amg/train.json`, `data/amg/val.json`, and `data/amg/test.json`. Relative image paths are resolved from the directory containing the JSON file.

Prepare a local Llama 3.2 Vision model and run:

```bash
for split in train val test; do
    python multi_perspective.py \
        --input_json "data/amg/${split}.json" \
        --model_path /path/to/llama3.2-vision \
        --device cuda:0
done
```

This adds multi-perspective rationales, predictions, and image descriptions to each JSON file in place.

## 5. Training

Start training on AMG:

```bash
bash run_amg.sh
```
