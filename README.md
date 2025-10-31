# DEGE Fire Detection

Real-time **fire vs. no-fire** detection with **PyTorch** and **OpenCV**.  
Train a ResNet-50 classifier on your images and run **live webcam inference** with frame enhancement, confidence smoothing, and a bold on-screen alert.

---

## Features

-  **ResNet-50 transfer learning** (ImageNet pretrained) for binary fire detection
-  **Data augmentation & normalization** (resize, flip, rotate, color jitter)
-  **Best-checkpoint saving** (`fire_detection_model.pth`) with early stopping
-  **Webcam inference** with CLAHE enhancement and overlay UI
-  **Stabilized output**: confidence history, consecutive detections, cooldown
-  **CLI flags** for camera index, thresholds, image size, batch size, etc.

---

## Repository Structure

```
.
├── fire_detection.py        # Train / evaluate the model
├── live_fire_detection.py   # Real-time webcam inference
├── prepare_dataset.py       # Dataset loader utilities
├── README.md
├── LICENSE
└── (optional) test_images/  # Put sample images here for quick checks
```

Expected dataset layout:

```
dataset/
├── fire/
│   ├── img_001.jpg
│   └── ...
└── no_fire/
    ├── img_101.jpg
    └── ...
```

---

## Requirements

- Python 3.9+ (3.10 recommended)
- PyTorch (+ CUDA if available), torchvision
- OpenCV-Python
- NumPy, Pillow, tqdm

Install (CPU example):

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python numpy pillow tqdm
```

> **GPU:** Install the CUDA-specific PyTorch build from pytorch.org, then keep the rest the same.

---

## Quick Start

### 1) Prepare data

Place your images under `dataset/fire` and `dataset/no_fire`.  
(You can add your own splits; the scripts handle train/val via dataset arguments.)

### 2) Train

```bash
python fire_detection.py   --data_root dataset   --epochs 15   --batch_size 32   --lr 3e-4   --img_size 224   --patience 5   --save_path fire_detection_model.pth
```

Common flags:
- `--data_root` path to the dataset folder
- `--img_size` image side length (default 224)
- `--epochs`, `--batch_size`, `--lr` training hyperparameters
- `--patience` early stopping patience (epochs without val improvement)
- `--save_path` where to store the best model

### 3) Run live detection (webcam)

```bash
python live_fire_detection.py   --model fire_detection_model.pth   --camera 0   --img_size 224   --min_conf 0.6   --consecutive 3   --cooldown 1.5
```

Useful flags:
- `--camera` webcam index (0, 1, …)
- `--min_conf` minimum probability to consider “fire”
- `--consecutive` number of consecutive frames above threshold required
- `--cooldown` seconds to wait before re-triggering alert
- `--width` / `--height` (optional) request capture resolution

---

## Tips & Notes

- **Data balance** helps. If classes are imbalanced, consider weighted loss or oversampling.
- Increase `--img_size` (e.g., 256) or **improve lighting** for better detection.
- For reproducibility, fix seeds and log hyperparameters you use.
- For deployment, export to TorchScript/ONNX and integrate in your app pipeline.

---

## Troubleshooting

- **CUDA not used:** Ensure you installed the CUDA-enabled PyTorch wheel and that `torch.cuda.is_available()` returns `True`.
- **Webcam not opening:** Try a different `--camera` index and close other apps using the camera.
- **Flickering alerts:** Raise `--consecutive`, increase `--cooldown`, or set a higher `--min_conf`.
- **Slow training:** Lower image size (`--img_size 192`), reduce batch size, or freeze more backbone layers.

---

## License

This project is released under the **MIT License**.  
© 2024 Dogan Ege BULTE


