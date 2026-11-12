# Automated Brain Tumor Detection from Medical Images

Desktop application (PyQt6) for analyzing brain MRI images using a deep learning model (fine-tuned ResNet50). It lets you load an image, run a binary prediction (tumor / no tumor), display a heatmap overlay when a tumor is detected, and generate a PDF report.

## Repository contents

- `detect.py` — GUI and prediction/report logic
- `brain_tumor_model.pth` — PyTorch model weights (provided locally)
- `tumeurs_guides.png` — illustration/guide image
- `Simulation.mp4` — project demonstration
- `.venv/`, `.idea/` — local environment/IDE settings (not required for the source code)

## Features

- Load MRI images (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`)
- Load a PyTorch `.pth` model from the UI
- Binary prediction with a confidence score
- Heatmap and contour visualization over activated regions (when a tumor is detected)
- PDF report generation (annotated image, result, probability, date/time, general recommendations)
- Automatic use of CUDA GPU when available, otherwise CPU

## Requirements

- Python 3.11+ (Windows environment as tested)
- Main dependencies: `torch`, `torchvision`, `PyQt6`, `Pillow`, `reportlab`, `numpy`, `opencv-python`

Example setup (virtual environment recommended):

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision PyQt6 Pillow reportlab numpy opencv-python
```

> Note: PyTorch installation can vary depending on CPU/CUDA configuration. Refer to the official PyTorch documentation if needed.

## Usage

1. Activate a Python environment with the required dependencies.
2. Start the application:
   ```bash
   python detect.py
   ```
3. In the interface:
   - **Load Image**: select an MRI image.
   - **Load Model**: select the `brain_tumor_model.pth` file.
   - **Predict**: run the analysis and view the result/heatmap.
   - **Generate Medical Report**: export a PDF report.

## Technical details

- Architecture: `ResNet50` (ImageNet-pretrained base) with a final layer adapted for binary output.
- Preprocessing: resize to `224×224`, ImageNet normalization.
- Output: sigmoid probability with a decision threshold of `0.5`.
- Visual localization: mean activation from `layer4` projected as a heatmap onto the image.

## Disclaimer

This project is a demonstration/research tool. Results are indicative and **do not constitute a medical diagnosis**. Any clinical interpretation must be performed by a qualified healthcare professional.
