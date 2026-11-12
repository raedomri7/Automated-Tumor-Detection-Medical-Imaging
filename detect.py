import sys
import os
import torch
from torch import nn
from torchvision import transforms, models
from PyQt6 import QtWidgets, QtCore, QtGui
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from io import BytesIO
import numpy as np
import cv2
from datetime import datetime


class BrainTumorApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Brain Tumor Detection - Deep Learning")
        self.setGeometry(100, 100, 1150, 720)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.image = None
        self.pred_label = None
        self.probability = None
        self.file_path = ""
        self.use_sigmoid_in_predict = True
        self.initUI()

    def initUI(self):
        layout = QtWidgets.QVBoxLayout()

        title = QtWidgets.QLabel("Brain Tumor Detection (ResNet50 Fine-Tuned)")
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #00e6ac; margin-bottom: 10px;")
        layout.addWidget(title)

        self.image_label = QtWidgets.QLabel("No image loaded")
        self.image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("border: 2px dashed #00e6ac; border-radius: 10px; min-height: 350px;")
        layout.addWidget(self.image_label)

        btn_layout = QtWidgets.QHBoxLayout()
        self.load_btn = QtWidgets.QPushButton("Load Image")
        self.load_btn.clicked.connect(self.load_image)
        btn_layout.addWidget(self.load_btn)

        self.init_btn = QtWidgets.QPushButton("Load Model")
        self.init_btn.clicked.connect(self.load_model)
        btn_layout.addWidget(self.init_btn)

        self.predict_btn = QtWidgets.QPushButton("Predict")
        self.predict_btn.clicked.connect(self.predict_image)
        self.predict_btn.setEnabled(False)
        btn_layout.addWidget(self.predict_btn)

        self.report_btn = QtWidgets.QPushButton("Generate Medical Report")
        self.report_btn.clicked.connect(self.save_pdf)
        self.report_btn.setEnabled(False)
        btn_layout.addWidget(self.report_btn)

        layout.addLayout(btn_layout)

        self.result_label = QtWidgets.QLabel("Result: —")
        self.result_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.result_label.setStyleSheet("font-size: 16px; color: #ffd166; margin-top: 10px;")
        layout.addWidget(self.result_label)

        self.setLayout(layout)

    def load_image(self):
        file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select an MRI image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if file:
            try:
                self.file_path = file
                self.image = Image.open(file).convert("RGB")
                pixmap = QtGui.QPixmap(file).scaledToWidth(500, QtCore.Qt.TransformationMode.SmoothTransformation)
                self.image_label.setPixmap(pixmap)
                self.result_label.setText("Image loaded successfully.")
                if self.model:
                    self.predict_btn.setEnabled(True)
            except Exception as e:
                self.result_label.setText(f"Error loading image: {str(e)}")

    def load_model(self):
        model_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load model (.pth)", "", "PyTorch Model (*.pth)"
        )
        if not model_path:
            return
        try:
            self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            state_dict = torch.load(model_path, map_location=self.device)

            fc_keys = [k for k in state_dict.keys() if k.startswith("fc")]
            num_features = self.model.fc.in_features
            if any("0." in k for k in fc_keys):
                self.model.fc = nn.Sequential(nn.Linear(num_features, 1), nn.Sigmoid())
                self.use_sigmoid_in_predict = False
            else:
                self.model.fc = nn.Linear(num_features, 1)
                self.use_sigmoid_in_predict = True

            self.model.load_state_dict(state_dict)
            self.model.to(self.device).eval()
            self.result_label.setText("Model loaded successfully.")
            if self.image:
                self.predict_btn.setEnabled(True)
        except Exception as e:
            self.result_label.setText(f"Error loading model: {str(e)}")

    def predict_image(self):
        if not self.image or not self.model:
            self.result_label.setText("Please load an image and a model first.")
            return
        try:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            img_tensor = transform(self.image).unsqueeze(0).to(self.device)

            self.model.eval()
            with torch.no_grad():
                output = self.model(img_tensor)
                prob = output.item()
                if self.use_sigmoid_in_predict:
                    prob = torch.sigmoid(torch.tensor(prob)).item()
            self.probability = prob

            label, color = ("Tumor detected", "#ff4d4d") if prob > 0.5 else ("No tumor detected", "#00e6ac")
            self.pred_label = f"{label} (Confidence: {prob * 100:.2f}%)"
            self.result_label.setText(self.pred_label)
            self.result_label.setStyleSheet(f"font-size:16px; color:{color}; margin-top:10px")

            # Heatmap
            if prob > 0.5:
                features = None

                def hook_fn(module, input, output):
                    nonlocal features
                    features = output

                handle = self.model.layer4.register_forward_hook(hook_fn)
                _ = self.model(img_tensor)
                handle.remove()

                heatmap = features.squeeze(0).mean(0).cpu().detach().numpy()
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)
                heatmap_resized = cv2.resize(heatmap, (self.image.width, self.image.height))
                heatmap_resized = (heatmap_resized * 255).astype(np.uint8)
                heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
                img_cv = np.array(self.image)
                overlay = cv2.addWeighted(img_cv, 0.6, heatmap_color, 0.4, 0)

                _, mask = cv2.threshold(heatmap_resized, 120, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

                img_qt = QtGui.QImage(
                    overlay.data,
                    overlay.shape[1],
                    overlay.shape[0],
                    overlay.strides[0],
                    QtGui.QImage.Format.Format_RGB888,
                )
                self.image_label.setPixmap(
                    QtGui.QPixmap.fromImage(img_qt).scaledToWidth(500, QtCore.Qt.TransformationMode.SmoothTransformation)
                )

            self.report_btn.setEnabled(True)
        except Exception as e:
            self.result_label.setText(f"Prediction error: {str(e)}")

    def save_pdf(self):
        if not self.file_path or not self.pred_label:
            self.result_label.setText("No result to save.")
            return

        pdf_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save PDF", "", "PDF Files (*.pdf)"
        )
        if not pdf_path:
            return
        if not pdf_path.lower().endswith(".pdf"):
            pdf_path += ".pdf"

        try:
            c = canvas.Canvas(pdf_path, pagesize=A4)
            width, height = A4

            # HEADER
            c.setFont("Helvetica-Bold", 22)
            c.drawCentredString(width / 2, height - 50, "Medical Report: Brain Tumor Detection")
            c.line(50, height - 60, width - 50, height - 60)

            # METADATA
            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, height - 100, "Analysis Information:")
            c.setFont("Helvetica", 12)
            c.drawString(70, height - 120, f"Analyzed image: {os.path.basename(self.file_path)}")
            c.drawString(70, height - 140, f"Result: {self.pred_label}")
            c.drawString(70, height - 160, f"Probability: {self.probability * 100:.2f}%")
            c.drawString(70, height - 180, "Model: ResNet50 Fine-Tuned (ImageNet)")
            c.drawString(70, height - 200, f"Date / Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            # ANNOTATED IMAGE
            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, height - 240, "Annotated Image:")
            pixmap = self.image_label.pixmap()
            arr = self.qimage_to_numpy(pixmap.toImage())
            img_buffer = BytesIO()
            Image.fromarray(arr).save(img_buffer, format="PNG")
            img_buffer.seek(0)
            img_width = 400
            img_height = int(400 * arr.shape[0] / arr.shape[1])
            c.drawImage(ImageReader(img_buffer), 50, height - 400 - img_height, width=img_width, height=img_height)

            # MEDICAL RECOMMENDATIONS
            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, height - 440 - img_height, "Medical Recommendations:")
            c.setFont("Helvetica", 12)
            c.drawString(70, height - 460 - img_height, "- Any detected abnormalities should be reviewed by a radiologist specialized in brain MRI.")
            c.drawString(70, height - 480 - img_height, "- Verify the exact location and size of the lesion on MRI slices.")
            c.drawString(70, height - 500 - img_height, "- Use T1, T2, and FLAIR sequences for a comprehensive assessment.")
            c.drawString(70, height - 520 - img_height, "- This report is indicative and does not replace professional medical advice.")

            # FOOTER
            c.setFont("Helvetica-Oblique", 10)
            c.drawCentredString(width / 2, 30, "Report automatically generated by BrainTumorApp - Deep Learning")

            c.showPage()
            c.save()
            self.result_label.setText("Medical PDF report saved successfully.")

        except Exception as e:
            self.result_label.setText(f"PDF error: {str(e)}")

    @staticmethod
    def qimage_to_numpy(qimage):
        qimage = qimage.convertToFormat(QtGui.QImage.Format.Format_RGB888)
        width = qimage.width()
        height = qimage.height()
        ptr = qimage.bits()
        ptr.setsize(height * width * 3)
        arr = np.array(ptr, dtype=np.uint8).reshape((height, width, 3))
        return arr


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = BrainTumorApp()
    window.show()
    sys.exit(app.exec())
