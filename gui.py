from PyQt5.QtWidgets import (
    QApplication, QLabel, QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout, QWidget
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import torch
import timm
from torchvision import transforms
from PIL import Image
import sys
import os
import datetime
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

class AnimalClassifier(QWidget):
    from PyQt5.QtWidgets import QMessageBox

    def load_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Klasör Seç")
        if folder_path:
            image_files = [f for f in os.listdir(folder_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

            total_confidence = 0
            processed_images = 0

            for image_name in image_files:
                full_path = os.path.join(folder_path, image_name)
                try:
                    image = Image.open(full_path).convert("RGB")
                except:
                    continue  # geçersiz dosyaları atla

                input_tensor = self.transform(image).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    output = self.model(input_tensor)
                    probabilities = torch.nn.functional.softmax(output[0], dim=0)
                    pred_index = torch.argmax(probabilities).item()
                    confidence = probabilities[pred_index].item()

                predicted_class = self.class_names[pred_index]
                timestamp = datetime.datetime.now().strftime("%H:%M:%S")

                self.prediction_history.append(
                    f"[{timestamp}] {predicted_class} (%{confidence * 100:.2f}) - {image_name}"
                )
                self.history_label.setText(
                    "<b>📋 Geçmiş Tahminler:</b><br>" + "<br>".join(self.prediction_history[-6:])
                )

                save_name = f"{predicted_class}_{confidence*100:.2f}_{image_name}".replace(" ", "_")
                save_path = os.path.join("saved", save_name)
                image.save(save_path)

                self.confidences.append(confidence * 100)
                total_confidence += confidence * 100
                processed_images += 1

            self.update_plot()

            if processed_images > 0:
                avg_accuracy = total_confidence / processed_images
                self.QMessageBox.information(self, "Ortalama Güven Skoru", f"{processed_images} görsel işlendi.\n\nOrtalama Tahmin Başarısı: %{avg_accuracy:.2f}")
            else:
                self.QMessageBox.warning(self, "Uyarı", "Klasörde geçerli görsel bulunamadı.")

    def __init__(self):
        super().__init__()
        self.setWindowTitle("🐾 Hayvan Türü Tanıma")
        self.setFixedSize(900, 550)
        self.setStyleSheet("background-color: #f0f2f5; font-family: Segoe UI;")

        self.confidences = []
        self.prediction_history = []

        # === Sol Panel: Görsel ve Tahmin ===
        self.image_label = QLabel("📷 Görsel seçilmedi")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(360, 360)
        self.image_label.setStyleSheet("""
            QLabel {
                border: 2px dashed #bbb;
                background-color: white;
                font-size: 14px;
                color: #999;
            }
        """)
        self.image_label.setScaledContents(True)

        self.load_button = QPushButton("📁 Görsel Yükle")
        self.load_button.setCursor(Qt.PointingHandCursor)
        self.load_button.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                padding: 10px;
                font-size: 15px;
                border-radius: 6px;
                margin-top: 10px;
            }
            QPushButton:hover {
                background-color: #219150;
            }
        """)
        self.load_button.clicked.connect(self.load_image)

        self.result_label = QLabel("")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("""
            QLabel {
                background-color: #fff;
                border: 1px solid #ccc;
                border-radius: 10px;
                padding: 10px;
                font-size: 15px;
                margin-top: 10px;
            }
        """)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.image_label)
        left_layout.addWidget(self.load_button, alignment=Qt.AlignCenter)
        left_layout.addWidget(self.result_label)
        left_layout.addStretch()

        # === Sağ Panel: Grafik ve Geçmiş ===
        self.figure, self.ax = plt.subplots(figsize=(4.5, 2.5))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet("border: 1px solid #ccc; background: white; margin-top: 5px;")

        self.history_label = QLabel("<b>📋 Geçmiş Tahminler:</b>")
        self.history_label.setAlignment(Qt.AlignLeft)
        self.history_label.setWordWrap(True)
        self.history_label.setStyleSheet("""
            QLabel {
                font-size: 13px;
                color: #333;
                background-color: #fff;
                border: 1px solid #ccc;
                border-radius: 8px;
                padding: 10px;
                margin-top: 10px;
            }
            
        """)

        self.folder_button = QPushButton("📂 Klasördeki Görselleri Test Et")
        self.folder_button.setCursor(Qt.PointingHandCursor)
        self.folder_button.setStyleSheet("""
            QPushButton {
                background-color: #2980b9;
                color: white;
                padding: 10px;
                font-size: 15px;
                border-radius: 6px;
                margin-top: 5px;
            }
            QPushButton:hover {
                background-color: #1f6aa5;
            }
        """)
        self.folder_button.clicked.connect(self.load_folder)
        left_layout.addWidget(self.folder_button, alignment=Qt.AlignCenter)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.canvas)
        right_layout.addWidget(self.history_label)
        right_layout.addStretch()

        # === Alt Bilgi ===
        self.footer_label = QLabel("YAZLAB 3 | 2024-2025 | Yiğit Dikbaş - Şenol Kumaş")
        self.footer_label.setAlignment(Qt.AlignCenter)
        self.footer_label.setStyleSheet("color: #888; font-size: 12px;")

        # === Ana Layout ===
        main_layout = QVBoxLayout()
        h_layout = QHBoxLayout()
        h_layout.addLayout(left_layout, 1)
        h_layout.addLayout(right_layout, 1)
        main_layout.addLayout(h_layout)
        main_layout.addWidget(self.footer_label)
        self.setLayout(main_layout)

        # Model yükleme
        self.class_names = ['antelope', 'badger', 'bat', 'bear', 'bee', 'beetle', 'bison', 'boar', 'butterfly',
                    'cat', 'caterpillar', 'chimpanzee', 'cockroach', 'cow', 'coyote', 'crab', 'crow',
                    'deer', 'dog', 'dolphin', 'donkey', 'dragonfly', 'duck', 'eagle', 'elephant',
                    'flamingo', 'fly', 'fox', 'goat', 'goldfish', 'goose', 'gorilla', 'grasshopper',
                    'hamster', 'hare', 'hedgehog', 'hippopotamus', 'hornbill', 'horse', 'hummingbird',
                    'hyena', 'jellyfish', 'kangaroo', 'koala', 'ladybugs', 'leopard', 'lion', 'lizard',
                    'lobster', 'mosquito', 'moth', 'mouse', 'octopus', 'okapi', 'orangutan', 'otter',
                    'owl', 'ox', 'oyster', 'panda', 'parrot', 'pelecaniformes', 'penguin', 'pig',
                    'pigeon', 'porcupine', 'possum', 'raccoon', 'rat', 'reindeer', 'rhinoceros',
                    'sandpiper', 'seahorse', 'seal', 'shark', 'sheep', 'snake', 'sparrow', 'squid',
                    'squirrel', 'starfish', 'swan', 'tiger', 'turkey', 'turtle', 'whale', 'wolf',
                    'wombat', 'woodpecker', 'zebra']

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = timm.create_model('swin_base_patch4_window7_224', pretrained=False, num_classes=len(self.class_names))
        model_path = os.path.join(os.path.dirname(__file__), "swin_model.pth")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        os.makedirs("saved", exist_ok=True)

    def update_plot(self):
        self.ax.clear()
        self.ax.plot(self.confidences, marker='o', color='#2980b9')
        self.ax.set_title("Tahmin Güven Eğrisi")
        self.ax.set_xlabel("Tahmin Sırası")
        self.ax.set_ylabel("Güven (%)")
        self.ax.set_ylim(0, 105)
        self.ax.grid(True, linestyle='--', alpha=0.3)
        self.canvas.draw()

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Görsel Seç", "", "Image Files (*.png *.jpg *.jpeg)")
        if path:
            pixmap = QPixmap(path)
            self.image_label.setPixmap(pixmap)

            image = Image.open(path).convert("RGB")
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                pred_index = torch.argmax(probabilities).item()
                confidence = probabilities[pred_index].item()

            predicted_class = self.class_names[pred_index]
            color = "#27ae60" if confidence >= 0.90 else "#f39c12" if confidence >= 0.70 else "#c0392b"

            self.result_label.setText(f"""
                <b style='color:#333;'>Tahmin:</b> <span style='color:{color}; font-weight:bold'>{predicted_class}</span><br>
                <b style='color:#333;'>Güven:</b> <span style='color:{color};'>%{confidence*100:.2f}</span>
            """)

            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            self.prediction_history.append(f"[{timestamp}] {predicted_class} (%{confidence*100:.2f})")
            self.history_label.setText("<b>📋 Geçmiş Tahminler:</b><br>" + "<br>".join(self.prediction_history[-6:]))

            image_name = f"{predicted_class}_{confidence*100:.2f}.jpg".replace(" ", "_")
            save_path = os.path.join("saved", image_name)
            image.save(save_path)

            self.confidences.append(confidence * 100)
            self.update_plot()
            

    

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AnimalClassifier()
    window.show()
    sys.exit(app.exec_())
