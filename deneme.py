import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import timm

# === Ayarlar ===
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Veri Yolu ve Dönüşüm ===
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_dataset = datasets.ImageFolder("val", transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# === Model Yükle ===
num_classes = len(val_dataset.classes)
model = timm.create_model('swin_base_patch4_window7_224', pretrained=False, num_classes=num_classes)
model.load_state_dict(torch.load("swin_model.pth", map_location=device))
model.to(device)
model.eval()

# === Tahmin ve Metrikler ===
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# === Metrikleri Hesapla ===
accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average='macro')
precision = precision_score(all_labels, all_preds, average='macro')
recall = recall_score(all_labels, all_preds, average='macro')

print(f"Doğruluk: %{accuracy * 100:.2f}")
print(f"F1 Skoru: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
