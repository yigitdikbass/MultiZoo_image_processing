from data_loader import prepare_dataloaders
from train_vit import train_vit
from train_swin import train_swin
import torch

# Veriyi hazırla
train_loader, val_loader, class_names = prepare_dataloaders("train", "val", batch_size=32)

# Cihazı belirle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan cihaz: {device}")

# Modeli eğit ve kaydet
model = train_swin(train_loader, val_loader, num_classes=len(class_names), device=device, epochs=20, patience=3)
