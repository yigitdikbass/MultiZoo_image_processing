# train_swin.py
import torch
import torch.nn as nn
import timm
from torch.optim import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import os
import matplotlib
matplotlib.use('Agg')

def train_swin(train_loader, val_loader, num_classes, device, epochs=20, patience=5):
    model = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-4)

    best_val_acc = 0.0
    counter = 0

    train_accs, val_accs = [], []
    train_f1s, val_f1s = [], []
    train_precisions, val_precisions = [], []
    train_recalls, val_recalls = [], []

    for epoch in range(epochs):
        model.train()
        y_true_train, y_pred_train = [], []

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            y_true_train.extend(labels.cpu().numpy())
            y_pred_train.extend(predicted.cpu().numpy())

        train_acc = accuracy_score(y_true_train, y_pred_train)
        train_f1 = f1_score(y_true_train, y_pred_train, average='macro', zero_division=1)
        train_precision = precision_score(y_true_train, y_pred_train, average='macro', zero_division=1)
        train_recall = recall_score(y_true_train, y_pred_train, average='macro', zero_division=1)

        train_accs.append(train_acc)
        train_f1s.append(train_f1)
        train_precisions.append(train_precision)
        train_recalls.append(train_recall)

        # Validation
        model.eval()
        y_true_val, y_pred_val = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                y_true_val.extend(labels.cpu().numpy())
                y_pred_val.extend(predicted.cpu().numpy())

        val_acc = accuracy_score(y_true_val, y_pred_val)
        val_f1 = f1_score(y_true_val, y_pred_val, average='macro', zero_division=1)
        val_precision = precision_score(y_true_val, y_pred_val, average='macro', zero_division=1)
        val_recall = recall_score(y_true_val, y_pred_val, average='macro', zero_division=1)

        val_accs.append(val_acc)
        val_f1s.append(val_f1)
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)

        print(f"Epoch {epoch+1}/{epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            counter = 0
            torch.save(model.state_dict(), "swin_model2.pth")
        else:
            counter += 1
            print(f"EarlyStopping counter: {counter}/{patience}")
            if counter >= patience:
                print("Erken durdurma tetiklendi. Eğitim durduruluyor.")
                break

    os.makedirs("metrics_swin", exist_ok=True)

    def plot_metric(train, val, label, filename, color1, color2):
        plt.figure()
        plt.plot(range(1, len(train)+1), train, label=f"Train {label}", color=color1)
        plt.plot(range(1, len(val)+1), val, label=f"Validation {label}", color=color2)
        plt.xlabel("Epochs")
        plt.ylabel(label)
        plt.title(f"{label} Learning Curve")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join("metrics_swin", filename))
        plt.close()

    plot_metric(train_accs, val_accs, "Accuracy", "accuracy_curve.png", "orange", "green")
    plot_metric(train_f1s, val_f1s, "F1 Score", "f1_curve.png", "purple", "red")
    plot_metric(train_precisions, val_precisions, "Precision", "precision_curve.png", "blue", "gray")
    plot_metric(train_recalls, val_recalls, "Recall", "recall_curve.png", "brown", "cyan")

    return model
