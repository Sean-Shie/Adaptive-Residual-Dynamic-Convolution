import os
import tarfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
from typing import Union, Tuple
import json
from models import BaselineCNN, ARD_CNN

# =========================
# Reproducibility
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# =========================
# CIFAR-10 Local Check
# =========================
DATA_DIR = "./data"
CIFAR_FOLDER = os.path.join(DATA_DIR, "cifar-10-batches-py")
TAR_PATH = "./cifar-10-python.tar.gz"
CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck']
# =========================
# Config
# =========================
class Config:
    batch_size = 1024
    lr = 1e-3   # 3e-4
    epochs = 200
    num_classes = len(cifar10_classes)
    loss_name = "cross_entropy"

def prepare_cifar10():
    folder_exists = os.path.exists(CIFAR_FOLDER)
    tar_exists = os.path.exists(TAR_PATH)

    if not folder_exists:
        if not tar_exists:
            raise FileNotFoundError(
                "Neither CIFAR-10 folder nor cifar-10-python.tar.gz found. Stopping execution."
            )
        else:
            print("Found cifar-10-python.tar.gz. Extracting...")
            os.makedirs(DATA_DIR, exist_ok=True)

            with tarfile.open(TAR_PATH, "r:gz") as tar:
                tar.extractall(path=DATA_DIR)

            if not os.path.exists(CIFAR_FOLDER):
                raise RuntimeError("Extraction failed: cifar-10-batches-py not found.")
            print("Extraction completed.")

    else:
        print("CIFAR-10 dataset already exists.")

# =========================
# Loss Factory
# =========================
def get_loss(name):
    if name == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif name == "mse":
        return nn.MSELoss()
    else:
        raise ValueError("Unknown loss")

# =========================
# Cut-out
# =========================
class Cutout(object):
    def __init__(self, n_holes=1, length=16):
        """
        n_holes : 要遮住幾個洞
        length  : 每個洞的邊長
        """

        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        img: Tensor, shape (C, H, W)
        """

        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)

        img = img * mask

        return img


# =========================
# Mixup
# =========================
def mixup_data(x, y, alpha=0.2, device='cuda'):
    """
    Returns mixed inputs, pairs of targets, and lambda
    x: input images
    y: labels
    """

    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)

    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]

    y_a = y
    y_b = y[index]

    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + \
           (1 - lam) * criterion(pred, y_b)

# =========================
# Cut-mix
# =========================
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]

    cut_rat = np.sqrt(1. - lam)

    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix_data(x, y, alpha=1.0):
    import numpy as np
    import torch

    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)

    # ★ 關鍵修正：使用 x.device
    rand_index = torch.randperm(
        batch_size,
        device=x.device
    )

    target_a = y
    target_b = y[rand_index]

    bbx1, bby1, bbx2, bby2 = rand_bbox(
        x.size(),
        lam
    )

    x[:, :, bbx1:bbx2, bby1:bby2] = \
        x[rand_index, :, bbx1:bbx2, bby1:bby2]

    # 根據實際 patch 面積重新計算 lam
    lam = 1 - (
        (bbx2 - bbx1) * (bby2 - bby1)
        / (x.size(-1) * x.size(-2))
    )

    return x, target_a, target_b, lam

# --------------------------------------------------
# Unified criterion
# --------------------------------------------------
def mixed_criterion(
    criterion,
    pred,
    target_a,
    target_b,
    lam
):
    return (
        lam * criterion(pred, target_a)
        + (1 - lam) * criterion(pred, target_b)
    )

def preprocess():
    set_seed()

    prepare_cifar10()

    cfg = Config()

    # =========================
    # Dataset (NO download)
    # =========================

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        #Cutout(n_holes=1, length=16),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    '''
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    '''
    trainset = torchvision.datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=False,   # 禁止下載
        transform=transform_train
    )

    testset = torchvision.datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=False,
        transform=transform_test
    )

    trainloader = DataLoader(trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=4)
    testloader = DataLoader(testset, batch_size=cfg.batch_size, shuffle=False, num_workers=4)

    criterion = get_loss(cfg.loss_name)

    return cfg, trainloader, testloader, criterion

def save_checkpoint(model, optimizer, epoch, best_acc, name):
    path = os.path.join(CHECKPOINT_DIR, f"{name}_best.pth")

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_acc": best_acc
    }, path)

    print(f"Saved best model to {path}")

def load_checkpoint(model, optimizer, name):
    path = os.path.join(CHECKPOINT_DIR, f"{name}_best.pth")

    if not os.path.exists(path):
        print(f"No checkpoint found for {name}")
        return 0, 0.0

    checkpoint = torch.load(path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"Loaded checkpoint: {path}")

    return checkpoint["epoch"], checkpoint["best_acc"]


# -----------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, use_mixup_cutmix=False):
    model.train()
    total_loss = 0

    running_loss = 0.0
    correct = 0
    total = 0

    mixup_prob = 0.5
    mixup_alpha = 0.2
    cutmix_alpha = 1.0

    pbar = tqdm(loader, desc="Train", leave=False)

    for x, y in pbar:
        x, y = x.to(DEVICE), y.to(DEVICE)

        if use_mixup_cutmix:
            # ---------------------------------
            # MixUp or CutMix
            # ---------------------------------
            r = np.random.rand()

            if r < mixup_prob:
                # Use MixUp
                x, target_a, target_b, lam = mixup_data(
                    x,
                    y,
                    alpha=mixup_alpha,
                    device=DEVICE
                )

            else:
                # Use CutMix
                x, target_a, target_b, lam = cutmix_data(
                    x,
                    y,
                    alpha=cutmix_alpha
                )

        optimizer.zero_grad()
        out = model(x)

        # loss for CutMix
        if use_mixup_cutmix:
            loss = mixed_criterion(
                criterion,
                out,
                target_a,
                target_b,
                lam
            )
        else:
            loss = criterion(out, y)  # normal loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        running_loss += loss.item() * x.size(0)

        _, preds = torch.max(out, 1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    train_loss = running_loss / total
    train_acc = correct / total

    return total_loss / len(pbar), train_loss, train_acc

def evaluate_old(model, loader, criterion):
    model.eval()
    correct = 0
    total = 0

    val_loss = 0.0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)

            loss = criterion(out, y)

            val_loss += loss.item() * x.size(0)

            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    val_loss = val_loss / total
    val_acc = correct / total

    return val_loss, val_acc

def evaluate(model, loader):
    y_true, y_pred, y_prob = evaluate_model(model, loader, DEVICE)

    compute_metrics(y_true, y_pred)

    per_class_accuracy(y_true, y_pred)

    plot_confusion_matrix(y_true, y_pred, classes=cifar10_classes)

    print("Top-5 Accuracy:", top_k_accuracy(y_true, y_prob, k=5))

    plot_roc(y_true, y_prob)

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def evaluate_model(model, dataloader, device):
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)

    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro): {recall:.4f}")
    print(f"F1-score (macro): {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=cifar10_classes))

def per_class_accuracy(y_true, y_pred, num_classes=10):
    acc_dict = {}
    for i in range(num_classes):
        idx = (y_true == i)
        acc = (y_pred[idx] == i).mean()
        acc_dict[i] = acc
        print(f"Class {i} ({cifar10_classes[i]}) Accuracy: {acc:.4f}")
    return acc_dict

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes,
                yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def top_k_accuracy(y_true, y_prob, k=5):
    top_k = np.argsort(y_prob, axis=1)[:, -k:]
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] in top_k[i]:
            correct += 1
    return correct / len(y_true)

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

def plot_roc(y_true, y_prob, num_classes=10):
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))

    plt.figure(figsize=(8,6))

    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{cifar10_classes[i]} (AUC={roc_auc:.2f})")

    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

def plot_training_curves(history):
    epochs = range(1, len(history["train_loss"]) + 1)

    # -------- Loss --------
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    # -------- Accuracy --------
    plt.figure()
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.show()

def run_experiment(model_class, name, cfg, trainloader, testloader, criterion, resume=True, test_only=False):
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    # ... 原有的 model 與 optimizer 定義 ...
    if isinstance(model_class, nn.Module):
        model = model_class.to(DEVICE)
    else:
        model = model_class().to(DEVICE)

    # 建議使用 AdamW 配合 MLP/Transformer 架構
    # SGD for ResNet
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.05)
    #optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=5e-4)

    # --- 新增：定義 Scheduler ---
    # T_max 通常設定為總 Epoch 數
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    # ----------------------------

    start_epoch = 0
    best_acc = 0.0

    # 🔹 Resume 邏輯
    if resume:
        # 注意：如果載入舊進度，也要載入 scheduler 的狀態
        old_start_epoch, best_acc = load_checkpoint(model, optimizer, name)
        print(F'Best Acc for {name}: {best_acc:.4f}')
        # 如果您的 load_checkpoint 有儲存 scheduler，此處應更新

    if test_only:
        evaluate(model, testloader)
        return model

    for epoch in range(start_epoch, cfg.epochs):
        loss, train_loss, train_acc = train_one_epoch(model, trainloader, optimizer, criterion, use_mixup_cutmix=True)
        val_loss, val_acc = evaluate_old(model, testloader, criterion)

        # =========================
        # SAVE HISTORY
        # =========================
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # --- 新增：更新 Scheduler ---
        # 每個 Epoch 結束後執行一次 step
        scheduler.step()
        # ----------------------------

        # 獲取當前學習率以便觀察
        current_lr = optimizer.param_groups[0]['lr']
        print(f"{name} | Epoch {epoch + 1}/{cfg.epochs} | Loss: {loss:.4f} | Acc: {val_acc:.4f} | LR: {current_lr:.6f}")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model, optimizer, epoch, best_acc, name)

    plot_training_curves(history)

    # 建議：先建立資料夾（若不存在）
    save_dir = "./results"
    os.makedirs(save_dir, exist_ok=True)

    # history 儲存路徑
    save_path = os.path.join(save_dir, F"{name}_training_history.json")

    # 將 history 存成 json
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)

    print(f"History saved to: {save_path}")

    print(f"Best Acc for {name}: {best_acc:.4f}")
    return model

if __name__=='__main__':
    cfg, trainloader, testloader, criterion = preprocess()

    cnn_model = run_experiment(BaselineCNN, "BaselineCNN", cfg, trainloader, testloader, criterion, resume=True, test_only=True)
    ardcnn_model = run_experiment(ARD_CNN, "ARDCNN", cfg, trainloader, testloader, criterion, resume=True, test_only=True)
