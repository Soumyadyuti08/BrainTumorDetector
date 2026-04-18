# Brain Tumor MRI Classification using Custom CNN
# Dataset: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

import os, copy, random, math, warnings
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedShuffleSplit

warnings.filterwarnings("ignore")

# hyperparameters
DATA_DIR   = r"/home/nanoz/Downloads/brain/brain_new/data"
IMG_SIZE   = 256
BATCH      = 24
EPOCHS     = 70
LR         = 2e-4       
WD         = 2e-4
PATIENCE   = 18
NC         = 4
CLASSES    = ["glioma", "meningioma", "notumor", "pituitary"]
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE       = "outputs"
SEED       = 42

os.makedirs(SAVE, exist_ok=True)
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

# crop black borders around the brain (recommended by dataset author)
def remove_margins(img, threshold=10, padding=5):
    gray = np.array(img.convert("L"))
    rows = np.where(gray.max(axis=1) > threshold)[0]
    cols = np.where(gray.max(axis=0) > threshold)[0]
    if len(rows) == 0 or len(cols) == 0:
        return img
    top    = max(rows[0] - padding, 0)
    bottom = min(rows[-1] + padding, gray.shape[0])
    left   = max(cols[0] - padding, 0)
    right  = min(cols[-1] + padding, gray.shape[1])
    return img.crop((left, top, right, bottom))

# custom dataset that applies margin removal before transforms
class CroppedImageFolder(Dataset):
    def __init__(self, root, transform=None):
        self.ds = datasets.ImageFolder(root)
        self.transform = transform
        self.samples = self.ds.samples
        self.targets = self.ds.targets
        self.classes = self.ds.classes
        self.class_to_idx = self.ds.class_to_idx
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        path, label = self.ds.samples[idx]
        img = Image.open(path).convert("RGB")
        img = remove_margins(img)
        if self.transform:
            img = self.transform(img)
        return img, label

# augmentation
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE+20, IMG_SIZE+20)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.1),
    transforms.RandomRotation(15),
    transforms.RandomAffine(0, translate=(0.05,0.05), scale=(0.92,1.08)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    transforms.RandomErasing(p=0.1, scale=(0.02,0.06)),
])

eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

train_dir = os.path.join(DATA_DIR, "Training")
test_dir  = os.path.join(DATA_DIR, "Testing")

full_aug  = CroppedImageFolder(train_dir, transform=train_tf)
full_eval = CroppedImageFolder(train_dir, transform=eval_tf)
test_ds   = CroppedImageFolder(test_dir,  transform=eval_tf)

# 85-15 stratified split
labels = [s[1] for s in full_aug.samples]
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
tr_idx, va_idx = next(sss.split(np.zeros(len(labels)), labels))

train_loader = DataLoader(Subset(full_aug, tr_idx), batch_size=BATCH, shuffle=True, num_workers=0, pin_memory=True)
val_loader   = DataLoader(Subset(full_eval, va_idx), batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=True)

# CBAM attention - channel + spatial
class CBAM(nn.Module):
    def __init__(self, ch):
        super().__init__()
        mid = max(ch//16, 8)
        self.mlp = nn.Sequential(nn.Linear(ch,mid,bias=False), nn.ReLU(), nn.Linear(mid,ch,bias=False))
        self.spatial = nn.Conv2d(2, 1, 7, padding=3, bias=False)
    def forward(self, x):
        b, c = x.size(0), x.size(1)
        # channel attention
        ca = torch.sigmoid(self.mlp(F.adaptive_avg_pool2d(x,1).view(b,c)) +
                           self.mlp(F.adaptive_max_pool2d(x,1).view(b,c))).view(b,c,1,1)
        x = x * ca
        # spatial attention
        sa = torch.sigmoid(self.spatial(torch.cat([x.mean(1,keepdim=True), x.max(1,keepdim=True)[0]], 1)))
        return x * sa

class ConvBN(nn.Sequential):
    def __init__(self, ic, oc, k=3, s=1, p=1):
        super().__init__(nn.Conv2d(ic,oc,k,s,p,bias=False), nn.BatchNorm2d(oc), nn.GELU())

# residual block with CBAM
class ResBlock(nn.Module):
    def __init__(self, ic, oc, stride=1, drop=0.1):
        super().__init__()
        self.conv = nn.Sequential(ConvBN(ic,oc), nn.Conv2d(oc,oc,3,1,1,bias=False), nn.BatchNorm2d(oc))
        self.cbam = CBAM(oc)
        self.drop = nn.Dropout2d(drop)
        self.pool = nn.MaxPool2d(2) if stride==2 else nn.Identity()
        self.skip = nn.Sequential(nn.Conv2d(ic,oc,1,bias=False), nn.BatchNorm2d(oc)) if ic!=oc else nn.Identity()
    def forward(self, x):
        return self.pool(F.gelu(self.drop(self.cbam(self.conv(x))) + self.skip(x)))

# main model - uses multi-scale fusion from stage3 and stage4
class BrainTumorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(ConvBN(3,48,7,2,3), nn.MaxPool2d(3,2,1))
        self.s1 = nn.Sequential(ResBlock(48,64,2,0.05), ResBlock(64,64,1,0.05))
        self.s2 = nn.Sequential(ResBlock(64,128,2,0.08), ResBlock(128,128,1,0.08))
        self.s3 = nn.Sequential(ResBlock(128,256,2,0.10), ResBlock(256,256,1,0.10), ResBlock(256,256,1,0.10))
        self.s4 = nn.Sequential(ResBlock(256,512,2,0.15), ResBlock(512,512,1,0.15))
        # pool from both stage3 and stage4 for multi-scale features
        self.gap3=nn.AdaptiveAvgPool2d(1); self.gmp3=nn.AdaptiveMaxPool2d(1)
        self.gap4=nn.AdaptiveAvgPool2d(1); self.gmp4=nn.AdaptiveMaxPool2d(1)
        self.fbn = nn.BatchNorm1d(1536)  # 256*2 + 512*2
        # classifier
        self.head = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(1536,512), nn.BatchNorm1d(512), nn.GELU(),
            nn.Dropout(0.3), nn.Linear(512,256), nn.BatchNorm1d(256), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(256,NC))
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, (nn.BatchNorm2d,nn.BatchNorm1d)):
                nn.init.constant_(m.weight,1); nn.init.constant_(m.bias,0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x); x = self.s1(x); x = self.s2(x)
        s3 = self.s3(x); s4 = self.s4(s3)
        # concat avg+max pool from both stages
        f3 = torch.cat([self.gap3(s3).flatten(1), self.gmp3(s3).flatten(1)], 1)
        f4 = torch.cat([self.gap4(s4).flatten(1), self.gmp4(s4).flatten(1)], 1)
        return self.head(self.fbn(torch.cat([f3, f4], 1)))

# focal loss - gamma=2 makes the model focus more on hard samples
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        return ((1 - torch.exp(-ce)) ** self.gamma * ce).mean()

# mixup augmentation
def mixup(imgs, labels, alpha=0.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(imgs.size(0), device=imgs.device)
    return lam*imgs+(1-lam)*imgs[idx], labels, labels[idx], lam

# early stopping
class EarlyStopping:
    def __init__(self, patience=18):
        self.patience = patience
        self.counter = 0
        self.best = None
        self.state = None
    def step(self, val_acc, model):
        if self.best is None or val_acc > self.best + 5e-5:
            self.best = val_acc
            self.state = copy.deepcopy(model.state_dict())
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

# train one epoch
def train_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss, correct, count = 0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        mixed_imgs, targets_a, targets_b, lam = mixup(imgs, labels)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=DEVICE.type=="cuda"):
            outputs = model(mixed_imgs)
            loss = lam * criterion(outputs, targets_a) + (1-lam) * criterion(outputs, targets_b)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(1)
        correct += (lam*(preds==targets_a).float() + (1-lam)*(preds==targets_b).float()).sum().item()
        count += imgs.size(0)
    return total_loss/count, correct/count

# evaluate on val/test set
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_loss, correct, count = 0, 0, 0
    all_preds, all_labels = [], []
    ce = nn.CrossEntropyLoss()
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        total_loss += ce(outputs, labels).item() * imgs.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        count += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss/count, correct/count, np.array(all_preds), np.array(all_labels)

# grad-cam
class GradCAM:
    def __init__(self, model, layer):
        self.model = model
        self.activations = None
        self.gradients = None
        layer.register_forward_hook(lambda m,i,o: setattr(self, 'activations', o.detach()))
        layer.register_full_backward_hook(lambda m,gi,go: setattr(self, 'gradients', go[0].detach()))

    @torch.enable_grad()
    def run(self, x, cls=None):
        self.model.eval()
        x = x.unsqueeze(0).to(DEVICE).requires_grad_(True)
        output = self.model(x)
        cls = cls or output.argmax(1).item()
        self.model.zero_grad()
        output[0, cls].backward()
        weights = self.gradients.mean(dim=[2,3], keepdim=True)
        cam = F.relu((weights * self.activations).sum(1, keepdim=True))
        cam = F.interpolate(cam, (IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8), cls

# plots
def plot_curves(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    epochs = range(1, len(history["train_loss"])+1)
    ax1.plot(epochs, history["train_acc"], "b-", lw=1.5, label="Train")
    ax1.plot(epochs, history["val_acc"], "r-", lw=1.5, label="Val")
    ax1.set_title("Accuracy"); ax1.legend(); ax1.grid(alpha=.3); ax1.set_xlabel("Epoch")
    ax2.plot(epochs, history["train_loss"], "b-", lw=1.5, label="Train")
    ax2.plot(epochs, history["val_loss"], "r-", lw=1.5, label="Val")
    ax2.set_title("Loss"); ax2.legend(); ax2.grid(alpha=.3); ax2.set_xlabel("Epoch")
    plt.tight_layout(); plt.savefig(f"{SAVE}/training_curves.png", dpi=150); plt.close()

def plot_cm(cm):
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, linewidths=.5, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion Matrix")
    plt.tight_layout(); plt.savefig(f"{SAVE}/confusion_matrix.png", dpi=150); plt.close()

def plot_metrics(labels, preds):
    report = classification_report(labels, preds, target_names=CLASSES, output_dict=True)
    x = np.arange(NC); w = 0.2
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (key, col) in enumerate(zip(["precision","recall","f1-score"],
                                        ["steelblue","darkorange","seagreen"])):
        vals = [report[c][key] for c in CLASSES]
        bars = ax.bar(x+(i-1)*w, vals, w, label=key.capitalize(), color=col, alpha=.85)
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, v+.01, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(CLASSES); ax.set_ylim(0, 1.1)
    ax.legend(); ax.grid(axis="y", alpha=.3); ax.set_title("Per-Class Metrics")
    plt.tight_layout(); plt.savefig(f"{SAVE}/per_class_metrics.png", dpi=150); plt.close()

def plot_gradcam(model, n=2):
    gcam = GradCAM(model, model.s4[1].conv[1])
    samples = []
    for ci, cn in enumerate(CLASSES):
        d = Path(test_dir) / cn
        imgs = list(d.glob("*.jpg")) + list(d.glob("*.png")) + list(d.glob("*.jpeg"))
        for p in random.sample(imgs, min(n, len(imgs))):
            samples.append((p, ci, cn))
    mean = torch.tensor([.485,.456,.406]).view(3,1,1)
    std  = torch.tensor([.229,.224,.225]).view(3,1,1)
    rows = len(samples)
    fig, axes = plt.subplots(rows, 3, figsize=(10, rows*3))
    for i, (img_path, true_cls, cls_name) in enumerate(samples):
        pil = remove_margins(Image.open(img_path).convert("RGB"))
        tensor = eval_tf(pil)
        cam, pred_cls = gcam.run(tensor)
        orig = (tensor.cpu()*std+mean).clamp(0,1).permute(1,2,0).numpy()
        heat = plt.cm.jet(cam)[...,:3]
        overlay = np.clip(.55*orig+.45*heat, 0, 1)
        axes[i,0].imshow(orig); axes[i,0].set_title(f"True: {cls_name}", fontsize=9); axes[i,0].axis("off")
        axes[i,1].imshow(cam, cmap="jet"); axes[i,1].set_title("Grad-CAM", fontsize=9); axes[i,1].axis("off")
        color = "green" if pred_cls == true_cls else "red"
        axes[i,2].imshow(overlay); axes[i,2].set_title(f"Pred: {CLASSES[pred_cls]}", fontsize=9, color=color); axes[i,2].axis("off")
    plt.tight_layout(); plt.savefig(f"{SAVE}/gradcam.png", dpi=150, bbox_inches="tight"); plt.close()


def main():
    model = BrainTumorNet().to(DEVICE)
    criterion = FocalLoss(gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    # warmup for 5 epochs then cosine decay
    warmup = 5
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda ep:
        (ep+1)/warmup if ep < warmup else 0.5*(1+math.cos(math.pi*(ep-warmup)/(EPOCHS-warmup))))

    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type=="cuda")
    early_stop = EarlyStopping(PATIENCE)
    history = {"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[]}

    for epoch in range(1, EPOCHS+1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, scaler)
        val_loss, val_acc, _, _ = evaluate(model, val_loader)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        star = " *" if early_stop.best is None or val_acc >= (early_stop.best or 0) else ""
        print(f"Epoch {epoch:02d}/{EPOCHS}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}{star}")

        if early_stop.step(val_acc, model):
            print(f"Early stopping at epoch {epoch}")
            break

    # load best weights and save
    model.load_state_dict(early_stop.state)
    torch.save(model.state_dict(), f"{SAVE}/best_model.pth")

    # test
    _, test_acc, preds, labels = evaluate(model, test_loader)
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(f"F1 Macro: {f1_score(labels, preds, average='macro'):.4f}")
    print(f"F1 Weighted: {f1_score(labels, preds, average='weighted'):.4f}")
    print(f"\n{classification_report(labels, preds, target_names=CLASSES, digits=4)}")

    plot_curves(history)
    plot_cm(confusion_matrix(labels, preds))
    plot_metrics(labels, preds)
    plot_gradcam(model)

if __name__ == "__main__":
    main()