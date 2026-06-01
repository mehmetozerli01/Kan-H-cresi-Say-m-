"""
BCCD veri seti üzerinde gerçek hücre sınıflandırma eğitimi (WBC / RBC / Platelets).
Kaynak kod (app.py, src/) değiştirilmez. Çıktılar: presentation/artifacts/

Çalıştırma:
  pip install -r presentation/requirements.txt
  python presentation/train_bccd_classifier.py
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
BCCD_ROOT = PROJECT_ROOT / "data" / "BCCD_Dataset-master" / "BCCD"
IMG_DIR = BCCD_ROOT / "JPEGImages"
ANN_DIR = BCCD_ROOT / "Annotations"
SPLIT_DIR = BCCD_ROOT / "ImageSets" / "Main"
ARTIFACTS = ROOT / "artifacts"
CROP_CACHE = ARTIFACTS / "crops"

CLASS_NAMES = ["WBC", "RBC", "Platelets"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
IMG_SIZE = 64
BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-3
SEED = 42
MIN_BOX = 10


def imread_unicode(path: Path) -> np.ndarray | None:
    """Windows Unicode yollarında cv2.imread yerine."""
    raw = np.fromfile(str(path), dtype=np.uint8)
    if raw.size == 0:
        return None
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix if path.suffix else ".jpg"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError(f"Görüntü kodlanamadı: {path}")
    encoded.tofile(str(path))


@dataclass
class CropSample:
    path: Path
    label: int


class CellCropDataset(Dataset):
    def __init__(self, samples: list[CropSample], augment: bool = False) -> None:
        self.samples = samples
        if augment:
            self.transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.ColorJitter(brightness=0.15, contrast=0.15),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225),
                    ),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225),
                    ),
                ]
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        img = imread_unicode(sample.path)
        if img is None:
            raise FileNotFoundError(sample.path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        tensor = self.transform(img)
        return tensor, sample.label


class SmallCellCNN(nn.Module):
    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def load_split_stems(split_name: str) -> set[str]:
    path = SPLIT_DIR / f"{split_name}.txt"
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    return {line.strip() for line in lines if line.strip()}


def parse_annotation(xml_path: Path) -> list[tuple[str, int, int, int, int]]:
    tree = ET.parse(xml_path)
    boxes: list[tuple[str, int, int, int, int]] = []
    for obj in tree.getroot().findall("object"):
        name = obj.findtext("name", "").strip()
        if name not in CLASS_TO_IDX:
            continue
        xmin = int(obj.findtext("bndbox/xmin", "0"))
        ymin = int(obj.findtext("bndbox/ymin", "0"))
        xmax = int(obj.findtext("bndbox/xmax", "0"))
        ymax = int(obj.findtext("bndbox/ymax", "0"))
        if xmax - xmin < MIN_BOX or ymax - ymin < MIN_BOX:
            continue
        boxes.append((name, xmin, ymin, xmax, ymax))
    return boxes


def build_crop_cache() -> dict[str, list[CropSample]]:
    """BCCD etiketlerinden kırpılmış hücre görselleri üretir."""
    splits = {
        "train": load_split_stems("train"),
        "val": load_split_stems("val"),
        "test": load_split_stems("test"),
    }
    stem_to_split: dict[str, str] = {}
    for split_name, stems in splits.items():
        for stem in stems:
            stem_to_split[stem] = split_name

    if CROP_CACHE.exists():
        for p in CROP_CACHE.glob("**/*"):
            if p.is_file():
                p.unlink()
    CROP_CACHE.mkdir(parents=True, exist_ok=True)

    datasets: dict[str, list[CropSample]] = {"train": [], "val": [], "test": []}
    counts = {c: 0 for c in CLASS_NAMES}

    for xml_path in sorted(ANN_DIR.glob("*.xml")):
        stem = xml_path.stem
        split = stem_to_split.get(stem)
        if split is None:
            continue

        img_path = IMG_DIR / f"{stem}.jpg"
        if not img_path.exists():
            continue

        img = imread_unicode(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        for box_idx, (name, xmin, ymin, xmax, ymax) in enumerate(
            parse_annotation(xml_path)
        ):
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(w, xmax)
            ymax = min(h, ymax)
            crop = img[ymin:ymax, xmin:xmax]
            if crop.size == 0:
                continue

            out_dir = CROP_CACHE / split / name
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{stem}_{box_idx:04d}.jpg"
            imwrite_unicode(out_path, crop)
            datasets[split].append(
                CropSample(path=out_path, label=CLASS_TO_IDX[name])
            )
            counts[name] += 1

    meta = {
        "class_names": CLASS_NAMES,
        "crop_counts_by_class": counts,
        "split_sizes": {k: len(v) for k, v in datasets.items()},
        "img_size": IMG_SIZE,
        "bccd_root": str(BCCD_ROOT),
    }
    (ARTIFACTS / "dataset_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return datasets


def compute_class_weights(samples: list[CropSample]) -> torch.Tensor:
    counts = np.zeros(len(CLASS_NAMES), dtype=np.float64)
    for s in samples:
        counts[s.label] += 1
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (len(CLASS_NAMES) * counts)
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(is_train):
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    for inputs, labels in loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        preds = outputs.argmax(dim=1).cpu().tolist()
        y_pred.extend(preds)
        y_true.extend(labels.tolist())
    return y_true, y_pred


def train() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if not BCCD_ROOT.exists():
        raise FileNotFoundError(f"BCCD veri seti bulunamadı: {BCCD_ROOT}")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    print("BCCD kırpılmış hücre veri seti hazırlanıyor...")
    datasets = build_crop_cache()
    for split, samples in datasets.items():
        print(f"  {split}: {len(samples)} örnek")

    if len(datasets["train"]) < 50:
        raise RuntimeError("Eğitim örneği yetersiz — BCCD yolu kontrol edin.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {device}")

    train_loader = DataLoader(
        CellCropDataset(datasets["train"], augment=True),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        CellCropDataset(datasets["val"], augment=False),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        CellCropDataset(datasets["test"], augment=False),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = SmallCellCNN(num_classes=len(CLASS_NAMES)).to(device)
    class_weights = compute_class_weights(datasets["train"]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    history: list[dict] = []
    best_val_acc = 0.0
    best_state = None

    print(f"Eğitim başlıyor ({EPOCHS} epoch)...")
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, None, device
        )
        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_loss": round(val_loss, 5),
            "train_acc": round(train_acc * 100, 3),
            "val_acc": round(val_acc * 100, 3),
        }
        history.append(record)
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train loss={train_loss:.4f} acc={train_acc*100:.2f}% | "
            f"val loss={val_loss:.4f} acc={val_acc*100:.2f}%"
        )
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), ARTIFACTS / "cell_classifier.pt")

    y_true, y_pred = evaluate_predictions(model, test_loader, device)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    test_acc = report["accuracy"]

    results = {
        "class_names": CLASS_NAMES,
        "epochs": EPOCHS,
        "img_size": IMG_SIZE,
        "device": str(device),
        "best_val_acc_percent": round(best_val_acc * 100, 3),
        "test_accuracy_percent": round(test_acc * 100, 3),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "history": history,
        "note": (
            "Gerçek BCCD VOC etiketlerinden kırpılmış hücreler ile eğitilmiş "
            "SmallCellCNN. ImageSets train/val/test bölünmesi kullanıldı."
        ),
    }
    (ARTIFACTS / "training_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nTest doğruluğu: {test_acc * 100:.2f}%")
    print(f"Sonuçlar: {ARTIFACTS / 'training_results.json'}")
    print(f"Model: {ARTIFACTS / 'cell_classifier.pt'}")


if __name__ == "__main__":
    train()
