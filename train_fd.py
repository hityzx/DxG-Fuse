import argparse
import json
import os
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "fd"
WEIGHTS_PATH = BASE_DIR / "best_fd_model.pth"

BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
PATIENCE = 5
NUM_WORKERS = 4


def get_device(device_arg=None):
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# deepfake & photoshop -> 1, normal -> 0
def label_mapper(folder_name):
    folder_name = folder_name.lower()
    if folder_name in ["deepfake", "photoshop"]:
        return 1
    if folder_name == "normal":
        return 0
    raise ValueError(f"Unknown class folder: {folder_name}. Expected normal, deepfake, or photoshop.")


def make_datasets(root_dir, transform):
    base_dataset = datasets.ImageFolder(root_dir, transform=transform)
    new_samples = []
    for path, class_index in base_dataset.samples:
        folder = base_dataset.classes[class_index]
        new_samples.append((path, label_mapper(folder)))
    base_dataset.samples = new_samples
    base_dataset.targets = [s[1] for s in new_samples]
    base_dataset.classes = ["normal", "tampered"]
    base_dataset.class_to_idx = {"normal": 0, "tampered": 1}
    return base_dataset


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.TrivialAugmentWide(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.5),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def build_model(num_classes=2, pretrained=False, device=None):
    try:
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        model = models.resnet34(weights=weights)
    except (AttributeError, TypeError):
        model = models.resnet34(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    if device is not None:
        model = model.to(device)
    return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(loader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(inputs), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validation", leave=False):
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return acc, prec, rec, f1


def train(args, device):
    train_dir = args.data_dir / "train"
    val_dir = args.data_dir / "val"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise FileNotFoundError(
            f"Training data must be organized as {args.data_dir}/train and {args.data_dir}/val, "
            "with each split containing the class folders (e.g. normal, deepfake, photoshop)."
        )

    train_dataset = make_datasets(str(train_dir), train_transform)
    val_dataset = make_datasets(str(val_dir), val_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

    model = build_model(pretrained=True, device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    args.weights.parent.mkdir(parents=True, exist_ok=True)
    best_acc = float("-inf")
    epochs_no_improve = 0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_acc, val_prec, val_rec, val_f1 = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | "
              f"Val Prec: {val_prec:.4f} | Val Rec: {val_rec:.4f} | Val F1: {val_f1:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), args.weights)
            print(f">> The new best model has been saved, with an accuracy of: {best_acc:.4f}")
        else:
            epochs_no_improve += 1
            print(f">> No improvement for {epochs_no_improve} epochs")
            if epochs_no_improve >= args.patience:
                print(f"Training stopped early at epoch {epoch}")
                break

    model.load_state_dict(load_state_dict(args.weights, device))
    acc, prec, rec, f1 = evaluate(model, val_loader, device)
    print("\nBest Model Validation Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    return model


def load_state_dict(weights_path, device):
    try:
        return torch.load(weights_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(weights_path, map_location=device)


def load_for_inference(weights_path, device):
    if not Path(weights_path).is_file():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")
    model = build_model(pretrained=False, device=device)
    model.load_state_dict(load_state_dict(weights_path, device))
    model.eval()
    return model


def resolve_image_path(item, input_dir):
    image_path = Path(str(item["image_path"])).expanduser()
    if not image_path.is_absolute():
        image_path = input_dir / image_path
    return image_path


def predict_images(model, data, input_dir, device, batch_size):
    model.eval()
    for start in tqdm(range(0, len(data), batch_size), desc="Detection"):
        batch = data[start:start + batch_size]
        tensors = []
        for item in batch:
            image_path = resolve_image_path(item, input_dir)
            try:
                with Image.open(image_path) as source:
                    image = source.convert("RGB")
                try:
                    tensors.append(val_transform(image))
                finally:
                    image.close()
            except Exception as error:
                raise RuntimeError(f"Failed to load {image_path}; input JSON will not be updated.") from error
        inputs = torch.stack(tensors).to(device)
        with torch.no_grad():
            preds = torch.argmax(model(inputs), dim=1).cpu().tolist()
        for item, pred in zip(batch, preds):
            item["forgery"] = int(pred)
    return len(data)


def save_json_in_place(path, data):
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         suffix=".tmp", delete=False) as handle:
            temporary_path = Path(handle.name)
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.chmod(path.stat().st_mode & 0o777)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def detect(args, device):
    input_path = args.input_json.expanduser().resolve()
    with input_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"{input_path} must contain a JSON array of records.")
    for index, item in enumerate(data):
        if not isinstance(item, dict) or not item.get("image_path"):
            raise ValueError(f"Record {index + 1} is missing image_path.")

    model = load_for_inference(args.weights, device)
    processed = predict_images(model, data, input_path.parent, device, args.batch_size)
    save_json_in_place(input_path, data)

    real_count = sum(1 for item in data if item.get("forgery") == 0)
    fake_count = sum(1 for item in data if item.get("forgery") == 1)
    print(f"Detected {processed}/{len(data)} images in {input_path} "
          f"(forgery=0: {real_count}, forgery=1: {fake_count})")


def main():
    parser = argparse.ArgumentParser(
        description="Train an image forgery detector on data/fd and annotate a JSON via forgery (0=real, 1=fake)."
    )
    parser.add_argument("--input_json", type=Path, default=None,
                        help="JSON array whose records contain image_path; results are written back in place.")
    parser.add_argument("--data_dir", type=Path, default=DATA_DIR,
                        help="Training data directory containing train/ and val/ splits.")
    parser.add_argument("--weights", type=Path, default=WEIGHTS_PATH,
                        help="Path to best model weights.")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--device", default=None, help="cuda / cuda:0 / cpu (default: auto).")
    parser.add_argument("--retrain", action="store_true",
                        help="Retrain even if weights already exist when --input_json is given.")
    args = parser.parse_args()
    for name in ("epochs", "batch_size", "patience"):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be greater than 0")
    if args.num_workers < 0:
        parser.error("num_workers must be non-negative")
    if args.learning_rate <= 0:
        parser.error("learning_rate must be greater than 0")

    device = get_device(args.device)
    weights_exist = args.weights.is_file()

    if args.input_json is not None and weights_exist and not args.retrain:
        detect(args, device)
        return

    if args.input_json is not None and not weights_exist:
        print(f"No weights found at {args.weights}; training first.")

    train(args, device)

    if args.input_json is not None:
        detect(args, device)


if __name__ == "__main__":
    main()
