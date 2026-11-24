"""Model 4 - fine-tuned CNN for deepfake images. Needs a GPU (e.g. Google Colab).

EfficientNet-B0, pretrained on ImageNet, fine-tuned on real vs. AI-generated faces.
Saved in Hugging Face format, which the gateway loads with transformers.pipeline().

    python training/image_finetune.py --data data/real_vs_fake/real-vs-fake
"""

import argparse
import random
import time
from pathlib import Path

import torch
from PIL import ImageOps
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from transformers import AutoImageProcessor, AutoModelForImageClassification

from common import MODELS_DIR, SEED, evaluate, find_split, plot_history


class ProcessorTransform:
    """Runs the base model's own image processor, so training preprocessing matches
    what transformers.pipeline() does in the gateway exactly."""

    def __init__(self, processor, augment):
        self.processor = processor
        self.augment = augment

    def __call__(self, img):
        img = img.convert("RGB")
        if self.augment and random.random() < 0.5:
            img = ImageOps.mirror(img)
        return self.processor(images=img, return_tensors="pt")["pixel_values"][0]


def subset(ds, limit):
    if not limit or limit >= len(ds):
        return ds
    return Subset(ds, random.Random(SEED).sample(range(len(ds)), limit))


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            logits = model(pixel_values=x.to(device)).logits
        probs.append(logits.float().softmax(1).cpu())
        labels.append(y)
    return torch.cat(probs), torch.cat(labels)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="folder containing train/ valid/ test/, each with fake/ and real/")
    ap.add_argument("--base", default="google/efficientnet-b0", help="pretrained model to fine-tune")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--train-limit", type=int, default=40000, help="training images to use (0 = all)")
    ap.add_argument("--eval-limit", type=int, default=5000, help="validation/test images to use (0 = all)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", type=Path, default=MODELS_DIR / "image-efficientnet")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("WARNING: no GPU found - this will be very slow. Use Google Colab (Runtime > Change runtime type > GPU).")

    processor = AutoImageProcessor.from_pretrained(args.base)
    train_full = datasets.ImageFolder(find_split(args.data, "train"), ProcessorTransform(processor, True))
    classes = train_full.classes
    if "fake" not in classes:
        raise SystemExit(f"Expected a 'fake' class folder, found {classes}")
    fake_idx = classes.index("fake")
    eval_tf = ProcessorTransform(processor, False)
    train = subset(train_full, args.train_limit)
    val = subset(datasets.ImageFolder(find_split(args.data, "valid"), eval_tf), args.eval_limit)
    test = subset(datasets.ImageFolder(find_split(args.data, "test"), eval_tf), args.eval_limit)
    print(f"classes {classes}; train {len(train)}, val {len(val)}, test {len(test)} images")

    loader = lambda ds, shuffle: DataLoader(
        ds, args.batch, shuffle=shuffle, num_workers=args.workers, persistent_workers=args.workers > 0
    )
    train_loader, val_loader, test_loader = loader(train, True), loader(val, False), loader(test, False)

    id2label = dict(enumerate(classes))
    model = AutoModelForImageClassification.from_pretrained(
        args.base, num_labels=len(classes), id2label=id2label,
        label2id={v: k for k, v in id2label.items()}, ignore_mismatched_sizes=True,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(train_loader))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    loss_fn = torch.nn.CrossEntropyLoss()

    history, best_acc = [], -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t, total, seen = time.time(), 0.0, 0
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = loss_fn(model(pixel_values=x).logits, y)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item() * len(y)
            seen += len(y)
            if i % 100 == 0:
                print(f"  epoch {epoch} step {i}/{len(train_loader)} loss {loss.item():.4f}", flush=True)

        probs, labels = predict(model, val_loader, device)
        val_acc = (probs.argmax(1) == labels).float().mean().item()
        history.append({"epoch": epoch, "train_loss": round(total / seen, 4), "val_accuracy": round(val_acc, 4)})
        print(f"epoch {epoch}: train loss {total / seen:.4f}, val accuracy {val_acc:.4f} ({time.time() - t:.0f}s)")
        if val_acc > best_acc:
            best_acc = val_acc
            model.save_pretrained(args.out)
            processor.save_pretrained(args.out)

    model = AutoModelForImageClassification.from_pretrained(args.out).to(device)
    probs, labels = predict(model, test_loader, device)
    plot_history(history, f"{args.base} fine-tuning", args.out / "training_curve.png")
    evaluate(
        f"{args.base} (fine-tuned)",
        (labels == fake_idx).int().numpy(),
        probs[:, fake_idx].numpy(),
        args.out,
        {"base_model": args.base, "epochs": args.epochs, "best_val_accuracy": round(best_acc, 4),
         "train_samples": len(train)},
    )


if __name__ == "__main__":
    main()
