"""Model 2 - our own deepfake image detector: TruthNet, a CNN trained from scratch on CPU.

The architecture is in ai-gateway/app/cnn.py, so the gateway loads the checkpoint directly.

    python training/image_cnn.py --data data/real_vs_fake/real-vs-fake
"""

import argparse
import os
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from common import MODELS_DIR, SEED, evaluate, find_split, plot_history
from app.cnn import IMAGE_SIZE, TruthNet, to_tensor


def subset(ds, limit):
    if not limit or limit >= len(ds):
        return ds
    return Subset(ds, random.Random(SEED).sample(range(len(ds)), limit))


@torch.no_grad()
def predict(model, loader):
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        probs.append(model(x).softmax(1))
        labels.append(y)
    return torch.cat(probs), torch.cat(labels)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="folder containing train/ valid/ test/, each with fake/ and real/")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--train-limit", type=int, default=20000, help="training images to use (0 = all)")
    ap.add_argument("--eval-limit", type=int, default=4000, help="validation/test images to use (0 = all)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", type=Path, default=MODELS_DIR / "image-cnn")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    torch.set_num_threads(os.cpu_count() or 4)

    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.1),
        transforms.Lambda(to_tensor),
    ])
    eval_tf = transforms.Lambda(to_tensor)
    train_full = datasets.ImageFolder(find_split(args.data, "train"), train_tf)
    classes = train_full.classes
    if "fake" not in classes:
        raise SystemExit(f"Expected a 'fake' class folder, found {classes}")
    fake_idx = classes.index("fake")

    train = subset(train_full, args.train_limit)
    val = subset(datasets.ImageFolder(find_split(args.data, "valid"), eval_tf), args.eval_limit)
    test = subset(datasets.ImageFolder(find_split(args.data, "test"), eval_tf), args.eval_limit)
    print(f"classes {classes}; train {len(train)}, val {len(val)}, test {len(test)} images")

    loader = lambda ds, shuffle: DataLoader(
        ds, args.batch, shuffle=shuffle, num_workers=args.workers, persistent_workers=args.workers > 0
    )
    train_loader, val_loader, test_loader = loader(train, True), loader(val, False), loader(test, False)

    model = TruthNet(len(classes))
    print(f"TruthNet parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(train_loader))
    loss_fn = nn.CrossEntropyLoss()

    args.out.mkdir(parents=True, exist_ok=True)
    history, best_acc = [], -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t, total, seen = time.time(), 0.0, 0
        for i, (x, y) in enumerate(train_loader):
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(y)
            seen += len(y)
            if i % 50 == 0:
                print(f"  epoch {epoch} batch {i}/{len(train_loader)} loss {loss.item():.3f}", flush=True)

        probs, labels = predict(model, val_loader)
        val_acc = (probs.argmax(1) == labels).float().mean().item()
        history.append({"epoch": epoch, "train_loss": round(total / seen, 4), "val_accuracy": round(val_acc, 4)})
        print(f"epoch {epoch}: train loss {total / seen:.4f}, val accuracy {val_acc:.4f} ({time.time() - t:.0f}s)")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"state_dict": model.state_dict(), "classes": classes, "image_size": IMAGE_SIZE},
                       args.out / "model.pt")

    model.load_state_dict(torch.load(args.out / "model.pt", weights_only=True)["state_dict"])
    probs, labels = predict(model, test_loader)
    plot_history(history, "TruthNet CNN training", args.out / "training_curve.png")
    evaluate(
        "TruthNet CNN (own, from scratch)",
        (labels == fake_idx).int().numpy(),
        probs[:, fake_idx].numpy(),
        args.out,
        {"epochs": args.epochs, "best_val_accuracy": round(best_acc, 4), "train_samples": len(train),
         "image_size": IMAGE_SIZE},
    )


if __name__ == "__main__":
    main()
