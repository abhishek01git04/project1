"""Model 3 - fine-tuned transformer for fake news. Needs a GPU (e.g. Google Colab).

DistilBERT, pretrained on general English, fine-tuned on WELFake. Saved in Hugging
Face format, which the gateway loads with transformers.pipeline().

    python training/text_finetune.py --data data/WELFake_Dataset.csv
"""

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from common import MODELS_DIR, SEED, evaluate, load_welfake, plot_history

ID2LABEL = {0: "real", 1: "fake"}


def make_loader(texts, labels, tokenizer, batch, max_len, shuffle):
    def collate(items):
        enc = tokenizer([t for t, _ in items], truncation=True, max_length=max_len,
                        padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor([int(y) for _, y in items])
        return enc

    return DataLoader(list(zip(texts, labels)), batch_size=batch, shuffle=shuffle, collate_fn=collate)


@torch.no_grad()
def predict(model, loader, device):
    """Returns (P(fake), labels) as numpy arrays."""
    model.eval()
    probs, labels = [], []
    for batch in loader:
        labels.append(batch.pop("labels"))
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            logits = model(**batch).logits
        probs.append(logits.float().softmax(1)[:, 1].cpu())
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="WELFake_Dataset.csv")
    ap.add_argument("--fake-label", default="auto", help="label value meaning fake (default: auto-detect)")
    ap.add_argument("--base", default="distilbert-base-uncased", help="pretrained model to fine-tune")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--train-limit", type=int, help="use only N training articles")
    ap.add_argument("--out", type=Path, default=MODELS_DIR / "text-distilbert")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("WARNING: no GPU found - this will be very slow. Use Google Colab (Runtime > Change runtime type > GPU).")

    data = load_welfake(args.data, args.fake_label, args.train_limit)
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base, num_labels=2, id2label=ID2LABEL, label2id={v: k for k, v in ID2LABEL.items()}
    ).to(device)

    train_loader = make_loader(*data["train"], tokenizer, args.batch, args.max_len, True)
    val_loader = make_loader(*data["val"], tokenizer, args.batch * 2, args.max_len, False)
    test_loader = make_loader(*data["test"], tokenizer, args.batch * 2, args.max_len, False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * len(train_loader)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    history, best_acc = [], -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t, total, seen = time.time(), 0.0, 0
        for i, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = model(**batch).loss
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item() * len(batch["labels"])
            seen += len(batch["labels"])
            if i % 200 == 0:
                print(f"  epoch {epoch} step {i}/{len(train_loader)} loss {loss.item():.4f}", flush=True)

        probs, labels = predict(model, val_loader, device)
        val_acc = float(((probs >= 0.5) == labels).mean())
        history.append({"epoch": epoch, "train_loss": round(total / seen, 4), "val_accuracy": round(val_acc, 4)})
        print(f"epoch {epoch}: train loss {total / seen:.4f}, val accuracy {val_acc:.4f} ({time.time() - t:.0f}s)")
        if val_acc > best_acc:
            best_acc = val_acc
            model.save_pretrained(args.out)
            tokenizer.save_pretrained(args.out)

    model = AutoModelForSequenceClassification.from_pretrained(args.out).to(device)
    probs, labels = predict(model, test_loader, device)
    plot_history(history, f"{args.base} fine-tuning", args.out / "training_curve.png")
    evaluate(
        f"{args.base} (fine-tuned)",
        labels,
        probs,
        args.out,
        {"base_model": args.base, "epochs": args.epochs, "best_val_accuracy": round(best_acc, 4),
         "train_samples": len(data["train"][0]), "max_len": args.max_len},
    )


if __name__ == "__main__":
    main()
