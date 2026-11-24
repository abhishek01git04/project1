"""Fine-tune an NLI model on AVeriTeC to tell evidence that states a claim from evidence that is only on
the same topic: the job the gateway's keyword guard cannot do (see eval/averitec_stance.py).

Starts from MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli and keeps its three outputs:
  entailment     the claim's own evidence, AVeriTeC verdict Supported
  contradiction  the claim's own evidence, verdict Refuted
  neutral        the claim's own evidence with verdict Not Enough Evidence, and the evidence of a
                 different claim on the same topic that passes the live keyword guard
Claims whose evidence conflicts are left out of training.

Data (made on the laptop by `eval/averitec_stance.py build`): data/averitec/stance-train.jsonl and
stance-val.jsonl, every AVeriTeC train claim split 90/10 by claim. The 10% picks the best epoch and
any threshold; the AVeriTeC dev pairs are scored once at the end and never used for a choice.

Standalone (torch and transformers only, Python 3.8+), so the same file runs on this laptop's CPU, on
the college HPC or in Colab. The 128k-word embedding table always stays frozen; on a CPU the bottom
layers stay frozen too, to save memory and time.

Usage (from the project root):
  python training/train_stance.py train --quick       # a few steps, to check the setup and the speed
  python training/train_stance.py train               # resumable; writes models/stance-averitec/
  python training/train_stance.py score --pairs data/averitec/pairs-dev.jsonl --out eval/runs/averitec-stance-dev.jsonl
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "averitec"
OUT = ROOT / "models" / "stance-averitec"
BASE = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
LABEL = {"supports": "entailment", "refutes": "contradiction", "nei": "neutral", "topic": "neutral"}
SHORT = {"entailment": "e", "neutral": "n", "contradiction": "c"}
SEED = 7


def load(path) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def keep_awake() -> None:
    """Stop Windows sleeping while this runs (released when the process exits)."""
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x1 | 0x2)


def load_model(path, device: str):
    """In float32: the base checkpoint is stored in float16, which transformers 5 keeps, and float16
    runs about 17 times slower on a laptop CPU. (On a GPU, training uses bfloat16 autocast instead.)"""
    return AutoModelForSequenceClassification.from_pretrained(path).float().to(device)


def autocast(device: str):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else contextlib.nullcontext()


def label_ids(model) -> dict:
    lab = {v.lower(): int(k) for k, v in model.config.id2label.items()}
    return {name: lab[name] for name in SHORT}


def encode(tok, rows: list, max_length: int, device: str) -> dict:
    """NLI order: premise = the evidence (cut from its end if too long), hypothesis = the claim."""
    enc = tok([r["evidence"] for r in rows], [r["claim"] for r in rows], truncation="only_first",
              max_length=max_length, padding=True, return_tensors="pt")
    return {k: v.to(device) for k, v in enc.items()}


@torch.no_grad()
def predict(model, tok, rows: list, max_length: int, device: str, batch: int = 16) -> list:
    """[{"e", "n", "c"} probabilities] for each row (needs "claim" and "evidence")."""
    was_training = model.training
    model.eval()
    ids = label_ids(model)
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["evidence"]) + len(rows[i]["claim"]))
    probs = [None] * len(rows)
    for s in range(0, len(order), batch):
        part = order[s:s + batch]
        with autocast(device):
            logits = model(**encode(tok, [rows[i] for i in part], max_length, device)).logits
        for i, q in zip(part, logits.float().softmax(-1).cpu().numpy()):
            probs[i] = {SHORT[name]: round(float(q[j]), 4) for name, j in ids.items()}
    model.train(was_training)
    return probs


def balanced_accuracy(rows: list, probs: list) -> tuple:
    hits = {}
    for r, p in zip(rows, probs):
        gold = SHORT[LABEL[r["kind"]]]
        hits.setdefault(gold, []).append(max(p, key=p.get) == gold)
    per = {k: round(float(np.mean(v)), 3) for k, v in sorted(hits.items())}
    return float(np.mean(list(per.values()))), per


def plan_batches(lengths: list, size: int, seed: int) -> list:
    """Shuffled batches of similar length, so little time goes into padding."""
    rng = random.Random(seed)
    order = list(range(len(lengths)))
    rng.shuffle(order)
    out = []
    for g in range(0, len(order), size * 50):
        group = sorted(order[g:g + size * 50], key=lambda i: lengths[i])
        out += [group[i:i + size] for i in range(0, len(group), size)]
    rng.shuffle(out)
    return out


def freeze(model, layers: int) -> None:
    """Freeze the word embeddings, and with layers > 0 the bottom `layers` encoder layers (plus the
    relative-position table they share, so no gradient has to flow back through them)."""
    parts = [model.deberta.embeddings]
    if layers > 0:
        enc = model.deberta.encoder
        parts += [m for m in (getattr(enc, "rel_embeddings", None), getattr(enc, "LayerNorm", None)) if m is not None]
        parts += list(enc.layer[:layers])
    for m in parts:
        for p in m.parameters():
            p.requires_grad = False


def save_checkpoint(model, opt, sched, state: dict, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    trainable = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    torch.save({"model": trainable, "opt": opt.state_dict(), "sched": sched.state_dict()}, folder / "train.pt.tmp")
    (folder / "train.pt.tmp").replace(folder / "train.pt")
    (folder / "state.json").write_text(json.dumps(state, indent=1), encoding="utf-8")


def batch_loss(model, tok, rows, part, y, inv, flp, loss_fn, args, device):
    """CE on the original pairs; with augmentation (RobustFND) also nu * CE on the invariant claims,
    mu * CE on the flipped claims, and lambda * a contrastive loss on the [CLS] vectors: the anchor is the
    original pair, the positive its invariant rewrite, the negatives every flipped claim in the batch and
    the batch's other original pairs (InfoNCE on cosine similarity / tau)."""
    use_aug = bool(args.aug) and any(inv[i] or flp[i] for i in part)
    with autocast(device):
        out = model(**encode(tok, [rows[i] for i in part], args.max_length, device), output_hidden_states=use_aug)
    loss = loss_fn(out.logits.float(), y[part].to(device))
    if not use_aug:
        return loss

    def side(aug):
        idx = [k for k, i in enumerate(part) if aug[i]]
        if not idx:
            return idx, None
        items = [{"claim": aug[part[k]][0], "evidence": rows[part[k]]["evidence"]} for k in idx]
        with autocast(device):
            o = model(**encode(tok, items, args.max_length, device), output_hidden_states=True)
        labels = torch.tensor([aug[part[k]][1] for k in idx], device=device)
        return idx, (o, labels)

    ii, inv_out = side(inv)
    fi, flp_out = side(flp)
    if inv_out:
        loss = loss + args.nu * loss_fn(inv_out[0].logits.float(), inv_out[1])
    if flp_out:
        loss = loss + args.mu * loss_fn(flp_out[0].logits.float(), flp_out[1])
    if inv_out:
        norm = torch.nn.functional.normalize
        zo = norm(out.hidden_states[-1][:, 0].float(), dim=-1)
        zi = norm(inv_out[0].hidden_states[-1][:, 0].float(), dim=-1)
        zf = norm(flp_out[0].hidden_states[-1][:, 0].float(), dim=-1) if flp_out else zi[:0]
        cands = torch.cat([zi, zf, zo])  # positives, flipped negatives, original pairs
        sim = zo[ii] @ cands.T / args.tau
        self_mask = torch.zeros_like(sim, dtype=torch.bool)
        for r, k in enumerate(ii):  # an anchor is not its own negative
            self_mask[r, len(ii) + len(fi) + k] = True
        sim = sim.masked_fill(self_mask, float("-inf"))
        loss = loss + args.lam * torch.nn.functional.cross_entropy(sim, torch.arange(len(ii), device=device))
    return loss


def train(args) -> None:
    keep_awake()
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.threads:
        torch.set_num_threads(args.threads)
    layers = args.freeze_layers if args.freeze_layers >= 0 else (0 if device == "cuda" else 6)
    epochs = args.epochs or (3 if device == "cuda" else 2)
    lr = args.lr or (2e-5 if layers == 0 else 3e-5)
    rows = [r for r in load(DATA / "stance-train.jsonl") if r["kind"] in LABEL]
    val = [r for r in load(DATA / "stance-val.jsonl") if r["kind"] in LABEL]
    if args.quick:
        if args.aug:  # test the augmented losses: use pairs that have augmentation
            have = {a["pid"] for a in load(args.aug)}
            rows = [r for r in rows if r["pid"] in have] or rows
        rows, val = random.Random(SEED).sample(rows, min(24, len(rows))), random.Random(SEED).sample(val, 12)
        epochs = 1

    out_dir = Path(args.save_dir) if args.save_dir else OUT
    tok = AutoTokenizer.from_pretrained(BASE)
    model = load_model(BASE, device)
    ids = label_ids(model)
    y = torch.tensor([ids[LABEL[r["kind"]]] for r in rows])
    # RobustFND augmentation (training/augment_stance.py): an invariant and a flipped claim per pair, or None
    inv, flp = [None] * len(rows), [None] * len(rows)
    if args.aug:
        aug = {a["pid"]: a for a in load(args.aug)}
        for k, r in enumerate(rows):
            a = aug.get(r["pid"])
            if a and a.get("invariant"):
                inv[k] = (a["invariant"], ids[LABEL[a["invariantKind"]]])
            if a and a.get("flipped"):
                flp[k] = (a["flipped"], ids[LABEL[a["flippedKind"]]])
        print(f"augmentation: {sum(x is not None for x in inv)} invariant and {sum(x is not None for x in flp)} "
              f"flipped claims; lambda {args.lam} (contrastive), mu {args.mu} (flipped CE), nu {args.nu} "
              f"(invariant CE), temperature {args.tau}", flush=True)
    counts = torch.bincount(y, minlength=3).float()
    weights = (counts.sum() / (3 * counts.clamp(min=1))).to(device)  # each label weighs the same in total
    freeze(model, layers)
    trainable = [p for p in model.parameters() if p.requires_grad]
    lengths = [len(x) for x in tok([r["evidence"] for r in rows], [r["claim"] for r in rows])["input_ids"]]
    print(f"{len(rows)} training pairs {dict(zip(ids, counts.tolist()))}, {len(val)} validation pairs; "
          f"device {device}, {torch.get_num_threads()} threads; {layers} of {len(model.deberta.encoder.layer)} "
          f"layers frozen, {sum(p.numel() for p in trainable) / 1e6:.0f}M parameters trained; "
          f"{np.mean([n > args.max_length for n in lengths]):.0%} of pairs cut to {args.max_length} tokens",
          flush=True)

    steps_per_epoch = math.ceil(len(plan_batches(lengths, args.batch, SEED)) / args.accum)
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps_per_epoch * epochs), steps_per_epoch * epochs)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    folder = out_dir / "checkpoint"
    setup = {"base": BASE, "frozenLayers": layers, "epochs": epochs, "lr": lr, "batch": args.batch,
             "accum": args.accum, "maxLength": args.max_length, "trainPairs": len(rows),
             "aug": str(args.aug or ""), "lambda": args.lam, "mu": args.mu, "nu": args.nu, "tau": args.tau}
    state = {"setup": setup, "epoch": 0, "batch": 0, "best": -1.0, "history": []}
    if not args.quick and (folder / "state.json").exists():
        saved_state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
        if saved_state["setup"] != setup:
            sys.exit(f"{folder} was made with other settings {saved_state['setup']}; delete it to start over")
        saved = torch.load(folder / "train.pt", map_location=device)
        model.load_state_dict(saved["model"], strict=False)
        opt.load_state_dict(saved["opt"])
        sched.load_state_dict(saved["sched"])
        state = saved_state
        print(f"resuming at epoch {state['epoch'] + 1}, batch {state['batch']}", flush=True)

    model.train()
    for epoch in range(state["epoch"], epochs):
        plan = plan_batches(lengths, args.batch, SEED + epoch)
        first = state["batch"] if epoch == state["epoch"] else 0
        start, losses = time.time(), []
        for b in range(first, len(plan)):
            part = plan[b]
            loss = batch_loss(model, tok, rows, part, y, inv, flp, loss_fn, args, device)
            (loss / args.accum).backward()
            losses.append(float(loss.detach()))
            if (b + 1) % args.accum == 0 or b + 1 == len(plan):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                if not args.quick and (b + 1) % args.save_every == 0 and b + 1 < len(plan):
                    state.update(epoch=epoch, batch=b + 1)
                    save_checkpoint(model, opt, sched, state, folder)
            done = b + 1 - first
            if done % args.log_every == 0 or b + 1 == len(plan):
                per = (time.time() - start) / done
                print(f"epoch {epoch + 1}/{epochs} batch {b + 1}/{len(plan)}: loss {np.mean(losses[-args.log_every:]):.3f}, "
                      f"{per:.1f} s per batch, about {per * (len(plan) - b - 1) / 60:.0f} min left in this epoch",
                      flush=True)
        probs = predict(model, tok, val, args.max_length, device)
        bal, per = balanced_accuracy(val, probs)
        state["history"].append({"epoch": epoch + 1, "valBalancedAccuracy": round(bal, 4), "valPerLabel": per,
                                 "trainLoss": round(float(np.mean(losses)), 4) if losses else None})
        print(f"epoch {epoch + 1}: validation balanced accuracy {bal:.1%} {per}", flush=True)
        if args.quick:
            print("quick run finished; nothing saved", flush=True)
            return
        if bal > state["best"]:
            state["best"] = bal
            model.save_pretrained(out_dir)
            tok.save_pretrained(out_dir)
            with open(out_dir / "val-scores.jsonl", "w", encoding="utf-8") as f:
                for r, p in zip(val, probs):
                    f.write(json.dumps({"pid": r["pid"], "kind": r["kind"], "ftE": p["e"], "ftN": p["n"],
                                        "ftC": p["c"]}) + "\n")
            print(f"  best so far: saved to {out_dir}", flush=True)
        state.update(epoch=epoch + 1, batch=0)
        save_checkpoint(model, opt, sched, state, folder)
    (out_dir / "train_report.json").write_text(json.dumps({**state, "labels": LABEL}, indent=1), encoding="utf-8")
    print(f"done; best validation balanced accuracy {state['best']:.1%}", flush=True)


def score(args) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.threads:
        torch.set_num_threads(args.threads)
    path = Path(args.model) if args.model else OUT
    tok = AutoTokenizer.from_pretrained(path)
    model = load_model(path, device)
    rows = load(args.pairs)
    start = time.time()
    probs = predict(model, tok, rows, args.max_length, device)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r, p in zip(rows, probs):
            f.write(json.dumps({"pid": r.get("pid"), "kind": r.get("kind"), "ftE": p["e"], "ftN": p["n"],
                                "ftC": p["c"]}) + "\n")
    print(f"scored {len(rows)} pairs in {time.time() - start:.0f} s -> {args.out}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["train", "score"])
    ap.add_argument("--quick", action="store_true", help="train: 3 batches and a tiny validation, to test")
    ap.add_argument("--epochs", type=int, default=0, help="default 3 on a GPU, 2 on a CPU")
    ap.add_argument("--freeze-layers", type=int, default=-1, help="default 0 on a GPU, 6 on a CPU")
    ap.add_argument("--lr", type=float, default=0.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2, help="batches per optimiser step")
    ap.add_argument("--max-length", type=int, default=384)
    ap.add_argument("--save-every", type=int, default=100, help="batches between checkpoints")
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--aug", help="train: augmentation file from augment_stance.py (RobustFND losses on)")
    ap.add_argument("--lam", type=float, default=0.5, help="train: contrastive loss weight")
    ap.add_argument("--mu", type=float, default=0.5, help="train: CE weight on flipped claims")
    ap.add_argument("--nu", type=float, default=0.5, help="train: CE weight on invariant claims")
    ap.add_argument("--tau", type=float, default=0.1, help="train: contrastive temperature")
    ap.add_argument("--save-dir", help="train: output folder (default models/stance-averitec)")
    ap.add_argument("--pairs", help="score: jsonl with claim and evidence")
    ap.add_argument("--out", help="score: output jsonl")
    ap.add_argument("--model", help="score: model folder (default models/stance-averitec)")
    args = ap.parse_args()
    if args.step == "score" and not (args.pairs and args.out):
        ap.error("score needs --pairs and --out")
    {"train": train, "score": score}[args.step](args)


if __name__ == "__main__":
    main()
