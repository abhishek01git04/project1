"""Train our own fake/real image classifier: label the images, split them 90% train / 10% test, train a
small neural network with a hyperparameter search, then predict fake or real.

The network is a small head on a frozen CLIP ViT-L/14 image encoder (the encoder UnivFD already uses). Every
training image is used twice: as a high-quality JPEG, and as a degraded copy like a re-shared image (smaller,
recompressed, sometimes framed), because degraded copies are where Smaug misses fakes. Both classes are
re-encoded as JPEG, so the file format cannot give the label away (in the training split some fakes are PNG
or WebP, while every real is JPEG). Image handling and CLIP live in image_head_features.py.

Training images: the julienlucas training split (fake and real), plus extra real photos from Unsplash Lite
(before 2022, with camera EXIF) and Open Images. The extras were added after the first frozen-test run showed
many false alarms on high-quality real photos, so frozen-test numbers from then on are slightly optimistic.
Fake and real get equal total weight in training.

The 10% test part comes from the same sources as the training part, so it overstates real-world accuracy.
The frozen OpenFake set (other generators, other real photos) is the honest measure. Both are measured only:
hyperparameters are chosen by cross-validation inside the 90%.

Steps (from the project root):
  1. Training features. On this CPU they take about 6 s per image, so they are made on a GPU:
     the college HPC (training/hpc/, see image_head_job.py) or Google Colab (training/colab_image_head.ipynb).
     Unzip the feature zips into models/image-head/features/train/.
     (Locally: train_image_head.py extract --parquet <folder with train-*.parquet>)
  2. Test-set features, locally (about 50 minutes):
       ai-gateway\\.venv\\Scripts\\python training\\train_image_head.py extract --testset data\\image-test-openfake
  3. ai-gateway\\.venv\\Scripts\\python training\\train_image_head.py fit
     Writes models/image-head/head.joblib and report.json. Nothing here changes the gateway.
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

import image_head_features as features
from common import MODELS_DIR, SEED
from app import config
from app import univfd

OUT_DIR = MODELS_DIR / "image-head"
FEATURES = OUT_DIR / "features"
SMAUG_SCORES = MODELS_DIR / "image-fusion" / "frozen-test"

# image_head_features.py is standalone (it also runs in Colab and on the HPC), so check it still matches
assert features.SEED == SEED
assert np.allclose(features.MEAN, univfd.MEAN) and np.allclose(features.STD, univfd.STD)

CANDIDATES = (
    [("logistic", {"C": c}) for c in (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)]
    + [("mlp", {"hidden_layer_sizes": h, "alpha": a})
       for h in ((256,), (512,), (512, 128)) for a in (1e-4, 1e-3, 1e-2)]
)


def source_of(item_id: str) -> str:
    return "julienlucas" if item_id.startswith("train-") else item_id.split(":", 1)[0]


def class_weights(labels) -> np.ndarray:
    """Equal total weight for fake and real."""
    labels = np.asarray(labels)
    weights = np.empty(len(labels), dtype=float)
    for c in (0, 1):
        weights[labels == c] = len(labels) / (2 * max(1, int((labels == c).sum())))
    return weights


def build(kind: str, params: dict):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if kind == "logistic":
        clf = LogisticRegression(max_iter=5000, **params)
    else:
        clf = MLPClassifier(early_stopping=True, max_iter=300, random_state=SEED, **params)
    return make_pipeline(StandardScaler(), clf)


def fit_weighted(kind: str, params: dict, X, Y):
    model = build(kind, params)
    return model.fit(X, Y, **{f"{model.steps[-1][0]}__sample_weight": class_weights(Y)})


def summary(y, p, weights=None) -> dict:
    from sklearn.metrics import log_loss, roc_auc_score

    y, p = np.asarray(y), np.asarray(p, dtype=float)
    pred = p >= 0.5
    return {"n": int(len(y)), "accuracy": round(float(np.average(pred == y.astype(bool), weights=weights)), 4),
            "auc": round(float(roc_auc_score(y, p)), 4) if len(set(y.tolist())) > 1 else None,
            "logLoss": round(float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1],
                                            sample_weight=weights)), 4),
            "fakesCaught": f"{int((pred & (y == 1)).sum())}/{int((y == 1).sum())}",
            "realsCorrect": f"{int((~pred & (y == 0)).sum())}/{int((y == 0).sum())}"}


def show(name: str, s: dict):
    auc = f"{s['auc']:.3f}" if s["auc"] is not None else "  -  "
    print(f"  {name:<28} n={s['n']:<5} acc={s['accuracy']:.1%}  auc={auc}  "
          f"fakes caught {s['fakesCaught']:<9} reals correct {s['realsCorrect']}")


def fit(args):
    import joblib
    from sklearn.model_selection import GroupKFold

    chunks = sorted(p for p in (FEATURES / "train").glob("*.npz") if not p.name.endswith(".tmp.npz"))
    if not chunks:
        raise SystemExit("No training features; see step 1 in this file's docstring")
    data = [np.load(c) for c in chunks]
    ids = np.concatenate([d["ids"] for d in data])
    y = np.concatenate([d["labels"] for d in data]).astype(int)
    clean = np.concatenate([d["clean"] for d in data])
    shared = np.concatenate([d["shared"] for d in data])
    sources = np.array([source_of(i) for i in ids])
    composition = {f"{s} {'fake' if c else 'real'}": int(n)
                   for (s, c), n in sorted(Counter(zip(sources.tolist(), y.tolist())).items())}
    print(f"{len(y)} training images ({y.sum()} fake, {(y == 0).sum()} real), two copies each: {composition}")

    # 90/10 split by image (both copies of an image stay together), stratified by source and label
    rng = np.random.default_rng(SEED)
    test_idx = np.concatenate([rng.permutation(np.flatnonzero((sources == s) & (y == c)))
                               [:round(((sources == s) & (y == c)).sum() * 0.1)]
                               for s in sorted(set(sources.tolist())) for c in (0, 1)])
    train_idx = np.setdiff1d(np.arange(len(y)), test_idx)
    X = np.vstack([clean[train_idx], shared[train_idx]])
    Y = np.concatenate([y[train_idx], y[train_idx]])
    groups = np.concatenate([train_idx, train_idx])

    print(f"\nHyperparameter search: {len(CANDIDATES)} candidates, {args.folds}-fold cross-validation "
          f"on the 90% ({len(train_idx)} images), fake and real weighted equally")
    results = []
    for kind, params in CANDIDATES:
        start = time.time()
        losses, accs = [], []
        for tr, va in GroupKFold(n_splits=args.folds).split(X, Y, groups):
            model = fit_weighted(kind, params, X[tr], Y[tr])
            s = summary(Y[va], model.predict_proba(X[va])[:, 1], weights=class_weights(Y[va]))
            losses.append(s["logLoss"])
            accs.append(s["accuracy"])
        results.append({"kind": kind, "params": params, "cvLogLoss": round(float(np.mean(losses)), 4),
                        "cvBalancedAccuracy": round(float(np.mean(accs)), 4)})
        print(f"  {kind:<8} {str(params):<46} log-loss {np.mean(losses):.4f}  bal. acc {np.mean(accs):.1%}  "
              f"({time.time() - start:.0f}s)", flush=True)
    best = min(results, key=lambda r: r["cvLogLoss"])
    print(f"-> chosen: {best['kind']} {best['params']}")
    model = fit_weighted(best["kind"], best["params"], X, Y)

    report = {"encoder": args.clip, "preprocessing": "shorter side 224, centre crop, CLIP normalisation",
              "trainingImages": composition, "classWeighting": "fake and real weighted equally",
              "note": "extra real photos were added after the first frozen-test run; frozen numbers are "
                      "slightly optimistic from then on",
              "search": results, "chosen": best, "results": {}}
    print("\n10% test part (same sources as training; overstates real-world accuracy):")
    for name, feats in (("clean copy", clean), ("shared copy", shared)):
        s = summary(y[test_idx], model.predict_proba(feats[test_idx])[:, 1])
        report["results"][f"split10 {name}"] = s
        show(name, s)
    for src in sorted(set(sources.tolist())):
        idx = test_idx[sources[test_idx] == src]
        s = summary(y[idx], model.predict_proba(shared[idx])[:, 1])
        report["results"][f"split10 shared copy {src}"] = s
        show(f"shared, {src}", s)

    for test_file in sorted(FEATURES.glob("*.npz")):
        t = np.load(test_file)
        tids, ty, tsrc = t["ids"], t["labels"].astype(int), t["sources"]
        p = model.predict_proba(t["features"])[:, 1]
        smaug_file = SMAUG_SCORES / f"{test_file.stem}-smaug.jsonl"
        smaug = {}
        if smaug_file.exists():
            for line in smaug_file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    smaug[r["id"]] = r["score"]
        print(f"\nFrozen test set {test_file.stem} (measured only, never tuned on):")
        for variant in ("clean", "shared", ""):
            mask = np.array([i.startswith(variant) for i in tids])
            label = variant or "both copies"
            s = summary(ty[mask], p[mask])
            report["results"][f"{test_file.stem} {label} head"] = s
            show(f"head, {label}", s)
            have = np.array([i in smaug for i in tids]) & mask
            if have.any():
                s = summary(ty[have], [smaug[i] for i in tids[have]])
                report["results"][f"{test_file.stem} {label} smaug"] = s
                show(f"smaug, {label}", s)
        for label, title, called in ((1, "fakes caught by generator", lambda q: q >= 0.5),
                                     (0, "reals called fake by source", lambda q: q >= 0.5)):
            table = {}
            for src in sorted(set(tsrc[ty == label].tolist())):
                m = (tsrc == src) & (ty == label)
                table[src] = {"n": int(m.sum()), "head": int(called(p[m]).sum()),
                              "smaug": int(sum(called(smaug[i]) for i in tids[m] if i in smaug)),
                              "smaugScored": int(sum(i in smaug for i in tids[m]))}
            report["results"][f"{test_file.stem} {title}"] = table
            print(f"  {title} (head / smaug, both copies):")
            for src, v in table.items():
                print(f"    {src:<20} {v['head']:>3}/{v['n']:<3}  smaug {v['smaug']}/{v['smaugScored']}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "encoder": args.clip, "size": features.SIZE}, OUT_DIR / "head.joblib")
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved {OUT_DIR / 'head.joblib'} and report.json (not used by the gateway)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["extract", "fit"])
    ap.add_argument("--parquet", type=Path, help="folder with train-*.parquet (extract)")
    ap.add_argument("--testset", type=Path, help="frozen test set folder (extract)")
    ap.add_argument("--clip", default=config.UNIVFD_CLIP)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--threads", type=int, default=0, help="CPU threads for torch (0 = default)")
    args = ap.parse_args()

    if args.step == "fit":
        fit(args)
        return
    if not (args.parquet or args.testset):
        ap.error("extract needs --testset and/or --parquet")
    if args.threads:
        import torch
        torch.set_num_threads(args.threads)
    encoder = features.Encoder(args.clip)
    if args.testset:
        features.extract_testset(encoder, args.testset, FEATURES / f"{args.testset.name}.npz")
    if args.parquet:
        features.extract_train(encoder, args.parquet, FEATURES / "train")


if __name__ == "__main__":
    main()
