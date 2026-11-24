"""Model 1 - our own fake-news classifier, trained from scratch on CPU.

TF-IDF word + bigram features with logistic regression. It is fast, small and
explainable: the gateway shows which words pushed a verdict towards fake or real.

    python training/text_baseline.py --data data/WELFake_Dataset.csv
"""

import argparse
import json
import time
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

from common import MODELS_DIR, evaluate, load_welfake


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="WELFake_Dataset.csv")
    ap.add_argument("--fake-label", default="auto", help="label value meaning fake (default: auto-detect)")
    ap.add_argument("--train-limit", type=int, help="use only N training articles")
    ap.add_argument("--out", type=Path, default=MODELS_DIR / "text-baseline")
    args = ap.parse_args()

    data = load_welfake(args.data, args.fake_label, args.train_limit)
    x_train, y_train = data["train"]
    x_val, y_val = data["val"]
    x_test, y_test = data["test"]

    t = time.time()
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2), min_df=3, max_df=0.9, max_features=200_000,
        sublinear_tf=True, stop_words="english",
    )
    xt_train = tfidf.fit_transform(x_train)
    xt_val = tfidf.transform(x_val)
    print(f"TF-IDF vocabulary: {len(tfidf.vocabulary_)} terms ({time.time() - t:.0f}s)")

    # Tune the regularisation strength on the validation set.
    best = None
    for c in (0.5, 2.0, 8.0):
        t = time.time()
        clf = LogisticRegression(C=c, max_iter=2000).fit(xt_train, y_train)
        acc = accuracy_score(y_val, clf.predict(xt_val))
        print(f"C={c}: validation accuracy {acc:.4f} ({time.time() - t:.0f}s)")
        if best is None or acc > best[0]:
            best = (acc, c, clf)
    val_acc, c, clf = best
    pipe = Pipeline([("tfidf", tfidf), ("clf", clf)])

    args.out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipe}, args.out / "model.joblib", compress=3)

    names = tfidf.get_feature_names_out()
    order = clf.coef_[0].argsort()
    top = {"fake": [names[i] for i in order[::-1][:30]], "real": [names[i] for i in order[:30]]}
    (args.out / "top_terms.json").write_text(json.dumps(top, indent=2))
    print("Strongest FAKE terms:", ", ".join(top["fake"][:12]))
    print("Strongest REAL terms:", ", ".join(top["real"][:12]))

    evaluate(
        "TF-IDF + Logistic Regression (own, from scratch)",
        y_test,
        pipe.predict_proba(x_test)[:, 1],
        args.out,
        {"C": c, "val_accuracy": round(val_acc, 4), "train_samples": len(x_train)},
    )


if __name__ == "__main__":
    main()
