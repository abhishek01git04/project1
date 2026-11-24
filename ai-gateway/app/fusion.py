"""Evidence Fusion & Decision: combine the AI detector with internet evidence into a
fake probability, a confidence and a Real / Uncertain / Fake verdict.

Works in log-odds: the detector's probability is the prior, and each piece of evidence
adds or subtracts a bounded amount. Every step is recorded so the report can explain
exactly how the verdict was reached.

All tunable numbers live in WEIGHTS; eval/evaluate.py measures accuracy on labelled claims
and can search for better values (pass weights=... to decide()).
"""

import math
from dataclasses import dataclass, field

FUSION_VERSION = "fusion-v3"
VERDICTS = ("Real", "Uncertain", "Fake")  # same order as MediaRegistry.Verdict

WEIGHTS = {
    # detector: its log-odds are multiplied by this (1.0 = as reported, <1 = trusted less)
    "detector_scale": 1.0,
    # AI model reading the evidence (scaled by its confidence x trust)
    "llm_refuted": 3.0, "llm_supported": 3.5, "llm_mixed": 0.6,
    "unverified_penalty": 0.3,          # confidence lost when the AI model found nothing either way
    # published fact-checks: best match x weight, + per extra agreeing fact-checker, capped
    "fc_refute": 4.5, "fc_refute_extra": 0.75, "fc_refute_cap": 5.0,
    "fc_support": 3.5, "fc_support_extra": 0.5, "fc_support_cap": 4.0,
    # news coverage by credible outlets (per outlet, capped)
    "news_debunk": 0.7, "news_debunk_cap": 2.0,
    # 0 = off: on 120 labelled claims, credible same-topic headlines appeared for 54/60 fake and
    # 50/60 real claims, so they do not show a claim is true. The AI model weighs them instead.
    "news_support": 0.0, "news_support_cap": 1.5,
    "credible_at": 0.75,                # publisher trust needed to count as credible
    # publisher of a submitted URL: (0.5 - trust) x weight
    "source": 3.0,
    # reverse image search
    "image_factcheck": 1.5, "image_credible": 0.8,
    # a retrieved source states the claim (MiniCheck score >= minicheck_at); 0 = off (not yet shown to help)
    "minicheck_support": 0.0, "minicheck_at": 0.5, "minicheck_clears_penalty": 0.0,
    # confidence formula and verdict thresholds
    "conf_base": 0.2, "conf_model": 0.3, "conf_extremity": 0.3, "conf_agree": 0.12, "conf_disagree": 0.25,
    "fake_at": 0.70, "real_at": 0.30, "min_confidence": 0.50,
}


@dataclass
class Decision:
    fake_probability: float
    confidence: float
    verdict: str
    steps: list[str] = field(default_factory=list)

    @property
    def score_bps(self) -> int:
        return max(0, min(10000, round(self.fake_probability * 10000)))

    @property
    def verdict_id(self) -> int:
        return VERDICTS.index(self.verdict)


def _logit(p: float) -> float:
    p = min(max(p, 0.02), 0.98)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def decide(detector_prob: float, detector_is_model: bool, evidence: dict, weights: dict | None = None) -> Decision:
    W = WEIGHTS if weights is None else {**WEIGHTS, **weights}
    logit = _logit(detector_prob) * W["detector_scale"]
    steps = [f"AI detector: {detector_prob:.0%} fake ({'model' if detector_is_model else 'heuristics only'})"]
    votes = []  # +1 = evidence points to fake, -1 = to real; one entry per independent source
    unverified_penalty = 0.0

    def add(delta: float, why: str, direction: int):
        nonlocal logit
        logit += delta
        steps.append(f"{'+' if delta >= 0 else '-'}{abs(delta):.2f} log-odds: {why}")
        votes.append(direction)

    # 0. AI model reading the retrieved evidence (Claude or a local model).
    llm = evidence.get("llm") or {}
    if llm.get("status") == "ok" and llm.get("verdict") in ("supported", "refuted", "mixed"):
        # Scaled by how far this model is trusted (a small local model counts less than Claude).
        c = llm.get("confidence", 0.0) * llm.get("trust", 1.0)
        n = len(llm.get("sources") or [])
        who = "Local AI model" if str(llm.get("engine", "")).startswith("ollama") else "Claude"
        if llm["verdict"] == "refuted":
            add(W["llm_refuted"] * c, f"{who}'s reading of the evidence refutes the claim ({c:.0%} confident, {n} sources)", +1)
        elif llm["verdict"] == "supported":
            add(-W["llm_supported"] * c, f"{who}'s reading of the evidence supports the claim ({c:.0%} confident, {n} sources)", -1)
        else:
            add(W["llm_mixed"] * c, f"{who}'s reading of the evidence finds the claim misleading or partly true ({n} sources)", +1)
    elif llm.get("status") == "ok":
        steps.append("The AI model found no reliable evidence either way (confidence lowered)")
        unverified_penalty = W["unverified_penalty"]  # "nothing found" must not read as "confirmed real"

    # 1. Professional fact checks - the strongest external signal.
    reviews = [r for r in ((evidence.get("factChecks") or {}).get("reviews") or []) if r["stance"] != "mixed"]
    if reviews:
        # The best-matching review counts most; each further agreeing fact-checker adds a little.
        refuting = sorted((r["relevance"] for r in reviews if r["stance"] == "refutes"), reverse=True)
        supporting = sorted((r["relevance"] for r in reviews if r["stance"] == "supports"), reverse=True)
        if sum(refuting) > sum(supporting):
            delta = min(W["fc_refute_cap"], W["fc_refute"] * refuting[0] + W["fc_refute_extra"] * (len(refuting) - 1))
            add(delta, f"{len(refuting)} published fact-check(s) rate a matching claim false", +1)
        elif sum(supporting) > sum(refuting):
            delta = min(W["fc_support_cap"], W["fc_support"] * supporting[0] + W["fc_support_extra"] * (len(supporting) - 1))
            add(-delta, f"{len(supporting)} published fact-check(s) rate a matching claim true", -1)

    # 2. News coverage: credible outlets reporting the same story, or debunking it.
    articles = (evidence.get("news") or {}).get("articles") or []
    credible = [a for a in articles if a["credibility"] >= W["credible_at"]]
    debunks = [a for a in credible if a["stance"] == "refutes"]
    corroborate = [a for a in credible if a["stance"] == "supports"]
    if debunks:
        add(min(W["news_debunk_cap"], W["news_debunk"] * len(debunks)),
            f"{len(debunks)} credible outlet(s) describe it as false or a hoax", +1)
    if corroborate and W["news_support"] > 0:
        add(-min(W["news_support_cap"], W["news_support"] * len(corroborate)),
            f"{len(corroborate)} credible outlet(s) report the same story", -1)
    elif corroborate:
        steps.append(f"{len(corroborate)} credible outlet(s) cover this topic (not counted as confirmation by itself)")
    elif (evidence.get("news") or {}).get("status") == "ok" and evidence.get("claims"):
        steps.append("No credible outlet found reporting this claim (weak signal, no adjustment)")

    # 3. Publisher of a submitted URL.
    source = evidence.get("source")
    if source and source["label"] != "unknown":
        delta = (0.5 - source["score"]) * W["source"]
        add(delta, f"Source {source['domain']}: {source['reason'].lower()}", +1 if delta > 0 else -1)

    # 4. Reverse image search.
    rev = evidence.get("reverseImage") or {}
    pages = rev.get("pages") or []
    if any(p["factCheck"] for p in pages):
        add(W["image_factcheck"], "Image appears on fact-checking / debunking pages", +1)
    elif any(p["credibility"] >= W["credible_at"] for p in pages):
        add(-W["image_credible"], "Image already published by credible outlets", -1)

    # 5. A retrieved article or headline states the claim, judged by MiniCheck (off unless weighted).
    best = (evidence.get("minicheck") or {}).get("best")
    if W["minicheck_support"] > 0 and best is not None and best >= W["minicheck_at"]:
        add(-W["minicheck_support"], f"A retrieved source states the claim (MiniCheck {best:.0%})", -1)
        if W["minicheck_clears_penalty"] > 0:
            unverified_penalty = 0.0  # a source stating the claim is the positive evidence the penalty asks for

    p = _sigmoid(logit)
    side = 1 if p >= 0.5 else -1
    agree = sum(v == side for v in votes)
    disagree = sum(v == -side for v in votes)
    confidence = (W["conf_base"] + W["conf_model"] * detector_is_model + W["conf_extremity"] * abs(2 * p - 1)
                  + W["conf_agree"] * agree - W["conf_disagree"] * disagree
                  - (unverified_penalty if p < 0.5 else 0.0))  # only "real" needs positive evidence
    confidence = round(min(0.99, max(0.05, confidence)), 3)

    if confidence < W["min_confidence"]:
        verdict = "Uncertain"
        steps.append(f"Confidence {confidence:.0%} is below {W['min_confidence']:.0%} -> Uncertain")
    elif p >= W["fake_at"]:
        verdict = "Fake"
    elif p <= W["real_at"]:
        verdict = "Real"
    else:
        verdict = "Uncertain"
    steps.append(f"Final: {p:.0%} fake, confidence {confidence:.0%} -> {verdict}")
    return Decision(p, confidence, verdict, steps)
