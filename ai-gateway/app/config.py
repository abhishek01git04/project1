import json
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
# An empty key line in .env must not count as a credential (it would also hide an `ant auth login` profile).
for _var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
    if os.environ.get(_var, None) == "":
        del os.environ[_var]

# Hardhat account #1 - a well-known DEV-ONLY key. Never use it on a real network.
GATEWAY_PRIVATE_KEY = os.getenv(
    "GATEWAY_PRIVATE_KEY",
    "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d",
)

def resolve_model(src: str) -> str:
    """Local model paths are relative to ai-gateway/; anything else is a Hugging Face Hub id."""
    path = Path(src) if Path(src).is_absolute() else BASE_DIR / src
    return str(path) if path.exists() else src


def model_name(src: str) -> str:
    """Short model name recorded on-chain, e.g. "text-baseline" for ../models/text-baseline/model.joblib."""
    path = Path(resolve_model(src))
    if not path.exists():
        return src
    return path.parent.name if path.is_file() else path.name


def _first_trained(*candidates: str, default: str) -> str:
    return next((c for c in candidates if (BASE_DIR / c).exists()), default)


# Models: our own trained ones from training/ when present (fine-tuned preferred over
# from-scratch), else a pretrained Hugging Face model. Override in .env.
TEXT_MODEL = os.getenv("TEXT_MODEL") or _first_trained(
    "../models/text-distilbert", "../models/text-baseline/model.joblib",
    default="hamzab/roberta-fake-news-classification",
)
# Images: comma-separated list; the most suspicious model decides. Default = Smaug, a general
# detector for AI-generated images of any subject (GPT-Image, Midjourney, FLUX, Nano Banana...).
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "abacusai/Smaug-AI-Image-Detector")
IMAGE_MODELS = [m.strip() for m in IMAGE_MODEL.split(",") if m.strip()]
# Smaug scores 3 crops per image by default; 1 is ~3x faster on CPU and slightly less accurate.
SMAUG_CROPS = int(os.getenv("SMAUG_CROPS", "3"))
# Image forensics for uploaded images; video frames keep using IMAGE_MODEL only. Empty = model off.
UNIVFD_MODEL = os.getenv("UNIVFD_MODEL", "../models/image-univfd/fc_weights.pth")
UNIVFD_CLIP = os.getenv("UNIVFD_CLIP", "openai/clip-vit-large-patch14")
TRUFOR_MODEL = os.getenv("TRUFOR_MODEL", "../models/image-trufor/trufor.pth.tar")
TRUFOR_MAX_SIDE = int(os.getenv("TRUFOR_MAX_SIDE", "1280"))  # larger images are reduced first (memory on CPU)
IMAGE_FUSION_FILE = os.getenv("IMAGE_FUSION_FILE", "../models/image-fusion/calibration.json")

# "auto" uses ML models when transformers/torch are installed, otherwise heuristics.
# "heuristic" forces the lightweight fallback.
DETECTOR_MODE = os.getenv("DETECTOR_MODE", "auto")

# Synthetic-voice detector (Hugging Face audio-classification model or local folder).
AUDIO_MODEL = os.getenv("AUDIO_MODEL", "MelodyMachine/Deepfake-audio-detection-V2")
# Video: frames sampled evenly through the clip and scored with the image models.
VIDEO_FRAMES = int(os.getenv("VIDEO_FRAMES", "6"))

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "15"))
MAX_VIDEO_MB = int(os.getenv("MAX_VIDEO_MB", "100"))

# ---- Internet Evidence Engine. Every source is optional; without a key it is skipped.
# "off" disables all web lookups (offline demos, tests).
EVIDENCE_MODE = os.getenv("EVIDENCE_MODE", "on")
EVIDENCE_TIMEOUT = float(os.getenv("EVIDENCE_TIMEOUT", "15"))
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")                # newsapi.org; GDELT (keyless) is used otherwise
NEWS_ENGINE = os.getenv("NEWS_ENGINE", "gdelt")           # gdelt (free) | newsapi (needs NEWSAPI_KEY)
GDELT_TIMESPAN = os.getenv("GDELT_TIMESPAN", "3m")       # how far back GDELT searches, e.g. 1w, 1m, 3m
GDELT_TIMEOUT = float(os.getenv("GDELT_TIMEOUT", "25"))
GDELT_COOLDOWN = float(os.getenv("GDELT_COOLDOWN", "60"))  # pause after a GDELT rate limit (429)
# Google News RSS (free, no key): fallback news search when GDELT is rate-limited or finds nothing.
GOOGLE_NEWS_FALLBACK = os.getenv("GOOGLE_NEWS_FALLBACK", "on") != "off"
GNEWS_LANG = os.getenv("GNEWS_LANG", "en-IN")
GNEWS_REGION = os.getenv("GNEWS_REGION", "IN")
# Find direct article links through Bing News RSS (free, no key) so the AI can read full articles.
# On 120 tuning claims: 74 right / 2 wrong / 44 Uncertain, against 71 / 3 / 46 with headlines only.
ARTICLE_SEARCH = os.getenv("ARTICLE_SEARCH", "on") != "off"
RETRIEVE_ARTICLES = int(os.getenv("RETRIEVE_ARTICLES", "4"))  # articles downloaded and read per claim
# How the evidence guard decides that a source states the claim (support.py):
# "keywords" = at least half of the claim's key terms (default); "minicheck" = the MiniCheck model.
SUPPORT_CHECK = os.getenv("SUPPORT_CHECK", "keywords")
MINICHECK_MODEL = os.getenv("MINICHECK_MODEL", "bespoke-minicheck")
MINICHECK_TIMEOUT = float(os.getenv("MINICHECK_TIMEOUT", "60"))
MINICHECK_MIN = float(os.getenv("MINICHECK_MIN", "0.5"))  # support probability that counts as "states the claim"
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
# Let the local AI model restate the claim and write the news-search queries (query.py) instead of
# picking keywords. Off until measured.
QUERY_REWRITE = os.getenv("QUERY_REWRITE", "off") == "on"
QUERY_REWRITE_TIMEOUT = float(os.getenv("QUERY_REWRITE_TIMEOUT", "60"))
# AVeriTeC-style question generation (questions.py): the local model writes 3-5 questions a
# fact-checker would ask, each is searched (GDELT, then the usual fallbacks), and the judge sees
# the evidence grouped by question. Off until measured.
QUESTION_SEARCH = os.getenv("QUESTION_SEARCH", "off") == "on"
QUESTION_TIMEOUT = float(os.getenv("QUESTION_TIMEOUT", "60"))
# How headlines and article paragraphs are matched to the claim: "keywords" (default) or "embedding".
RELEVANCE = os.getenv("RELEVANCE", "keywords")
EMBED_MIN = float(os.getenv("EMBED_MIN", "0.65"))
# Fact-check matching: "keywords" (>=60% key terms either way) or "embedding" (cosine >= FACTCHECK_EMBED_MIN).
# With QUERY_REWRITE on, the fact-check search also uses the AI-written queries.
FACTCHECK_MATCH = os.getenv("FACTCHECK_MATCH", "keywords")
FACTCHECK_EMBED_MIN = float(os.getenv("FACTCHECK_EMBED_MIN", "0.8"))  # embedding similarity a headline needs to be kept
GOOGLE_FACTCHECK_KEY = os.getenv("GOOGLE_FACTCHECK_KEY", "")  # Google Fact Check Tools API
GOOGLE_VISION_KEY = os.getenv("GOOGLE_VISION_KEY", "")    # Cloud Vision web detection = reverse image search
SOURCE_RATINGS_FILE = os.getenv("SOURCE_RATINGS_FILE", "")  # optional JSON {"domain": 0..1} overrides
# Sent when fetching articles for /verify-url. Some sites (e.g. Wikipedia) reject requests without
# the standard "compatible" bot format; add a contact URL or e-mail for production use.
FETCH_USER_AGENT = os.getenv("FETCH_USER_AGENT", "Mozilla/5.0 (compatible; TruthChain-Verifier/1.0)")

# ---- LLM evidence reasoning: Claude searches the web, reads sources and judges each claim.
# Needs ANTHROPIC_API_KEY (or an `ant auth login` profile); skipped without credentials.
LLM_REASONING = os.getenv("LLM_REASONING", "on")
# Which model judges the claims: auto = Claude if Anthropic credentials exist, else local Ollama;
# "ollama" = always the free local model; "anthropic" = always Claude.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto")
# Evidence-reasoning prompt: v3 adds today's date and a stricter rule for "refuted"; v2 is the earlier wording.
LLM_PROMPT_VERSION = os.getenv("LLM_PROMPT_VERSION", "v3")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "240"))
OLLAMA_MAX_TOKENS = int(os.getenv("OLLAMA_MAX_TOKENS", "600"))  # cap on the local model's answer
# How much fusion trusts a local model's verdict compared with Claude's (1.0).
LLM_LOCAL_TRUST = float(os.getenv("LLM_LOCAL_TRUST", "0.6"))
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5-5")
LLM_EFFORT = os.getenv("LLM_EFFORT", "medium")              # low | medium | high | xhigh | max
# Where Claude's evidence comes from: "gdelt" = the articles GDELT found (default, no search fees),
# "web_search" = Claude runs its own Anthropic web searches (billed per search).
LLM_RETRIEVAL = os.getenv("LLM_RETRIEVAL", "gdelt")
# When GDELT fails or is rate-limited, let Claude run its own web search for that request (billed per search).
LLM_WEB_FALLBACK = os.getenv("LLM_WEB_FALLBACK", "on") != "off"
LLM_MAX_SEARCHES = int(os.getenv("LLM_MAX_SEARCHES", "5"))  # web searches per verification
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "180"))

# Verification reports (JSON) are stored here and served at /reports/{hash}.
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", str(BASE_DIR / "reports")))

CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")

DEPLOYMENT_FILE = BASE_DIR / "deployment.json"


def load_deployment() -> dict:
    """Contract address and chain id, from CONTRACT_ADDRESS/CHAIN_ID env vars (hosted
    deployments) or the deployment.json written by blockchain/scripts/deploy.js.

    The file is read on every request so the gateway does not need restarting after a redeploy.
    """
    if os.getenv("CONTRACT_ADDRESS") and os.getenv("CHAIN_ID"):
        return {"address": os.environ["CONTRACT_ADDRESS"], "chainId": int(os.environ["CHAIN_ID"])}
    if not DEPLOYMENT_FILE.exists():
        raise RuntimeError(
            "deployment.json not found - run `npm run deploy:local` in blockchain/ first"
        )
    data = json.loads(DEPLOYMENT_FILE.read_text())
    return {"address": data["address"], "chainId": int(data["chainId"])}
