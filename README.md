# TruthChain — Decentralized Fake News & Deepfake Media Registry with AI Verification Gateway

A tamper-proof public registry where news text, URLs, images, videos and audio are fingerprinted,
checked by an AI verification gateway that combines detection models with internet evidence, and
recorded on a blockchain together with the verdict, a hash of the verification report and community votes.

## Architecture

```
 User ─▶ Frontend (React + Vite)          AI Verification Gateway (FastAPI, ai-gateway/)
         Verify · Lookup · Registry       API Router & Orchestrator (main.py)
                │                         /analyze/text  /analyze/image  /analyze/video
                │ 2. upload + metadata    /analyze/audio /verify-url     /reports/{hash}
                └───────────────────────▶ ┌────────────────┬──────────────────┬──────────────────┐
                                          │ AI Detection   │ Internet Evidence│ Evidence Fusion  │
                                          │ detectors.py   │ evidence.py      │ & Decision       │
                                          │ image forensics│ live news search │ fusion.py        │
                                          │ deepfake video │ reverse image    │ claim extraction │
                                          │ synthetic audio│ fact-check search│ multi-source     │
                                          │ text classifier│ source credibil. │ confidence →     │
                                          │                │                  │ Real/Uncertain/  │
                                          │                │                  │ Fake             │
                                          └────────────────┴──────────────────┴──────────────────┘
                ◀── 6. result + report ── Report Generation & Signing (report.py, signer.py)
                │                         JSON report → SHA-256 reportHash → EIP-191 signature
                │ 3. registerMedia(attestation, title, sourceUrl, signature) via MetaMask
                ▼
 MediaRegistry.sol (blockchain/)
   stores contentHash · reportHash · verdict · aiScore · modelVersion · issuer · timestamp
   verifies the gateway signature (ecrecover) · one-address-one-vote · final = 60% AI + 40% votes
```

**Why signed attestations?** The user (not the gateway) sends the transaction and pays the gas, but
cannot change the result: the contract recomputes the signed digest (bound to the chain id, the contract
address, the content hash, media type, score, verdict, report hash, model version and a timestamp)
and rejects it unless an authorised verifier signed it. Attestations expire after 1 hour, and each hash can
be registered only once.

**Verifiable reports.** Every verification produces a JSON report (detector signals, extracted claims,
news articles, fact-checks, reverse-image matches, source credibility and each fusion step). Its SHA-256 is
stored on-chain as `reportHash`; the gateway serves the exact bytes at `/reports/{hash}`, and the frontend
re-hashes them to show whether the report is intact.

**Content identity.** Media are identified by the SHA-256 hash of their exact bytes; URLs by the hash of the
URL string. Lookups hash in the browser, so nothing is uploaded just to check something.

### How a verdict is made
1. **AI detection** gives a fake probability (text classifier, Smaug AI-image detector, per-frame video
   scoring, wav2vec2 synthetic-voice detector) blended with transparent heuristics.
2. **Internet evidence**: the main claims are extracted and searched in the news (GDELT by default - free, no key)
   and in published fact-checks (Google Fact Check API); images and a video key frame are reverse-searched
   (Google Vision); the publisher of a URL is rated for credibility.
3. **AI evidence reasoning** (optional): the gateway downloads the top GDELT articles and extracts the
   paragraphs about the claim; an AI model reads those passages and
   judges each claim as supported / refuted / mixed / unverifiable, citing the articles it relied on
   (`app/reasoner.py`). `LLM_RETRIEVAL=web_search` lets Claude run its own (billed) web searches instead.
   - **Free:** a local model through [Ollama](https://ollama.com) (`ollama pull qwen2.5:7b`, 4.7 GB, about 8 GB
     RAM), used automatically when no Anthropic key is set. On CPU the AI step takes about 45 s, data stays on the PC;
     its verdict is weighted at 60% of Claude's (`LLM_LOCAL_TRUST`) because small models make more mistakes.
   - **Paid:** Claude (`claude-opus-5-5`) with `ANTHROPIC_API_KEY` - better reasoning.
4. **Fusion** (log-odds): the AI model's verdict (weighted by its confidence and trust), fact-checks, credible coverage or debunks, source credibility and reverse-image hits
   move the detector's probability; agreement between sources raises confidence, conflict lowers it.

| Fused fake probability | Confidence ≥ 50% | Confidence < 50% |
|------------------------|------------------|------------------|
| ≥ 70%                  | Fake             | Uncertain        |
| 30% – 70%              | Uncertain        | Uncertain        |
| ≤ 30%                  | Real             | Uncertain        |

Community votes count once an item has at least 3 votes: final score = 60% AI + 40% share of "fake" votes,
mapped with the same 30% / 70% thresholds.

**GDELT limits.** GDELT is free but slow (about 10-15 s per search) and rate-limited per IP address, often
more strictly than its documented one request per 5 seconds. The gateway runs the search in parallel with the
AI models, makes a second (broader) search only when the first finds nothing, and caches identical queries for
15 minutes. After a rate limit (HTTP 429) it pauses GDELT for `GDELT_COOLDOWN` seconds (default 60) instead of
retrying; during the pause the report says news search was skipped and the verdict relies on the remaining
evidence. Verifying many items in quick succession will hit this limit. While GDELT is paused the gateway
falls back automatically: NewsAPI takes over news search when `NEWSAPI_KEY` is set, and Claude runs its own web
search when `ANTHROPIC_API_KEY` is set (`LLM_WEB_FALLBACK=off` disables that). Fact-checks (Google) don't
depend on GDELT and keep working. Without any key, **Google News RSS** (free) takes over news search while GDELT is
paused or finds nothing; its links are Google redirects, so those articles are judged by headline and publisher.

### Measured accuracy
`eval/` measures the text pipeline on recent labelled claims (half rated false by fact-checkers; half rated true by
them or headlines of established outlets), with each claim's own source removed from the evidence.

**Independent test (the number to quote).** The default setup (`qwen2.5:7b`, prompt v3, fusion-v3) was frozen, then
run once on 120 new claims (`eval/claims_fresh.jsonl`) that played no part in choosing it:

| Setup | Right | Wrong | Uncertain | Accuracy | Right when it gives a verdict | Real called Fake | Fake called Real |
|-------|-------|-------|-----------|----------|-------------------------------|------------------|------------------|
| Detector only | 50 | 41 | 29 | 42% | 55% | 4/60 | 37/60 |
| + news + fact-checks | 66 | 31 | 23 | 55% | 68% | 8/60 | 23/60 |
| **+ local AI (full pipeline)** | **71** | **6** | **43** | **59%** (95% range 50-68%) | **92%** (84-96%) | **4/60** | **2/60** |

Uncertain counts as not correct in Accuracy. This covers text claims only; images, video, audio and URL checks were
not measured. The AI read full article text for 8 of the 120 claims and headlines only for most others. Of the "real"
claims, 30 were rated true by Snopes and 30 are headlines of established outlets, which were not separately verified.
Tuning on this set would make it no longer independent: to re-test after changes, build a new one with
`eval/build_claims.py --seed <new> --out <file> --exclude eval/claims.jsonl eval/claims_fresh.jsonl`.

**Full-article reading (`ARTICLE_SEARCH=on`, now the default).** The gateway finds direct article links through
Bing News RSS and the AI reads up to 4 articles. Measured on the tuning set only, against the same claims with
headlines only: 74 right / 2 wrong / 44 Uncertain against 71 / 3 / 46 (the difference could be chance: +3 right,
95% range -2 to +8). The AI step took about twice as long. It has not been measured on an independent set, so the
independent numbers above are for headlines only.

**Evidence guard.** A "supported" verdict from the AI now counts only if an article it read, or a headline, contains
at least half of the claim's key terms; otherwise it becomes "unverifiable". It was added after a made-up claim
("15 new districts to be formed in Maharashtra, confirmed by CM Fadanvis") came out Real because the AI treated
same-topic headlines and the submitted text as confirmation. On the tuning set it cost 1 correct verdict. The gateway
also no longer attaches sources the AI did not cite.

**Prompt v4 (tested, not used).** v4 tells the AI that the submitted text is not evidence and that it must answer
"unverifiable" unless a numbered source states the claim. On the tuning set, with the same saved articles, it scored
69 right / 3 wrong / 48 Uncertain against v3's 74 / 2 / 44, and called 2 real claims Fake (v3: 0). It was worse on 9
claims and better on 3 (net -6, 95% range -13 to 0). It mostly turned correct verdicts into "unverifiable", so v3
stays the default. `LLM_PROMPT_VERSION=v4` still selects it.

**MiniCheck guard (tested, not used).** `SUPPORT_CHECK=minicheck` replaces the keyword guard with the
`bespoke-minicheck` model (Ollama), which judges whether a source states the claim. On the tuning set it separates
real from fake claims best (AUC 0.78, against 0.60 for keyword overlap and 0.63 for `bge-small` embeddings). As a
guard it still did worse: 64 right / 3 wrong / 53 Uncertain at 0.5 against the keyword guard's 73 / 2 / 45 (net -10,
95% range -17 to -3). The tuned threshold scored net -4. The AI's "supported" verdicts on these claims were mostly
right, so the stricter check removed more correct verdicts than wrong ones. Among claims left Uncertain, MiniCheck
found a source stating the claim for 6 of 19 real and 1 of 25 fake claims, so it was also tried as an extra fusion
signal (`minicheck_support` in `fusion.WEIGHTS`, 0 = off): with weight and threshold chosen on one half of the claims
and scored on the other, 76 right / 3 wrong / 41 Uncertain against 73 / 2 / 45 (net +2, 95% range -2 to +6), and
every setting called one more fake claim Real. Too small and uncertain to justify ~5 GB more RAM and 30-60 s per
check on CPU, so it stays off. Scripts: `eval/support_features.py`, `eval/support_eval.py`, `eval/minicheck_fusion.py`.

**Evidence-stance model trained on AVeriTeC (tested, not used).** [AVeriTeC](https://github.com/MichSchli/AVeriTeC)
(CC BY-NC 4.0) has real fact-checked claims with the evidence fact-checkers collected. `eval/averitec_stance.py`
turns it into claim-evidence pairs: the claim's own evidence (supports, refutes or not enough), and "topic" pairs that
give a claim the evidence of a different claim which passes the live keyword guard. `training/train_stance.py`
fine-tuned `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` on 4646 train pairs (college HPC GPU, all layers, 3 epochs,
about 5 minutes; `hpc/stance_run.sh`); the epoch and thresholds were chosen on a 10% validation split. On the 800
AVeriTeC dev pairs, measured once, the model's top answer was right for 78% of supporting evidence, 86% of refuting
evidence and 92% of topic-only evidence (the NLI model before training: 56%, 50%, 81%); over the 762 dev pairs
with a clear label, accuracy 85.8% and macro F1 84.2% (95% range 81-87%) against 64.3% and 62.1% before training
(`eval/averitec_stance.py final`). The AVeriTeC test split has no public labels, so dev is the final test, and it
uses the fact-checkers' own evidence: with retrieved web evidence the numbers would be lower. The live AI judge
(`qwen2.5:7b`, prompt v3) given 40 dev pairs of each kind (`eval/averitec_stance.py judge`) said "supported" for
26/40 supporting, 1/40 refuting and 0/40 topic-only evidence, but "refuted" for 15 of the 40 topic-only pairs: it
treats evidence that does not mention the claim as a contradiction. A contradiction check with the trained model would
keep 31 of its 32 correct "refuted" verdicts and remove 12 of those 15, but would keep 7 of the 9 wrong "refuted"
verdicts on supporting evidence. On our tuning claims (`eval/stance_tuning.py harm`), a contradiction threshold of
0.6-0.7 removes 4 of the AI's 7 false "refuted" verdicts and keeps 26 of its 28 correct ones, but no final verdict
changes: fusion already leaves all 7 of those real claims Uncertain. So this set cannot show a verdict-level gain,
and the guard was not tried on the independent holdout. The keyword guard passes most of
every kind and cannot tell them apart (AUC 0.55 for supports against refutes; the trained model 0.95). On our tuning
claims (`eval/stance_tuning.py`), used as the support guard it scored 74 right / 2 wrong / 44 Uncertain against the
keyword guard's 73 / 2 / 45 (net +1, 95% range -2 to +4): it removed one fake→Real verdict but added one real→Fake.
As a guard for "refuted" verdicts it changed nothing. Five of the six fake claims the AI wrongly called supported are
"video shows X" captions where X really happened: the evidence does state the event, so no text stance model can
catch them. Not deployed. Note: transformers 5 loads this checkpoint in float16, which runs about 17× slower on CPU,
so the scripts load it in float32.

**AI-written search queries (tested on the tuning set, not yet used).** With `QUERY_REWRITE=on` the local model rewrites
the claim into three short English queries (translating Hindi/Hinglish), and `RELEVANCE=embedding` ranks headlines and
paragraphs by meaning (`bge-small`) instead of shared keywords. `eval/search_compare.py` ran the old and new search on
each tuning claim minutes apart (GDELT off for all arms) and scored every source the AI would read with MiniCheck:

| Search | Real claims with a source stating them | Fake claims with one | Fake claims matched to a false-rated fact-check (keyword rule / embedding ≥ 0.8) | Real claims wrongly matched to one (same rules) |
|--------|------|------|------|------|
| Live (old sentence split + keywords) | 31/60 | 4/60 | 23 / 24 | 2 / 0 |
| Abbreviation fix + keywords | 33/60 | 4/60 | 23 / 24 | 2 / 0 |
| AI queries + embeddings | **41/60** | 4/60 | **32 / 32** | 3 / 1 |

Against live, the new search found a stating source for 15 real claims that live missed and missed 5 that live found
(net +10, 95% range +2 to +18), without finding more for fake claims. The AI queries also find a false-rated
fact-check for 8–9 more fake claims; matched by embedding (≥ 0.8) instead of the live keyword rule, that costs one
real claim wrongly matched instead of three. This measures the evidence, not the verdicts: the AI judge and
fusion have not been re-run with it, and the rewrite adds about 20 s per check on CPU.

**Tuning set.** 120 other claims (`eval/claims.jsonl`), used to choose the settings, so these results are optimistic.
The last row uses fusion-v3; the others use fusion-v2:

| Setup | Accuracy | Gets a verdict | Right when it gives one | Real called Fake | Fake called Real |
|-------|----------|----------------|-------------------------|------------------|------------------|
| Detector only | 45% | 77% | 59% | 1/60 | 37/60 |
| + news + fact-checks | 61% | 80% | 76% | 3/60 | 20/60 |
| + `qwen2.5:3b`, permissive prompt | 57% | 70% | 82% | 5/60 | 10/60 |
| + `qwen2.5:3b`, strict prompt | 35% | 39% | 89% | 4/60 | 1/60 |
| + `qwen2.5:7b`, strict prompt | 54% | 59% | 92% | 3/60 | 3/60 |
| **+ `qwen2.5:7b`, prompt v3 (default)** | **59%** | **62%** | **96%** | **1/60** | **2/60** |

Uncertain counts as not correct in Accuracy. The default was chosen to make few wrong verdicts: it gives 3 wrong
verdicts against 15 for the permissive 3B setup, at the cost of more Uncertain results.
Reproduce with `eval/build_claims.py`, `eval/collect.py`, `eval/rerun_llm.py` and `eval/evaluate.py`.

### Optional API keys (`ai-gateway/.env`)
| Key | Enables | Without it |
|-----|---------|-----------|
| `NEWSAPI_KEY` + `NEWS_ENGINE=newsapi` | NewsAPI.org news search instead of GDELT | GDELT (free, no key) |
| `GOOGLE_FACTCHECK_KEY` | Google Fact Check Tools search | fact-check step skipped |
| `GOOGLE_VISION_KEY` | Reverse image search (Cloud Vision web detection) | step skipped |
| `ANTHROPIC_API_KEY` | AI evidence reasoning with Claude instead of the free local model | local Ollama model, or skipped if Ollama isn't running |

## AI models

We train our own models: a TF-IDF + logistic regression text model and a CNN (TruthNet)
built from scratch, plus fine-tuned DistilBERT and EfficientNet-B0. See
**[training/README.md](training/README.md)**. The gateway automatically uses whichever
trained models are in `models/`.

## Project layout
```
blockchain/   Hardhat project: contracts/MediaRegistry.sol, tests, deploy + smoke-test scripts
ai-gateway/   FastAPI service: main.py (router/orchestrator), detectors.py (AI detection), evidence.py +
              credibility.py + claims.py (internet evidence), fusion.py (decision), report.py + signer.py
frontend/     React (Vite) dApp: Verify & Register (text, URL, image, video, audio), Lookup, Registry + voting
```

## Running it locally (Windows / macOS / Linux)

Prerequisites: Node.js 18+, Python 3.10+, the MetaMask browser extension.

Use four terminals:

**1. Blockchain node**
```bash
cd blockchain
npm install
npx hardhat node
```

**2. Deploy the contract** (writes the address and ABI to `frontend/` and `ai-gateway/`)
```bash
cd blockchain
npm run deploy:local
```

**3. AI gateway**
```bash
cd ai-gateway
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-ml.txt   # optional: real AI models (large download)
copy .env.example .env               # macOS/Linux: cp
uvicorn app.main:app --port 8000
```
API docs: http://127.0.0.1:8000/docs

Without `requirements-ml.txt` installed, or with `DETECTOR_MODE=heuristic`, the gateway uses transparent
heuristics: clickbait phrases, capital-letter ratio, attribution checks, EXIF metadata, error-level analysis
and generator-typical resolutions. The models download on the first request:
- Text: `hamzab/roberta-fake-news-classification`
- Image (and video frames): `abacusai/Smaug-AI-Image-Detector`
- Audio: `MelodyMachine/Deepfake-audio-detection-V2`

You can change either one in `.env`.

**4. Frontend**
```bash
cd frontend
npm install
npm run dev
```
Open http://localhost:5173

### MetaMask setup
1. Click **Connect MetaMask** in the app. It adds or switches to the *Hardhat Local* network (chain id 31337).
2. Import a test account. `npx hardhat node` prints 20 accounts with private keys; use account **#2 or later**.
   Account #0 is the deployer and #1 is the gateway key.
3. If you restart `hardhat node`, run the deploy step again. Then, in MetaMask, go to
   *Settings → Advanced → Clear activity tab data* to reset the nonce.

## Tests
```bash
cd blockchain
npx hardhat test                                         # contract unit tests
npx hardhat run scripts/e2e-check.js --network localhost  # gateway → report → chain smoke test (GATEWAY_URL=… to override :8000)
```

## Deploying online (Sepolia + Vercel)

| Part | Host |
|---|---|
| Smart contract | Sepolia testnet (`npm run deploy:sepolia`) |
| AI gateway | Vercel Python function (`ai-gateway/api/index.py`, heuristic mode — PyTorch exceeds Vercel's size limit) |
| Frontend | Vercel static site |

1. `cd blockchain`, run `node scripts/new-key.js` to create a gateway key, then:
   `npx hardhat vars set SEPOLIA_RPC_URL`, `npx hardhat vars set DEPLOYER_KEY`, `npx hardhat vars set GATEWAY_ADDRESS`.
   Fund the deployer with Sepolia ETH, then run `npm run deploy:sepolia`.
2. `cd ai-gateway`, run `vercel`, set the env vars `GATEWAY_PRIVATE_KEY`, `CONTRACT_ADDRESS`, `CHAIN_ID=11155111`,
   `DETECTOR_MODE=heuristic` and `CORS_ORIGINS`, then run `vercel --prod`.
3. `cd frontend`, run `vercel`, set the env vars `VITE_GATEWAY_URL`, `VITE_RPC_URL` and `VITE_CHAIN_NAME=Sepolia`,
   then run `vercel --prod`.

## Security notes
- The default gateway key is Hardhat's public dev key **#1**. Generate a fresh key for any real deployment and
  pass its address via `GATEWAY_ADDRESS` when deploying.
- The contract owner can add or remove verifiers (`setVerifier`). On a real network, make the owner a multisig
  or DAO so that several independent AI gateways can attest.
- AI detection is probabilistic. The registry records evidence; it is not ground truth. That is why
  community votes and source URLs are stored alongside the AI score.

## Possible extensions
- Store the original files on IPFS and save the CID on-chain
- Perceptual hashing (pHash) so re-encoded or resized copies match
- Video deepfake detection by sampling frames
- Verifier staking/slashing and reputation-weighted voting
- Deploy to Sepolia or Polygon Amoy (a `sepolia` network is already in `hardhat.config.js`)
