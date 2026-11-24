import { useState } from "react";
import { GATEWAY_URL, KINDS, errorMessage } from "../lib/chain.js";
import MediaInput, { hasInput } from "./MediaInput.jsx";
import ScoreBar from "./ScoreBar.jsx";
import EvidencePanel from "./EvidencePanel.jsx";

const PROGRESS = {
  text: "Running AI detection and searching news & fact-checks…",
  url: "Fetching the page, checking the source and searching for evidence…",
  image: "Running image forensics and reverse image search…",
  video: "Sampling frames and running deepfake detection… (can take a minute)",
  audio: "Checking for synthetic / cloned voice…",
};

export default function Verify({ wallet, onConnect }) {
  const [kind, setKind] = useState("text");
  const [text, setText] = useState("");
  const [url, setUrl] = useState("");
  const [file, setFile] = useState(null);
  const [title, setTitle] = useState("");
  const [sourceUrl, setSourceUrl] = useState("");
  const [result, setResult] = useState(null);
  const [analyzedImage, setAnalyzedImage] = useState(null);
  const [status, setStatus] = useState({ busy: false, msg: "", error: "" });

  async function analyze() {
    setResult(null);
    setAnalyzedImage(kind === "image" ? file : null);
    setStatus({ busy: true, msg: PROGRESS[kind], error: "" });
    try {
      const endpoint = `${GATEWAY_URL}${KINDS[kind].endpoint}`;
      let res;
      if (kind === "text" || kind === "url") {
        res = await fetch(endpoint, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify(kind === "text" ? { text: text.trim() } : { url: url.trim() }),
        });
      } else {
        const form = new FormData();
        form.append("file", file);
        res = await fetch(endpoint, { method: "POST", body: form });
      }
      const body = await res.json();
      if (!res.ok) throw new Error(body.detail || "Gateway error");
      setResult(body);
      setStatus({ busy: false, msg: "", error: "" });
    } catch (err) {
      setStatus({ busy: false, msg: "", error: `AI gateway: ${err.message}. Is it running on ${GATEWAY_URL}?` });
    }
  }

  function defaultTitle() {
    if (kind === "text") return text.trim().slice(0, 80);
    if (kind === "url") return result.report?.details?.title?.slice(0, 120) || url.trim();
    return file.name;
  }

  async function register() {
    setStatus({ busy: true, msg: "Confirm the transaction in MetaMask…", error: "" });
    try {
      const { signature, signer, contract, chainId, ...attestation } = result.attestation;
      const source = sourceUrl || (kind === "url" ? url.trim() : "");
      const tx = await wallet.contract.registerMedia(attestation, title || defaultTitle(), source, signature);
      setStatus({ busy: true, msg: "Waiting for block confirmation…", error: "" });
      const receipt = await tx.wait();
      setStatus({ busy: false, msg: `Registered on-chain in block ${receipt.blockNumber}.`, error: "" });
    } catch (err) {
      setStatus({ busy: false, msg: "", error: errorMessage(err) });
    }
  }

  const inputs = { text, url, file };
  const canAnalyze = !status.busy && hasInput(kind, inputs);

  return (
    <section className="card">
      <h2>Verify media</h2>
      <p className="muted">
        The AI gateway checks the content with its detection models, searches the web for evidence, fuses both into a
        verdict and signs a verification report. Registering stores the content hash, verdict and report hash
        permanently on the blockchain.
      </p>
      <MediaInput {...{ kind, setKind, text, setText, url, setUrl, file, setFile }} />
      <div className="row">
        <input placeholder="Title (optional)" value={title} onChange={(e) => setTitle(e.target.value)} />
        {kind !== "url" && (
          <input placeholder="Source URL (optional)" value={sourceUrl} onChange={(e) => setSourceUrl(e.target.value)} />
        )}
      </div>
      <button disabled={!canAnalyze} onClick={analyze}>Analyse with AI</button>

      {status.msg && <p className="info">{status.msg}</p>}
      {status.error && <p className="error">{status.error}</p>}

      {result && (
        <div className="result">
          <ScoreBar score={result.aiScore} verdict={result.verdict} confidence={result.confidence} />
          <EvidencePanel report={result.report} imageFile={analyzedImage} />
          <dl>
            <dt>Content hash</dt><dd className="mono">{result.contentHash}</dd>
            <dt>Report hash</dt>
            <dd className="mono">
              <a href={`${GATEWAY_URL}/reports/${result.reportHash}`} target="_blank" rel="noreferrer">{result.reportHash}</a>
            </dd>
            <dt>Model version</dt><dd>{result.modelVersion}</dd>
            <dt>Signed by gateway</dt><dd className="mono">{result.attestation.signer}</dd>
          </dl>
          {wallet ? (
            <button disabled={status.busy} onClick={register}>Register on blockchain</button>
          ) : (
            <button onClick={onConnect}>Connect wallet to register</button>
          )}
        </div>
      )}
    </section>
  );
}
