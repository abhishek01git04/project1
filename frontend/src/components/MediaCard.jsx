import { useEffect, useState } from "react";
import { MEDIA_TYPES, VERDICTS, errorMessage, fetchReport, readContract } from "../lib/chain.js";
import ScoreBar from "./ScoreBar.jsx";
import EvidencePanel from "./EvidencePanel.jsx";

/** One registry entry: final (AI + community) verdict, on-chain proof, report and vote buttons. */
export default function MediaCard({ media, wallet, onChanged }) {
  const [final, setFinal] = useState(null);
  const [voted, setVoted] = useState(false);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [report, setReport] = useState(null); // { loading } | { report, verified } | { error }

  useEffect(() => {
    const c = readContract();
    Promise.all([c.finalScore(media.contentHash), c.finalVerdict(media.contentHash)]).then(([s, v]) =>
      setFinal({ score: Number(s), verdict: Number(v) })
    );
    if (wallet) c.hasVoted(media.contentHash, wallet.address).then(setVoted);
  }, [media, wallet]);

  async function vote(isFake) {
    setError("");
    setBusy(true);
    try {
      await (await wallet.contract.vote(media.contentHash, isFake)).wait();
      onChanged?.();
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setBusy(false);
    }
  }

  async function toggleReport() {
    if (report) return setReport(null);
    setReport({ loading: true });
    try {
      setReport(await fetchReport(media.reportHash));
    } catch (err) {
      setReport({ error: err.message });
    }
  }

  const date = new Date(Number(media.timestamp) * 1000).toLocaleString();
  const votes = Number(media.votesReal) + Number(media.votesFake);
  return (
    <article className="media">
      <div className="media-head">
        <span className="pill">{MEDIA_TYPES[Number(media.mediaType)]}</span>
        <strong>{media.title || "(untitled)"}</strong>
      </div>
      {final && <ScoreBar score={final.score} verdict={final.verdict} />}
      <dl>
        <dt>AI verdict</dt>
        <dd>{VERDICTS[Number(media.verdict)]} · {(Number(media.aiScore) / 100).toFixed(1)}% fake</dd>
        <dt>Community</dt>
        <dd>
          👍 {Number(media.votesReal)} real · 👎 {Number(media.votesFake)} fake
          {votes < 3 && <span className="muted"> (counts from 3 votes)</span>}
        </dd>
        <dt>Model version</dt><dd>{media.modelVersion}</dd>
        <dt>Registered</dt><dd>{date} by <span className="mono">{media.submitter.slice(0, 10)}…</span></dd>
        <dt>Issuer</dt><dd className="mono">{media.issuer}</dd>
        {media.sourceUrl && (<><dt>Source</dt><dd><a href={media.sourceUrl} target="_blank" rel="noreferrer">{media.sourceUrl}</a></dd></>)}
        <dt>Content hash</dt><dd className="mono">{media.contentHash}</dd>
        <dt>Report hash</dt><dd className="mono">{media.reportHash}</dd>
      </dl>
      <div className="row">
        <button className="secondary" onClick={toggleReport}>{report ? "Hide report" : "View verification report"}</button>
        {wallet && !voted && (
          <>
            <button disabled={busy} className="secondary" onClick={() => vote(false)}>Vote real</button>
            <button disabled={busy} className="danger" onClick={() => vote(true)}>Vote fake</button>
          </>
        )}
      </div>
      {voted && <p className="muted">You voted on this item.</p>}
      {error && <p className="error">{error}</p>}
      {report?.loading && <p className="info">Downloading report…</p>}
      {report?.error && <p className="error">{report.error}</p>}
      {report?.report && (
        <>
          <p className={report.verified ? "verified" : "error"}>
            {report.verified
              ? "✓ Report integrity verified: its SHA-256 matches the hash stored on-chain."
              : "✗ Report does not match the on-chain hash. It was altered after registration."}
          </p>
          <EvidencePanel report={report.report} />
        </>
      )}
    </article>
  );
}
