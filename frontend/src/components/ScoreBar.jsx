import { VERDICTS } from "../lib/chain.js";

/** Fake-probability meter. score is in basis points (0..10000); confidence is 0..1 when known. */
export default function ScoreBar({ score, verdict, confidence }) {
  const label = typeof verdict === "number" ? VERDICTS[verdict] : verdict;
  const pct = score / 100;
  return (
    <div className="score">
      <div className="score-head">
        <span className={`badge ${label.toLowerCase()}`}>{label}</span>
        <span className="muted">
          {pct.toFixed(1)}% likely fake
          {confidence != null && ` · ${(confidence * 100).toFixed(0)}% confidence`}
        </span>
      </div>
      <div className="bar">
        <div className={`fill ${label.toLowerCase()}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}
