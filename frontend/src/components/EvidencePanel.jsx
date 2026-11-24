import { useEffect, useState } from "react";

const MODEL_NAMES = { smaug: "Smaug", univfd: "UnivFD", trufor: "TruFor" };

function useObjectUrl(file) {
  const [url, setUrl] = useState(null);
  useEffect(() => {
    if (!file) {
      setUrl(null);
      return undefined;
    }
    const objectUrl = URL.createObjectURL(file);
    setUrl(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [file]);
  return url;
}

/** Each image model's score, and which of them counted towards the verdict. */
function ModelScores({ models }) {
  return (
    <table className="model-scores">
      <thead>
        <tr><th>Model</th><th>Measures</th><th>Score</th><th>Counted</th></tr>
      </thead>
      <tbody>
        {models.map((m) => (
          <tr key={m.model}>
            <td>{MODEL_NAMES[m.model] || m.model}</td>
            <td>{m.role}</td>
            <td>{(m.score * 100).toFixed(1)}%</td>
            <td>{m.counted ? "Yes" : "No"}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/** TruFor's map as a transparent layer, over the image when the browser still has it. */
function Heatmap({ src, imageFile }) {
  const imageUrl = useObjectUrl(imageFile);
  if (!src?.startsWith("data:image/png;base64,")) return null;
  return (
    <figure className="heatmap">
      <div className="heatmap-stage">
        {imageUrl && <img src={imageUrl} alt="Analysed upload" />}
        <img className={imageUrl ? "heatmap-layer" : ""} src={src} alt="Regions TruFor suspects were manipulated" />
      </div>
      <figcaption className="muted">
        Manipulation heatmap: red marks regions TruFor suspects were edited. It is a lead to check, not proof.
      </figcaption>
    </figure>
  );
}

/** Everything the gateway found: detector signals, claims, web evidence and how the verdict was fused. */
export default function EvidencePanel({ report, imageFile = null }) {
  const ev = report.evidence || {};
  const news = ev.news;
  const facts = ev.factChecks;
  const rev = ev.reverseImage;
  const details = report.details || {};

  return (
    <div className="evidence">
      <section>
        <h3>AI detection <span className="muted">· {report.detection?.modelId}</span></h3>
        <ul>{(report.detection?.signals || []).map((s, i) => <li key={i}>{s}</li>)}</ul>
        {report.detection?.models?.length > 0 && <ModelScores models={report.detection.models} />}
        <Heatmap src={report.detection?.heatmap} imageFile={imageFile} />
      </section>

      {ev.claims?.length > 0 && (
        <section>
          <h3>Claims checked</h3>
          <ol>{ev.claims.map((c, i) => <li key={i}>{c}</li>)}</ol>
        </section>
      )}

      {ev.llm && (
        <section>
          <h3>AI web research <EngineNote block={ev.llm} /></h3>
          {ev.llm.status === "ok" && (
            <>
              <p className="evidence-line">
                <span className={`stance ${LLM_STANCE[ev.llm.verdict]}`}>{ev.llm.verdict}</span>
                <span className="muted">
                  {Math.round(ev.llm.confidence * 100)}% confident ·{" "}
                  {ev.llm.retrieval === "gdelt"
                    ? `read ${ev.llm.passagesRead} article${ev.llm.passagesRead === 1 ? "" : "s"} found by GDELT`
                    : `${ev.llm.searches} web searches`}
                </span>
              </p>
              <p>{ev.llm.summary}</p>
              {ev.llm.claims?.length > 0 &&
                !(ev.llm.claims.length === 1 && ev.llm.claims[0].explanation === ev.llm.summary) && (
                <ul>
                  {ev.llm.claims.map((c, i) => (
                    <li key={i}>
                      <span className={`stance ${LLM_STANCE[c.status]}`}>{c.status}</span> {c.claim}
                      <span className="muted"> — {c.explanation}</span>
                    </li>
                  ))}
                </ul>
              )}
              {ev.llm.sources?.length > 0 && (
                <ul className="links">
                  {ev.llm.sources.map((s, i) => (
                    <li key={i}>
                      <span className={`stance ${s.stance === "context" ? "mixed" : s.stance}`}>{s.stance}</span>{" "}
                      <a href={s.url} target="_blank" rel="noreferrer">{s.title || s.url}</a>
                      <span className="muted"> · {s.publisher} — {s.note}</span>
                    </li>
                  ))}
                </ul>
              )}
            </>
          )}
        </section>
      )}

      {ev.source && (
        <section>
          <h3>Source credibility</h3>
          <p className="evidence-line">
            <span className={`cred ${ev.source.label}`}>{ev.source.label}</span>
            <strong>{ev.source.domain}</strong>
            <span className="muted">{ev.source.reason}</span>
          </p>
          {details.title && <p className="muted">Article: {details.title}{details.published && ` · published ${details.published.slice(0, 10)}`}</p>}
        </section>
      )}

      {facts && (
        <section>
          <h3>Fact checks <EngineNote block={facts} /></h3>
          {facts.reviews?.length > 0 ? (
            <ul className="links">
              {facts.reviews.map((r, i) => (
                <li key={i}>
                  <span className={`stance ${r.stance}`}>{r.rating}</span>{" "}
                  <a href={r.url} target="_blank" rel="noreferrer">{r.publisher}</a>
                  <span className="muted"> — {r.claim}</span>
                </li>
              ))}
            </ul>
          ) : facts.status === "ok" && <p className="muted">No published fact-check matches this claim.</p>}
        </section>
      )}

      {news && (
        <section>
          <h3>News coverage <EngineNote block={news} /></h3>
          {news.articles?.length > 0 ? (
            <ul className="links">
              {news.articles.map((a, i) => (
                <li key={i}>
                  <span className={`stance ${a.stance}`}>{a.stance === "refutes" ? "debunks" : "reports"}</span>{" "}
                  <a href={a.url} target="_blank" rel="noreferrer">{a.title}</a>
                  <span className="muted"> · {a.domain} (trust {Math.round(a.credibility * 100)}%)</span>
                </li>
              ))}
            </ul>
          ) : news.status === "ok" && <p className="muted">No matching coverage found.</p>}
        </section>
      )}

      {rev && (
        <section>
          <h3>Reverse image search <EngineNote block={rev} /></h3>
          {rev.status === "ok" && (
            rev.pages?.length > 0 ? (
              <>
                {rev.bestGuess?.length > 0 && <p className="muted">Best guess: {rev.bestGuess.join(", ")}</p>}
                <ul className="links">
                  {rev.pages.map((p, i) => (
                    <li key={i}>
                      {p.factCheck && <span className="stance refutes">fact-check</span>}{" "}
                      <a href={p.url} target="_blank" rel="noreferrer">{p.title || p.url}</a>
                      <span className="muted"> · {p.domain}</span>
                    </li>
                  ))}
                </ul>
              </>
            ) : <p className="muted">This image was not found elsewhere on the web.</p>
          )}
        </section>
      )}

      {report.fusion?.length > 0 && (
        <section>
          <h3>How the verdict was reached</h3>
          <ol className="steps">{report.fusion.map((s, i) => <li key={i}>{s}</li>)}</ol>
        </section>
      )}
    </div>
  );
}

const LLM_STANCE = { supported: "supports", refuted: "refutes", mixed: "mixed", unverifiable: "" };

function EngineNote({ block }) {
  if (block.status === "ok") return <span className="muted">· {block.engine}</span>;
  return <span className="muted">· {block.status}{block.reason ? `: ${block.reason}` : ""}</span>;
}
