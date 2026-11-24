import { useCallback, useEffect, useState } from "react";
import { readContract } from "../lib/chain.js";
import MediaCard from "./MediaCard.jsx";

const PAGE = 20;

export default function Registry({ wallet }) {
  const [items, setItems] = useState([]);
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const [error, setError] = useState("");

  const load = useCallback(async () => {
    try {
      const c = readContract();
      setTotal(Number(await c.totalMedia()));
      setItems(await c.listMedia(offset, PAGE));
      setError("");
    } catch (err) {
      setError(`Cannot reach the blockchain: ${err.shortMessage || err.message}`);
    }
  }, [offset]);

  useEffect(() => {
    load();
  }, [load]);

  return (
    <section className="card">
      <div className="row between">
        <h2>Registry ({total})</h2>
        <button className="secondary" onClick={load}>Refresh</button>
      </div>
      {error && <p className="error">{error}</p>}
      {!error && items.length === 0 && <p className="muted">Nothing registered yet.</p>}
      <div className="grid">
        {items.map((m) => (
          <MediaCard key={m.contentHash} media={m} wallet={wallet} onChanged={load} />
        ))}
      </div>
      {total > PAGE && (
        <div className="row">
          <button className="secondary" disabled={offset === 0} onClick={() => setOffset(offset - PAGE)}>Newer</button>
          <button className="secondary" disabled={offset + PAGE >= total} onClick={() => setOffset(offset + PAGE)}>Older</button>
        </div>
      )}
    </section>
  );
}
