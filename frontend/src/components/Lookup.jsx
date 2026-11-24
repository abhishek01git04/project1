import { useState } from "react";
import { contentBytes, readContract, sha256Hex } from "../lib/chain.js";
import MediaInput, { hasInput } from "./MediaInput.jsx";
import MediaCard from "./MediaCard.jsx";

/** Check whether a piece of content is already in the registry, by hashing it locally. */
export default function Lookup({ wallet }) {
  const [kind, setKind] = useState("text");
  const [text, setText] = useState("");
  const [url, setUrl] = useState("");
  const [file, setFile] = useState(null);
  const [state, setState] = useState(null);

  async function lookup() {
    setState({ loading: true });
    try {
      const hash = await sha256Hex(await contentBytes(kind, { text, url, file }));
      const c = readContract();
      if (!(await c.isRegistered(hash))) return setState({ hash, notFound: true });
      setState({ hash, media: await c.getMedia(hash) });
    } catch (err) {
      setState({ error: err.shortMessage || err.message });
    }
  }

  return (
    <section className="card">
      <h2>Lookup</h2>
      <p className="muted">
        Content is hashed in your browser, nothing is uploaded. Even a one-byte change gives a different hash. URLs
        are matched exactly as typed.
      </p>
      <MediaInput {...{ kind, setKind, text, setText, url, setUrl, file, setFile }} />
      <button disabled={!hasInput(kind, { text, url, file })} onClick={lookup}>Search registry</button>
      {state?.loading && <p className="info">Searching…</p>}
      {state?.error && <p className="error">{state.error}</p>}
      {state?.notFound && (
        <p className="info">Not registered yet (hash <span className="mono">{state.hash}</span>). Verify it in the first tab.</p>
      )}
      {state?.media && <MediaCard media={state.media} wallet={wallet} onChanged={lookup} />}
    </section>
  );
}
