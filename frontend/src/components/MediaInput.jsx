import { useEffect, useState } from "react";
import { KINDS } from "../lib/chain.js";

/** Text area, URL field or file picker (image / video / audio), shared by Verify and Lookup. */
export default function MediaInput({ kind, setKind, text, setText, url, setUrl, file, setFile }) {
  const [preview, setPreview] = useState("");

  useEffect(() => {
    if (!file) return setPreview("");
    const objectUrl = URL.createObjectURL(file);
    setPreview(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [file]);

  function choose(k) {
    setKind(k);
    setFile(null);
  }

  const k = KINDS[kind];
  return (
    <>
      <div className="segmented" role="tablist">
        {Object.entries(KINDS).map(([key, { label }]) => (
          <button key={key} className={kind === key ? "active" : ""} onClick={() => choose(key)}>
            {label}
          </button>
        ))}
      </div>
      {kind === "text" && (
        <textarea
          rows={7}
          placeholder="Paste a news article, headline or social media post…"
          value={text}
          onChange={(e) => setText(e.target.value)}
        />
      )}
      {kind === "url" && (
        <input
          type="url"
          placeholder="https://news-site.com/article… (or a direct link to an image, video or audio file)"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
        />
      )}
      {k.accept && (
        <label className="dropzone">
          <input key={kind} type="file" accept={k.accept} onChange={(e) => setFile(e.target.files[0] || null)} />
          {!file && <span className="muted">Click to choose {kind === "audio" ? "an" : "a"} {kind} file ({k.hint})</span>}
          {file && kind === "image" && <img src={preview} alt="Selected image preview" />}
          {file && kind === "video" && <video src={preview} controls muted />}
          {file && kind === "audio" && <audio src={preview} controls />}
          {file && <span className="muted">{file.name} · {(file.size / 1024 / 1024).toFixed(1)} MB</span>}
        </label>
      )}
    </>
  );
}

/** True when the selected kind has its input filled in. */
export function hasInput(kind, { text, url, file }) {
  if (kind === "text") return Boolean(text.trim());
  if (kind === "url") return /^https?:\/\/\S+\.\S+/.test(url.trim());
  return Boolean(file);
}
