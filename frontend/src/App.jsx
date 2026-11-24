import { useState } from "react";
import { connectWallet, errorMessage } from "./lib/chain.js";
import Verify from "./components/Verify.jsx";
import Lookup from "./components/Lookup.jsx";
import Registry from "./components/Registry.jsx";

const TABS = { verify: "Verify & Register", lookup: "Lookup", registry: "Registry" };

export default function App() {
  const [tab, setTab] = useState("verify");
  const [wallet, setWallet] = useState(null);
  const [error, setError] = useState("");

  async function connect() {
    setError("");
    try {
      setWallet(await connectWallet());
    } catch (err) {
      setError(errorMessage(err));
    }
  }

  return (
    <div className="app">
      <header>
        <div>
          <h1>TruthChain</h1>
          <p className="muted">Decentralized fake news & deepfake registry with AI verification</p>
        </div>
        {wallet ? (
          <span className="pill">{wallet.address.slice(0, 6)}…{wallet.address.slice(-4)}</span>
        ) : (
          <button onClick={connect}>Connect MetaMask</button>
        )}
      </header>
      {error && <p className="error">{error}</p>}

      <nav>
        {Object.entries(TABS).map(([key, label]) => (
          <button key={key} className={tab === key ? "tab active" : "tab"} onClick={() => setTab(key)}>
            {label}
          </button>
        ))}
      </nav>

      <main>
        {tab === "verify" && <Verify wallet={wallet} onConnect={connect} />}
        {tab === "lookup" && <Lookup wallet={wallet} />}
        {tab === "registry" && <Registry wallet={wallet} />}
      </main>
    </div>
  );
}
