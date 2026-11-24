import { BrowserProvider, Contract, JsonRpcProvider } from "ethers";
import deployment from "../contracts/MediaRegistry.json";

export const RPC_URL = import.meta.env.VITE_RPC_URL || "http://127.0.0.1:8545";
export const GATEWAY_URL = import.meta.env.VITE_GATEWAY_URL || "http://127.0.0.1:8000";
export const MEDIA_TYPES = ["Text", "Image", "Video", "Audio", "URL"]; // MediaRegistry.MediaType
export const VERDICTS = ["Real", "Uncertain", "Fake"]; // MediaRegistry.Verdict

/** Input kinds offered in Verify and Lookup, and the gateway endpoint for each. */
export const KINDS = {
  text: { label: "News text", endpoint: "/analyze/text" },
  url: { label: "URL", endpoint: "/verify-url" },
  image: { label: "Image", endpoint: "/analyze/image", accept: "image/*", hint: "JPEG, PNG, WebP" },
  video: { label: "Video", endpoint: "/analyze/video", accept: "video/*", hint: "MP4, WebM, MOV - up to 100 MB" },
  audio: { label: "Audio", endpoint: "/analyze/audio", accept: "audio/*,.wav,.flac,.ogg,.mp3", hint: "WAV, FLAC, OGG, MP3" },
};

/** Bytes that identify an item on-chain: trimmed text / URL, or the raw file. */
export async function contentBytes(kind, { text, url, file }) {
  if (kind === "text") return textBytes(text);
  if (kind === "url") return textBytes(url);
  return new Uint8Array(await file.arrayBuffer());
}

/** Download a verification report from the gateway and check it against the on-chain hash. */
export async function fetchReport(reportHash) {
  const res = await fetch(`${GATEWAY_URL}/reports/${reportHash}`);
  if (!res.ok) throw new Error(res.status === 404 ? "This gateway does not hold the report." : `Gateway error ${res.status}`);
  const bytes = new Uint8Array(await res.arrayBuffer());
  const verified = (await sha256Hex(bytes)) === reportHash.toLowerCase();
  return { report: JSON.parse(new TextDecoder().decode(bytes)), verified };
}

const readProvider = new JsonRpcProvider(RPC_URL);

/** Read-only contract; works without a wallet. */
export function readContract() {
  return new Contract(deployment.address, deployment.abi, readProvider);
}

/** Connects MetaMask, switches to the deployment chain and returns a signer-backed contract. */
export async function connectWallet() {
  if (!window.ethereum) throw new Error("MetaMask not found. Install it to submit or vote.");
  const chainHex = "0x" + deployment.chainId.toString(16);
  try {
    await window.ethereum.request({ method: "wallet_switchEthereumChain", params: [{ chainId: chainHex }] });
  } catch (err) {
    if (err.code !== 4902) throw err;
    await window.ethereum.request({
      method: "wallet_addEthereumChain",
      params: [{
        chainId: chainHex,
        chainName: import.meta.env.VITE_CHAIN_NAME || "Hardhat Local",
        rpcUrls: [RPC_URL],
        nativeCurrency: { name: "Ether", symbol: "ETH", decimals: 18 },
      }],
    });
  }
  const provider = new BrowserProvider(window.ethereum);
  const signer = await provider.getSigner();
  return { address: await signer.getAddress(), contract: new Contract(deployment.address, deployment.abi, signer) };
}

/** Human-readable message for contract reverts and wallet errors. */
export function errorMessage(err) {
  const name = err?.revert?.name;
  const known = {
    AlreadyRegistered: "This content is already registered.",
    NotAVerifier: "Attestation signer is not an authorised AI gateway.",
    AttestationExpired: "The AI attestation expired — analyse again.",
    AlreadyVoted: "You have already voted on this item.",
    NotRegistered: "This content is not in the registry.",
  };
  if (name && known[name]) return known[name];
  if (err?.code === "ACTION_REJECTED") return "Transaction rejected in wallet.";
  return err?.shortMessage || err?.message || String(err);
}

export async function sha256Hex(bytes) {
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return "0x" + [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

export const textBytes = (text) => new TextEncoder().encode(text.trim());
