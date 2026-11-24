// Smoke test: gateway analysis + evidence + report -> on-chain registration -> lookup.
// Usage: npx hardhat run scripts/e2e-check.js --network localhost
// Set GATEWAY_URL if the gateway is not on http://127.0.0.1:8000.
const hre = require("hardhat");
const deployment = require("../../frontend/src/contracts/MediaRegistry.json");

const GATEWAY_URL = process.env.GATEWAY_URL || "http://127.0.0.1:8000";
const VERDICTS = ["Real", "Uncertain", "Fake"];

async function main() {
  const [, , user] = await hre.ethers.getSigners();
  const registry = new hre.ethers.Contract(deployment.address, deployment.abi, user);
  const text = `SHOCKING!!! Doctors HATE this miracle cure they don't want you to know! ${Date.now()}`;
  const res = await fetch(`${GATEWAY_URL}/analyze/text`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ text }),
  }).then((r) => r.json());
  console.log("Gateway:", res.verdict, res.aiScore, `confidence ${res.confidence}`);
  console.log("Fusion:", res.fusion);

  const localHash = hre.ethers.sha256(hre.ethers.toUtf8Bytes(text));
  if (localHash !== res.contentHash) throw new Error("content hash mismatch");

  // The report served by the gateway must hash to the reportHash that gets signed.
  const reportBytes = new Uint8Array(await (await fetch(`${GATEWAY_URL}/reports/${res.reportHash}`)).arrayBuffer());
  if (hre.ethers.sha256(reportBytes) !== res.reportHash) throw new Error("report hash mismatch");

  const { signature, ...a } = res.attestation;
  await (await registry.registerMedia(a, "Miracle cure", "", signature)).wait();
  const m = await registry.getMedia(res.contentHash);
  console.log("On-chain:", {
    verdict: VERDICTS[Number(m.verdict)],
    score: Number(m.aiScore),
    reportHash: m.reportHash,
    modelVersion: m.modelVersion,
    issuer: m.issuer,
    finalVerdict: VERDICTS[Number(await registry.finalVerdict(res.contentHash))],
  });
  console.log("E2E OK");
}
main().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
