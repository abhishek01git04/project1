const fs = require("fs");
const path = require("path");
const hre = require("hardhat");

const { vars } = require("hardhat/config");

// Address of the AI gateway signing key. Locally defaults to Hardhat account #1,
// matching the default GATEWAY_PRIVATE_KEY in ai-gateway/.env.example. Public
// networks require `npx hardhat vars set GATEWAY_ADDRESS`.
const isLocal = ["hardhat", "localhost"].includes(hre.network.name);
const GATEWAY_ADDRESS =
  process.env.GATEWAY_ADDRESS ||
  (isLocal ? "0x70997970C51812dc3A010C7d01b50e0d17dc79C8" : vars.get("GATEWAY_ADDRESS"));

async function main() {
  const [deployer] = await hre.ethers.getSigners();
  console.log("Deploying with:", deployer.address);

  const Registry = await hre.ethers.getContractFactory("MediaRegistry");
  const registry = await Registry.deploy(GATEWAY_ADDRESS);
  await registry.waitForDeployment();

  const address = await registry.getAddress();
  const { chainId } = await hre.ethers.provider.getNetwork();
  console.log("MediaRegistry deployed at:", address);
  console.log("Authorised AI gateway:", GATEWAY_ADDRESS);

  const artifact = await hre.artifacts.readArtifact("MediaRegistry");
  const deployment = { address, chainId: Number(chainId), abi: artifact.abi };

  const targets = [
    path.join(__dirname, "../../frontend/src/contracts/MediaRegistry.json"),
    path.join(__dirname, "../../ai-gateway/deployment.json"),
  ];
  for (const file of targets) {
    fs.mkdirSync(path.dirname(file), { recursive: true });
    fs.writeFileSync(file, JSON.stringify(deployment, null, 2));
    console.log("Wrote", path.relative(path.join(__dirname, "../.."), file));
  }
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
