require("@nomicfoundation/hardhat-toolbox");
const { vars } = require("hardhat/config");

// Set once with `npx hardhat vars set SEPOLIA_RPC_URL` / `npx hardhat vars set DEPLOYER_KEY`.
const SEPOLIA_RPC_URL = vars.get("SEPOLIA_RPC_URL", process.env.SEPOLIA_RPC_URL || "");
const DEPLOYER_KEY = vars.get("DEPLOYER_KEY", process.env.DEPLOYER_KEY || "");

/** @type import('hardhat/config').HardhatUserConfig */
module.exports = {
  solidity: {
    version: "0.8.28",
    settings: { optimizer: { enabled: true, runs: 200 }, viaIR: true, evmVersion: "cancun" },
  },
  networks: {
    localhost: { url: "http://127.0.0.1:8545", chainId: 31337 },
    ...(SEPOLIA_RPC_URL && DEPLOYER_KEY
      ? { sepolia: { url: SEPOLIA_RPC_URL, accounts: [DEPLOYER_KEY], chainId: 11155111 } }
      : {}),
  },
};
