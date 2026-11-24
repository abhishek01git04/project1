// Generates a fresh Ethereum key pair, e.g. for the AI gateway on a public network.
// Usage: node scripts/new-key.js
const { Wallet } = require("ethers");

const w = Wallet.createRandom();
console.log("Address:    ", w.address);
console.log("Private key:", w.privateKey);
