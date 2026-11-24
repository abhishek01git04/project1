const { expect } = require("chai");
const { ethers } = require("hardhat");
const { time } = require("@nomicfoundation/hardhat-network-helpers");

const REAL = 0, UNCERTAIN = 1, FAKE = 2;

describe("MediaRegistry", function () {
  let registry, owner, gateway, user, voters;
  const contentHash = ethers.sha256(ethers.toUtf8Bytes("Breaking: aliens land in Paris"));
  const reportHash = ethers.sha256(ethers.toUtf8Bytes('{"verdict":"Fake"}'));

  async function attest(signer, overrides = {}) {
    const a = {
      contentHash,
      mediaType: 0,
      aiScore: 8500,
      verdict: FAKE,
      reportHash,
      modelVersion: "text:roberta-fake-news|fusion-v1",
      issuedAt: await time.latest(),
      ...overrides,
    };
    const digest = await registry.attestationDigest(a);
    return { a, signature: await signer.signMessage(ethers.getBytes(digest)) };
  }

  function register(from, { a, signature }) {
    return registry.connect(from).registerMedia(a, "Aliens", "https://example.com", signature);
  }

  beforeEach(async function () {
    [owner, gateway, user, ...voters] = await ethers.getSigners();
    registry = await (await ethers.getContractFactory("MediaRegistry")).deploy(gateway.address);
  });

  it("registers media with a valid gateway attestation", async function () {
    await expect(register(user, await attest(gateway)))
      .to.emit(registry, "MediaRegistered")
      .withArgs(contentHash, user.address, 0, FAKE, 8500, reportHash);

    const m = await registry.getMedia(contentHash);
    expect(m.submitter).to.equal(user.address);
    expect(m.issuer).to.equal(gateway.address);
    expect(m.aiScore).to.equal(8500);
    expect(m.verdict).to.equal(FAKE);
    expect(m.reportHash).to.equal(reportHash);
    expect(m.timestamp).to.be.greaterThan(0);
    expect(await registry.totalMedia()).to.equal(1);
  });

  it("accepts every media type, including URLs", async function () {
    for (const mediaType of [1, 2, 3, 4]) {
      const hash = ethers.sha256(ethers.toUtf8Bytes(`item ${mediaType}`));
      await register(user, await attest(gateway, { contentHash: hash, mediaType, verdict: UNCERTAIN }));
      expect((await registry.getMedia(hash)).mediaType).to.equal(mediaType);
    }
  });

  it("rejects attestations not signed by a verifier", async function () {
    await expect(register(user, await attest(user))).to.be.revertedWithCustomError(registry, "NotAVerifier");
  });

  it("rejects tampered scores, verdicts and report hashes", async function () {
    for (const change of [{ aiScore: 100 }, { verdict: REAL }, { reportHash: ethers.ZeroHash }]) {
      const signed = await attest(gateway);
      signed.a = { ...signed.a, ...change }; // submitter tries to whitewash a fake
      await expect(register(user, signed)).to.be.revertedWithCustomError(registry, "NotAVerifier");
    }
  });

  it("rejects expired attestations and duplicates", async function () {
    const old = await attest(gateway, { issuedAt: (await time.latest()) - 7200 });
    await expect(register(user, old)).to.be.revertedWithCustomError(registry, "AttestationExpired");

    await register(user, await attest(gateway));
    await expect(register(user, await attest(gateway))).to.be.revertedWithCustomError(registry, "AlreadyRegistered");
  });

  it("keeps the gateway verdict until enough votes, then blends in the community", async function () {
    await register(user, await attest(gateway, { aiScore: 5000, verdict: UNCERTAIN }));
    expect(await registry.finalVerdict(contentHash)).to.equal(UNCERTAIN);

    for (const v of voters.slice(0, 4)) await registry.connect(v).vote(contentHash, true);
    // 0.6 * 5000 + 0.4 * 10000 = 7000 -> Fake
    expect(await registry.finalScore(contentHash)).to.equal(7000);
    expect(await registry.finalVerdict(contentHash)).to.equal(FAKE);

    await expect(registry.connect(voters[0]).vote(contentHash, false)).to.be.revertedWithCustomError(
      registry,
      "AlreadyVoted"
    );
  });

  it("only the owner manages verifiers", async function () {
    await expect(registry.connect(user).setVerifier(user.address, true)).to.be.revertedWithCustomError(
      registry,
      "OwnableUnauthorizedAccount"
    );
    await registry.setVerifier(gateway.address, false);
    await expect(register(user, await attest(gateway))).to.be.revertedWithCustomError(registry, "NotAVerifier");
  });

  it("lists media newest first", async function () {
    const h2 = ethers.sha256(ethers.toUtf8Bytes("second"));
    await register(user, await attest(gateway));
    await register(user, await attest(gateway, { contentHash: h2 }));
    const page = await registry.listMedia(0, 10);
    expect(page.map((m) => m.contentHash)).to.deep.equal([h2, contentHash]);
  });
});
