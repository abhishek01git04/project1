// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {Ownable} from "@openzeppelin/contracts/access/Ownable.sol";
import {ECDSA} from "@openzeppelin/contracts/utils/cryptography/ECDSA.sol";
import {MessageHashUtils} from "@openzeppelin/contracts/utils/cryptography/MessageHashUtils.sol";

/// @title MediaRegistry
/// @notice Decentralized registry of news, images, videos, audio and URLs together with
///         the AI gateway's verdict, a hash of its verification report, and community votes.
/// @dev    Media are identified by the SHA-256 hash of their raw content (for URLs: of the
///         URL string). A record can only be written with an attestation signed by an
///         authorised AI verification gateway, so a submitter cannot fake the verdict.
contract MediaRegistry is Ownable {
    using ECDSA for bytes32;
    using MessageHashUtils for bytes32;

    enum MediaType { Text, Image, Video, Audio, Url }
    enum Verdict { Real, Uncertain, Fake }

    /// What the gateway signs. Mirrors ai-gateway/app/signer.py.
    struct Attestation {
        bytes32 contentHash;  // SHA-256 of the media
        MediaType mediaType;
        uint16 aiScore;       // fused fake probability, basis points (0..10000)
        Verdict verdict;      // fused verdict: Real / Uncertain / Fake
        bytes32 reportHash;   // SHA-256 of the JSON verification report
        string modelVersion;  // model(s) and fusion version used
        uint64 issuedAt;      // gateway clock, seconds
    }

    struct Media {
        bytes32 contentHash;
        MediaType mediaType;
        Verdict verdict;
        uint16 aiScore;
        bytes32 reportHash;
        string modelVersion;
        address issuer;       // gateway that signed the attestation
        uint64 timestamp;     // block time of registration
        address submitter;
        string title;
        string sourceUrl;
        uint32 votesReal;
        uint32 votesFake;
    }

    /// Attestations older than this are rejected (replay / staleness protection).
    uint64 public constant ATTESTATION_TTL = 1 hours;
    /// Minimum number of community votes before they influence the final verdict.
    uint32 public constant MIN_VOTES = 3;

    mapping(address => bool) public isVerifier;
    mapping(bytes32 => Media) private _media;
    mapping(bytes32 => mapping(address => bool)) public hasVoted;
    bytes32[] private _hashes;

    event VerifierUpdated(address indexed verifier, bool allowed);
    event MediaRegistered(
        bytes32 indexed contentHash,
        address indexed submitter,
        MediaType mediaType,
        Verdict verdict,
        uint16 aiScore,
        bytes32 reportHash
    );
    event Voted(bytes32 indexed contentHash, address indexed voter, bool isFake);

    error NotAVerifier(address signer);
    error AlreadyRegistered(bytes32 contentHash);
    error NotRegistered(bytes32 contentHash);
    error AttestationExpired();
    error InvalidScore();
    error AlreadyVoted();

    constructor(address initialVerifier) Ownable(msg.sender) {
        if (initialVerifier != address(0)) {
            isVerifier[initialVerifier] = true;
            emit VerifierUpdated(initialVerifier, true);
        }
    }

    // ------------------------------------------------------------------ admin

    function setVerifier(address verifier, bool allowed) external onlyOwner {
        isVerifier[verifier] = allowed;
        emit VerifierUpdated(verifier, allowed);
    }

    // --------------------------------------------------------------- writing

    /// @notice Digest the gateway must sign. Bound to this chain and contract.
    function attestationDigest(Attestation calldata a) public view returns (bytes32) {
        return keccak256(
            abi.encodePacked(
                block.chainid,
                address(this),
                a.contentHash,
                uint8(a.mediaType),
                a.aiScore,
                uint8(a.verdict),
                a.reportHash,
                keccak256(bytes(a.modelVersion)),
                a.issuedAt
            )
        );
    }

    /// @notice Record a verified item. The caller pays gas; the gateway's signature
    ///         proves the verdict, score and report hash were not altered.
    function registerMedia(
        Attestation calldata a,
        string calldata title,
        string calldata sourceUrl,
        bytes calldata signature
    ) external {
        if (_media[a.contentHash].timestamp != 0) revert AlreadyRegistered(a.contentHash);
        if (a.aiScore > 10000) revert InvalidScore();
        if (a.issuedAt + ATTESTATION_TTL < block.timestamp) revert AttestationExpired();

        address signer = attestationDigest(a).toEthSignedMessageHash().recover(signature);
        if (!isVerifier[signer]) revert NotAVerifier(signer);

        Media storage m = _media[a.contentHash];
        m.contentHash = a.contentHash;
        m.mediaType = a.mediaType;
        m.verdict = a.verdict;
        m.aiScore = a.aiScore;
        m.reportHash = a.reportHash;
        m.modelVersion = a.modelVersion;
        m.issuer = signer;
        m.timestamp = uint64(block.timestamp);
        m.submitter = msg.sender;
        m.title = title;
        m.sourceUrl = sourceUrl;
        _hashes.push(a.contentHash);

        emit MediaRegistered(a.contentHash, msg.sender, a.mediaType, a.verdict, a.aiScore, a.reportHash);
    }

    /// @notice Community vote on whether a registered item is fake. One vote per address.
    function vote(bytes32 contentHash, bool isFake) external {
        Media storage m = _media[contentHash];
        if (m.timestamp == 0) revert NotRegistered(contentHash);
        if (hasVoted[contentHash][msg.sender]) revert AlreadyVoted();
        hasVoted[contentHash][msg.sender] = true;
        if (isFake) m.votesFake++;
        else m.votesReal++;
        emit Voted(contentHash, msg.sender, isFake);
    }

    // --------------------------------------------------------------- reading

    function isRegistered(bytes32 contentHash) external view returns (bool) {
        return _media[contentHash].timestamp != 0;
    }

    function getMedia(bytes32 contentHash) external view returns (Media memory) {
        if (_media[contentHash].timestamp == 0) revert NotRegistered(contentHash);
        return _media[contentHash];
    }

    function totalMedia() external view returns (uint256) {
        return _hashes.length;
    }

    /// @notice Paginated listing, newest first.
    function listMedia(uint256 offset, uint256 limit) external view returns (Media[] memory page) {
        uint256 n = _hashes.length;
        if (offset >= n) return new Media[](0);
        uint256 count = n - offset < limit ? n - offset : limit;
        page = new Media[](count);
        for (uint256 i = 0; i < count; i++) {
            page[i] = _media[_hashes[n - 1 - offset - i]];
        }
    }

    /// @notice Final score combining the AI (60%) and the community (40%) once
    ///         enough votes exist. Returns the fake probability in basis points.
    function finalScore(bytes32 contentHash) public view returns (uint16) {
        Media storage m = _media[contentHash];
        if (m.timestamp == 0) revert NotRegistered(contentHash);
        uint256 total = uint256(m.votesReal) + m.votesFake;
        if (total < MIN_VOTES) return m.aiScore;
        uint256 community = (uint256(m.votesFake) * 10000) / total;
        return uint16((uint256(m.aiScore) * 60 + community * 40) / 100);
    }

    /// @notice The gateway's verdict until MIN_VOTES votes exist, then the verdict of the blended score.
    function finalVerdict(bytes32 contentHash) external view returns (Verdict) {
        Media storage m = _media[contentHash];
        if (m.timestamp == 0) revert NotRegistered(contentHash);
        if (uint256(m.votesReal) + m.votesFake < MIN_VOTES) return m.verdict;
        return scoreToVerdict(finalScore(contentHash));
    }

    function scoreToVerdict(uint16 score) public pure returns (Verdict) {
        if (score >= 7000) return Verdict.Fake;
        if (score > 3000) return Verdict.Uncertain;
        return Verdict.Real;
    }
}
