"""Signs verification results so the MediaRegistry contract can verify they came from this gateway."""

import time

from eth_account import Account
from eth_account.messages import encode_defunct
from web3 import Web3

from . import config

MEDIA_TYPES = {"text": 0, "image": 1, "video": 2, "audio": 3, "url": 4}  # MediaRegistry.MediaType

_account = Account.from_key(config.GATEWAY_PRIVATE_KEY)


def gateway_address() -> str:
    return _account.address


def sign_attestation(
    content_hash: str, media_type: str, ai_score: int, verdict_id: int, report_hash: str, model_version: str
) -> dict:
    """Mirror of MediaRegistry.attestationDigest followed by an EIP-191 signature."""
    deployment = config.load_deployment()
    issued_at = int(time.time())
    digest = Web3.solidity_keccak(
        ["uint256", "address", "bytes32", "uint8", "uint16", "uint8", "bytes32", "bytes32", "uint64"],
        [
            deployment["chainId"],
            Web3.to_checksum_address(deployment["address"]),
            bytes.fromhex(content_hash.removeprefix("0x")),
            MEDIA_TYPES[media_type],
            ai_score,
            verdict_id,
            bytes.fromhex(report_hash.removeprefix("0x")),
            Web3.keccak(text=model_version),
            issued_at,
        ],
    )
    signed = _account.sign_message(encode_defunct(primitive=digest))
    return {
        "contentHash": content_hash,
        "mediaType": MEDIA_TYPES[media_type],
        "aiScore": ai_score,
        "verdict": verdict_id,
        "reportHash": report_hash,
        "modelVersion": model_version,
        "issuedAt": issued_at,
        "signature": "0x" + signed.signature.hex().removeprefix("0x"),
        "signer": _account.address,
        "contract": deployment["address"],
        "chainId": deployment["chainId"],
    }
