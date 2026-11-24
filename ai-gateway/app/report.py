"""Verification reports: canonical JSON, its SHA-256 (the on-chain reportHash) and storage.

The exact bytes that were hashed are stored, so anyone can download a report from
/reports/{hash} and check it against the hash recorded on the blockchain.
"""

import hashlib
import json
import re
from datetime import datetime, timezone

from . import config

SCHEMA = "truthchain.report/v1"
_HASH = re.compile(r"^0x[0-9a-f]{64}$")


def build(**fields) -> dict:
    return {"schema": SCHEMA, "createdAt": datetime.now(timezone.utc).isoformat(timespec="seconds"), **fields}


def encode(report: dict) -> tuple[bytes, str]:
    data = json.dumps(report, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return data, "0x" + hashlib.sha256(data).hexdigest()


def save(report: dict) -> str:
    data, report_hash = encode(report)
    try:
        config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        (config.REPORTS_DIR / f"{report_hash}.json").write_bytes(data)
    except OSError:
        pass  # read-only hosting: the hash is still valid, the report is returned inline
    return report_hash


def load(report_hash: str) -> bytes | None:
    report_hash = report_hash.lower()
    if not _HASH.match(report_hash):
        return None
    path = config.REPORTS_DIR / f"{report_hash}.json"
    return path.read_bytes() if path.exists() else None
