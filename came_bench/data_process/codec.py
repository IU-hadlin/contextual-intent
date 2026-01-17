#!/usr/bin/env python3
"""
CAME-Bench encoder/decoder (Option A):
- For each file, generate a per-file canary (UUID).
- Derive a keystream from SHA-256(canary) repeated to file length.
- XOR plaintext bytes with keystream -> ciphertext.
- Base64-encode ciphertext and store as .b64 text.

This is obfuscation to avoid publishing plaintext, not cryptographic security.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import pathlib
import sys
import uuid
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional


# ----------------------------
# Core primitives
# ----------------------------

def sha256_bytes(s: str) -> bytes:
    return hashlib.sha256(s.encode("utf-8")).digest()

def derive_keystream(canary: str, n: int) -> bytes:
    """
    Repeat SHA256(canary) bytes until length n.
    """
    seed = sha256_bytes(canary)  # 32 bytes
    if n <= 0:
        return b""
    reps = (n + len(seed) - 1) // len(seed)
    return (seed * reps)[:n]

def xor_bytes(data: bytes, key: bytes) -> bytes:
    if len(data) != len(key):
        raise ValueError("xor_bytes: data and key must have same length")
    return bytes(d ^ k for d, k in zip(data, key))

def encrypt_bytes(plaintext: bytes, canary: str) -> bytes:
    key = derive_keystream(canary, len(plaintext))
    return xor_bytes(plaintext, key)

def decrypt_bytes(ciphertext: bytes, canary: str) -> bytes:
    # XOR is symmetric
    return encrypt_bytes(ciphertext, canary)


# ----------------------------
# Metadata schema
# ----------------------------

@dataclass
class FileRecord:
    relpath: str                 # original relative path
    blob: str                    # path under encoded/ (e.g. blobs/<id>.b64)
    canary: str                  # per-file UUID
    nbytes: int                  # original file size
    sha256: str                  # sha256 hex of plaintext (integrity check)


@dataclass
class Manifest:
    version: str
    created_by: str
    files: List[FileRecord]


# ----------------------------
# Helpers
# ----------------------------

TEXT_EXTS = {".txt", ".json", ".jsonl", ".md", ".csv", ".tsv", ".yaml", ".yml"}

def is_hidden_path(p: pathlib.Path) -> bool:
    return any(part.startswith(".") for part in p.parts)

def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def stable_blob_name(relpath: str) -> str:
    """
    Deterministic blob filename from relpath (avoids leaking the relpath in blob name).
    """
    h = hashlib.sha256(relpath.encode("utf-8")).hexdigest()
    return f"{h}.b64"

def ensure_parent(path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Encode / Decode
# ----------------------------

def encode_dir(raw_dir: pathlib.Path, out_dir: pathlib.Path) -> None:
    raw_dir = raw_dir.resolve()
    out_dir = out_dir.resolve()

    blobs_dir = out_dir / "blobs"
    blobs_dir.mkdir(parents=True, exist_ok=True)

    records: List[FileRecord] = []

    for fp in raw_dir.rglob("*"):
        if fp.is_dir():
            continue
        if is_hidden_path(fp.relative_to(raw_dir)):
            continue

        relpath = fp.relative_to(raw_dir).as_posix()
        data = fp.read_bytes()

        canary = str(uuid.uuid4())
        ct = encrypt_bytes(data, canary)
        b64 = base64.b64encode(ct).decode("ascii")

        blob_name = stable_blob_name(relpath)
        blob_path = blobs_dir / blob_name
        ensure_parent(blob_path)
        blob_path.write_text(b64, encoding="utf-8")

        rec = FileRecord(
            relpath=relpath,
            blob=f"blobs/{blob_name}",
            canary=canary,
            nbytes=len(data),
            sha256=sha256_hex(data),
        )
        records.append(rec)

    manifest = Manifest(
        version="1.0",
        created_by="came-bench codec.py (base64+xor+sha256(canary) keystream)",
        files=records,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metadata.json").write_text(
        json.dumps(asdict(manifest), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[encode] raw_dir={raw_dir}")
    print(f"[encode] out_dir={out_dir}")
    print(f"[encode] files={len(records)}")
    print(f"[encode] wrote {out_dir/'metadata.json'} and {blobs_dir}/")


def decode_dir(encoded_dir: pathlib.Path, out_dir: pathlib.Path, strict: bool = True) -> None:
    encoded_dir = encoded_dir.resolve()
    out_dir = out_dir.resolve()

    manifest_path = encoded_dir / "metadata.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files", [])

    n_ok = 0
    for rec in files:
        relpath = rec["relpath"]
        blob_rel = rec["blob"]
        canary = rec["canary"]
        expected_nbytes = int(rec["nbytes"])
        expected_sha = rec["sha256"]

        blob_path = encoded_dir / blob_rel
        b64 = blob_path.read_text(encoding="utf-8").strip()
        ct = base64.b64decode(b64.encode("ascii"))

        pt = decrypt_bytes(ct, canary)

        if strict:
            if len(pt) != expected_nbytes:
                raise ValueError(f"size mismatch for {relpath}: got {len(pt)} expected {expected_nbytes}")
            got_sha = sha256_hex(pt)
            if got_sha != expected_sha:
                raise ValueError(f"sha256 mismatch for {relpath}: got {got_sha} expected {expected_sha}")

        out_path = out_dir / relpath
        ensure_parent(out_path)
        out_path.write_bytes(pt)
        n_ok += 1

    print(f"[decode] encoded_dir={encoded_dir}")
    print(f"[decode] out_dir={out_dir}")
    print(f"[decode] files={n_ok}")
    if strict:
        print("[decode] strict integrity checks: ON")


# ----------------------------
# CLI
# ----------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_enc = sub.add_parser("encode", help="encode raw_dir -> out_dir (encoded)")
    ap_enc.add_argument("--raw_dir", required=True, type=str)
    ap_enc.add_argument("--out_dir", required=True, type=str)

    ap_dec = sub.add_parser("decode", help="decode encoded_dir -> out_dir (reconstructed)")
    ap_dec.add_argument("--encoded_dir", required=True, type=str)
    ap_dec.add_argument("--out_dir", required=True, type=str)
    ap_dec.add_argument("--no_strict", action="store_true", help="disable sha/size checks")

    args = ap.parse_args(argv)

    if args.cmd == "encode":
        encode_dir(pathlib.Path(args.raw_dir), pathlib.Path(args.out_dir))
        return 0
    if args.cmd == "decode":
        decode_dir(pathlib.Path(args.encoded_dir), pathlib.Path(args.out_dir), strict=(not args.no_strict))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
