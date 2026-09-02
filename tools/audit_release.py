#!/usr/bin/env python3
"""Fail if a release tree contains datasets, weights, outputs or likely secrets."""

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_DIRS = {
    "data",
    "runs",
    "checkpoints",
}
IGNORED_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
}
FORBIDDEN_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".npy",
    ".npz",
    ".o",
    ".pt",
    ".pth",
    ".so",
}
SECRET_PATTERNS = {
    "GitHub token": re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"),
    "AWS access key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}
MAX_FILE_BYTES = 5 * 1024 * 1024


def main() -> int:
    findings = []
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if any(part in IGNORED_DIRS for part in relative.parts):
            continue
        if relative.parts and relative.parts[0] in FORBIDDEN_DIRS:
            findings.append(f"forbidden directory: {relative}")
            continue
        if path.is_symlink():
            findings.append(f"symlink not allowed: {relative}")
            continue
        if not path.is_file():
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            findings.append(f"forbidden artifact: {relative}")
        if path.stat().st_size > MAX_FILE_BYTES:
            findings.append(f"file exceeds 5 MiB: {relative}")
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(content):
                findings.append(f"possible {label}: {relative}")

    if findings:
        print("Release audit FAILED:")
        for finding in sorted(set(findings)):
            print(" -", finding)
        return 1
    print("Release audit passed: no data, weights, binaries or known secret patterns.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
