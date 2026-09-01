#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import tarfile
import zipfile
from pathlib import Path


def safe_target(root: Path, name: str) -> Path:
    target = (root / name).resolve()
    if os.path.commonpath([str(root.resolve()), str(target)]) != str(root.resolve()):
        raise RuntimeError(f"Unsafe archive member: {name}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Safely unpack a downloaded model archive.")
    parser.add_argument("--archive", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    archive = Path(args.archive).resolve()
    destination = Path(args.destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as handle:
            for member in handle.infolist():
                safe_target(destination, member.filename)
            handle.extractall(destination)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as handle:
            for member in handle.getmembers():
                safe_target(destination, member.name)
            handle.extractall(destination, filter="data")
    else:
        raise RuntimeError(f"Unsupported or incomplete model archive: {archive}")

    candidates = sorted(destination.rglob("model_index.json"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one Diffusers model_index.json after extraction, found {len(candidates)}"
        )
    print(candidates[0].parent)


if __name__ == "__main__":
    main()
