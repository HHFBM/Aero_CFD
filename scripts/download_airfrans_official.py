#!/usr/bin/env python3
"""Download AirfRANS using the official library entrypoint."""

from __future__ import annotations

import argparse
import ssl
import zipfile
from pathlib import Path
from urllib.error import URLError

import airfrans as af


def _is_valid_zip(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with zipfile.ZipFile(path, "r") as zip_file:
            return zip_file.testzip() is None
    except zipfile.BadZipFile:
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download AirfRANS with the official airfrans API.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/data/airfrans_raw"),
        help="Directory where the dataset zip and extracted files will be stored.",
    )
    parser.add_argument(
        "--file-name",
        type=str,
        default="Dataset",
        help="Base filename used by airfrans.dataset.download().",
    )
    parser.add_argument(
        "--openfoam",
        action="store_true",
        help="Download the raw OpenFOAM archive instead of the processed AirfRANS dataset.",
    )
    parser.add_argument(
        "--no-unzip",
        action="store_true",
        help="Only download the archive without extracting it.",
    )
    parser.add_argument(
        "--force-clean",
        action="store_true",
        help="Delete an existing archive before download.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    archive_path = root / f"{args.file_name}.zip"
    if args.force_clean and archive_path.exists():
        archive_path.unlink()
        print(f"Removed existing archive: {archive_path}")
    elif archive_path.exists() and not _is_valid_zip(archive_path):
        archive_path.unlink()
        print(f"Removed incomplete archive before official download: {archive_path}")

    print(f"Downloading AirfRANS into: {root}")
    try:
        af.dataset.download(
            root=str(root),
            file_name=args.file_name,
            unzip=not args.no_unzip,
            OpenFOAM=args.openfoam,
        )
    except URLError as exc:
        reason = getattr(exc, "reason", None)
        if not isinstance(reason, ssl.SSLCertVerificationError):
            raise
        print("Official download hit local SSL certificate verification failure, retrying with insecure SSL context.")
        ssl._create_default_https_context = ssl._create_unverified_context  # type: ignore[attr-defined]
        af.dataset.download(
            root=str(root),
            file_name=args.file_name,
            unzip=not args.no_unzip,
            OpenFOAM=args.openfoam,
        )
    print("AirfRANS official download finished.")


if __name__ == "__main__":
    main()
