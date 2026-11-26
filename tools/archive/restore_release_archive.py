"""Restore or verify raw TAR volumes from the EMAformer reproduction draft."""

import argparse
from contextlib import contextmanager, nullcontext
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from urllib.parse import urlsplit

import requests

from private_release_sync import (
    ALLOWED_GITHUB_PROXY, BackupError, EXPECTED_MANIFEST, EXPECTED_TAG, GitHub,
    MERGED_REPOSITORY, MERGED_REPOSITORY_ID, asset_matches, credential_token,
)
from release_split_local import BUFFER, MAX_PART_BYTES, SCHEMA, disk_path, no_links

RELEASE_ID = 459344454
MANIFEST_ASSET = f"export_manifest_{EXPECTED_MANIFEST[:16]}.json"
EXPECTED_VOLUMES = [f"main288_{number:03d}.tar" for number in range(1, 26)]
ASSET_HOSTS = {
    "release-assets.githubusercontent.com", "objects.githubusercontent.com",
    "github-releases.githubusercontent.com",
}


@contextmanager
def asset_stream(github, asset_id):
    request = f"{github.base}/releases/assets/{asset_id}"
    response = github.request(
        "GET", request, accept="application/octet-stream",
        allow_asset_redirect=True, stream=True,
    )
    try:
        if response.status_code in (301, 302, 303, 307, 308):
            location = response.headers.get("Location", "")
            target = urlsplit(location)
            if (target.scheme != "https" or target.hostname not in ASSET_HOSTS
                    or target.username or target.password or target.port not in (None, 443)):
                raise BackupError("Unexpected GitHub asset download host")
            response.close()
            # This signed asset URL needs no Authorization header.
            response = github.session.get(
                location, stream=True, allow_redirects=False, timeout=(30, 180),
            )
        if response.status_code != 200:
            raise BackupError(f"Asset download failed with HTTP {response.status_code}")
        yield response
    except requests.RequestException as error:
        raise BackupError(f"Asset transport failed: {type(error).__name__}") from None
    finally:
        response.close()


def asset_blocks(github, asset, count, digest):
    checksum = hashlib.sha256()
    bytes_read = 0
    with asset_stream(github, asset["id"]) as response:
        for block in response.iter_content(chunk_size=BUFFER):
            if block:
                bytes_read += len(block)
                if bytes_read > count:
                    raise BackupError(f"Asset is larger than expected: {asset['name']}")
                checksum.update(block)
                yield block
    if bytes_read != count or checksum.hexdigest() != digest:
        raise BackupError(f"Downloaded bytes do not match SHA-256 and size: {asset['name']}")


def json_asset(github, assets, name, digest):
    asset = assets.get(name)
    if (asset is None or type(asset.get("size")) is not int or not 0 < asset["size"] <= 8_000_000
            or type(asset.get("id")) is not int or asset["id"] <= 0
            or not asset_matches(asset, asset["size"], digest)):
        raise BackupError(f"Invalid remote metadata asset: {name}")
    return json.loads(b"".join(asset_blocks(github, asset, asset["size"], digest)))


def indexed_parts(manifest, index, assets, selected):
    if manifest["schema"] != "emaformer-export-v1":
        raise BackupError("Unexpected export manifest schema")
    volumes = {volume["name"]: volume for volume in manifest["volumes"]}
    if list(volumes) != EXPECTED_VOLUMES or len(manifest["volumes"]) != len(volumes):
        raise BackupError("Unexpected export volume inventory")
    if (index["schema"] != "emaformer-release-backup-index-v1"
            or index["repository"] != MERGED_REPOSITORY
            or index["repository_id"] != MERGED_REPOSITORY_ID
            or index["release_id"] != RELEASE_ID
            or index["manifest_sha256"] != EXPECTED_MANIFEST
            or index["volume_order"] != EXPECTED_VOLUMES
            or index["release_remains_draft"] is not True):
        raise BackupError("Backup index does not identify the retained draft")

    split = index["local_split_index"]
    part_limit = index["maximum_part_bytes"]
    if (type(part_limit) is not int or not 0 < part_limit <= MAX_PART_BYTES
            or split["maximum_part_bytes"] != part_limit
            or split["schema"] != SCHEMA
            or split["manifest_sha256"] != EXPECTED_MANIFEST):
        raise BackupError("Backup index uses an unexpected part format")
    split_volumes = {row["name"]: row for row in split["volumes"]}
    receipts = {row["name"]: row for row in index["remote_parts"]}
    verified = index["complete_uploaded_volumes"]
    if (len(verified) != len(set(verified)) or set(verified) - volumes.keys()
            or list(split_volumes) != EXPECTED_VOLUMES
            or len(split["volumes"]) != len(split_volumes)
            or len(index["remote_parts"]) != len(receipts)):
        raise BackupError("Invalid backup index inventory")
    if not selected or len(selected) != len(set(selected)) or not set(selected) <= set(verified):
        raise BackupError("Every selected volume must be indexed as remotely verified")

    plans = {}
    for name in selected:
        expected = volumes[name]
        row = split_volumes[name]
        if (row["status"] != "complete" or row["bytes"] != expected["bytes"]
                or row["sha256"] != expected["sha256"]
                or row["source_whole_hash_verified"] is not True
                or row["concatenated_raw_hash_verified"] is not True):
            raise BackupError(f"Invalid volume identity: {name}")
        parts = row["parts"]
        offsets = range(0, expected["bytes"], part_limit)
        if len(parts) != len(offsets):
            raise BackupError(f"Incomplete split inventory: {name}")
        plan = []
        for number, (offset, part) in enumerate(zip(offsets, parts), start=1):
            part_name = f"{name}.part{number:03d}"
            count = min(part_limit, expected["bytes"] - offset)
            digest = part["sha256"]
            if (part["number"] != number or part["name"] != part_name
                    or part["offset"] != offset or part["bytes"] != count
                    or not re.fullmatch(r"[0-9a-f]{64}", digest)):
                raise BackupError(f"Invalid indexed split boundary: {part_name}")
            asset_name = part_name if part_limit == MAX_PART_BYTES else f"raw{part_limit}-{part_name}"
            receipt = receipts.get(asset_name)
            asset = assets.get(asset_name)
            if (receipt is None or asset is None or receipt["remote_digest_verified"] is not True
                    or type(asset.get("id")) is not int or asset["id"] <= 0
                    or receipt["asset_id"] != asset["id"] or receipt["bytes"] != count
                    or receipt["sha256"] != digest or not asset_matches(asset, count, digest)):
                raise BackupError(f"Live GitHub asset does not match its receipt: {asset_name}")
            plan.append((asset, count, digest))
        plans[name] = (expected, plan)
    return plans


@contextmanager
def volume_output(destination, name):
    target = destination / name
    temporary = target.with_suffix(".tar.restore-partial")
    no_links(target)
    no_links(temporary)
    if target.exists():
        raise FileExistsError(target)
    with open(disk_path(temporary), "xb") as handle:
        yield handle
        handle.flush()
        os.fsync(handle.fileno())
    # Both names are in the same directory. Link/create avoids ever replacing
    # an existing target, including a file created during the download.
    if os.name == "nt":
        os.rename(disk_path(temporary), disk_path(target))
    else:
        os.link(disk_path(temporary), disk_path(target))
        os.unlink(disk_path(temporary))


def restore(github, index_name, volumes, destination=None):
    filename = re.fullmatch(r"backup_index_([0-9a-f]{16})\.json", index_name)
    if not filename:
        raise BackupError("Specify an immutable backup_index_<sha256-prefix>.json asset")
    github.verify_identity()
    assets = github.assets()
    index_asset = assets.get(index_name, {})
    digest = index_asset.get("digest") or ""
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest) or digest[7:23] != filename[1]:
        raise BackupError("Backup index digest does not match its immutable filename")
    index = json_asset(github, assets, index_name, digest[7:])
    manifest = json_asset(github, assets, MANIFEST_ASSET, EXPECTED_MANIFEST)
    plans = indexed_parts(manifest, index, assets, volumes)
    if destination is not None:
        no_links(destination)
        destination.mkdir(parents=True, exist_ok=True)
    for name, (expected, parts) in plans.items():
        whole = hashlib.sha256()
        count = 0
        with volume_output(destination, name) if destination is not None else nullcontext() as output:
            for asset, length, digest in parts:
                for block in asset_blocks(github, asset, length, digest):
                    whole.update(block)
                    count += len(block)
                    if output is not None:
                        output.write(block)
            if count != expected["bytes"] or whole.hexdigest() != expected["sha256"]:
                raise BackupError(f"Reassembled volume does not match the export manifest: {name}")
        yield {
            "repository": MERGED_REPOSITORY, "release_id": RELEASE_ID, "index": index_name,
            "volume": name, "bytes": count, "sha256": whole.hexdigest(),
            "readback_verified": True, "restored_to_disk": destination is not None,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-asset", required=True)
    parser.add_argument("--volume", action="append", choices=EXPECTED_VOLUMES, required=True)
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--verify-only", action="store_true")
    output.add_argument("--destination", type=Path)
    parser.add_argument("--github-proxy", choices=[ALLOWED_GITHUB_PROXY])
    args = parser.parse_args()
    try:
        github = GitHub(
            credential_token(MERGED_REPOSITORY), EXPECTED_TAG, repository=MERGED_REPOSITORY,
            release_id=RELEASE_ID, repository_visibility="public", github_proxy=args.github_proxy,
        )
        for receipt in restore(github, args.index_asset, args.volume, args.destination):
            print(json.dumps(receipt), flush=True)
    except BackupError as error:
        parser.exit(1, f"Restore failed: {error}\n")
    except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError) as error:
        # An exception value may include a short-lived signed URL.
        parser.exit(1, f"Restore failed: {type(error).__name__}\n")


if __name__ == "__main__":
    main()
