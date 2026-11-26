"""Resumable raw TAR splitting for release assets. Local files only.

Does not upload, contact a network, extract a TAR, deserialize weights, or delete
source/partial files. Each finished part is at most1,900,000,000 bytes. A trusted
export-manifest hash is required; whole source and reconstructed raw hashes are
verified before a volume is marked complete in the index.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat

MAX_PART_BYTES = 1_900_000_000
BUFFER = 4 * 1024 * 1024
SCHEMA = "emaformer-release-raw-parts-v1"


def now():
    return datetime.now(timezone.utc).isoformat()


def disk_path(path):
    value = os.path.abspath(os.fspath(path))
    if os.name == "nt" and not value.startswith("\\\\?\\"):
        return "\\\\?\\UNC\\" + value[2:] if value.startswith("\\\\") else "\\\\?\\" + value
    return value


def no_links(path):
    value = Path(disk_path(path))
    for parent in list(reversed(value.parents)) + [value]:
        try:
            info = os.lstat(parent)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("Symlink/reparse point prohibited: " + str(parent))


def file_exists(path):
    no_links(path)
    return os.path.isfile(disk_path(path))


def size(path):
    no_links(path)
    return os.stat(disk_path(path)).st_size


def fingerprint(path):
    no_links(path)
    info = os.stat(disk_path(path))
    return info.st_size, info.st_mtime_ns


def sha256(path):
    no_links(path)
    digest = hashlib.sha256()
    with open(disk_path(path), "rb") as handle:
        for block in iter(lambda: handle.read(BUFFER), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    no_links(path)
    with open(disk_path(path), "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def write_json(path, value):
    no_links(path)
    temporary = Path(str(path) + ".tmp")
    no_links(temporary)
    with open(disk_path(temporary), "w", encoding="utf8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(disk_path(temporary), disk_path(path))


@contextmanager
def exclusive_lock(path):
    no_links(path)
    with open(disk_path(path), "a+b") as handle:
        if os.fstat(handle.fileno()).st_size == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def load_manifest(path, expected_sha256):
    if not re.fullmatch(r"[a-fA-F0-9]{64}", expected_sha256):
        raise ValueError("Expected manifest SHA256 must be64 hexadecimal characters")
    actual = sha256(path)
    if actual != expected_sha256.lower():
        raise ValueError("Export manifest SHA256 mismatch")
    manifest = read_json(path)
    if manifest.get("schema") != "emaformer-export-v1" or not manifest.get("volumes"):
        raise ValueError("Expected nonempty emaformer-export-v1 manifest")
    names = set()
    for row in manifest["volumes"]:
        name = row.get("name", "")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.tar", name) or name.casefold() in names:
            raise ValueError("Invalid/duplicate TAR basename")
        names.add(name.casefold())
        if type(row.get("bytes")) is not int or row["bytes"] <= 0:
            raise ValueError("Source size must be a positive integer")
        if not re.fullmatch(r"[a-fA-F0-9]{64}", row.get("sha256", "")):
            raise ValueError("Invalid source SHA256")
    return manifest, actual


def compare_segment(source, part, offset, expected_size):
    """Read-only exact comparison; a conflicting partial is never overwritten."""
    existing = size(part)
    if existing > expected_size:
        raise ValueError("Part is larger than its expected segment: " + str(part))
    digest = hashlib.sha256()
    with open(disk_path(source), "rb") as raw, open(disk_path(part), "rb") as chunk:
        raw.seek(offset)
        remaining = existing
        while remaining:
            block = chunk.read(min(BUFFER, remaining))
            if not block or raw.read(len(block)) != block:
                raise ValueError("Part prefix differs from verified source; preserved: " + str(part))
            digest.update(block)
            remaining -= len(block)
    return existing, digest


def split_volume(source, volume, destination, max_part_bytes):
    if os.path.exists(disk_path(str(source) + ".partial")):
        raise ValueError("Downloaded source still has a .partial sibling; do not read it")
    before = fingerprint(source)
    if before[0] != volume["bytes"] or sha256(source) != volume["sha256"].lower():
        raise ValueError("Source whole TAR size/SHA256 mismatch: " + volume["name"])
    if fingerprint(source) != before:
        raise ValueError("Source changed during initial hash verification")
    parts = []
    # Sequential parts minimize external-drive seek contention. Completed parts
    # and interrupted prefixes are compared against the verified source first.
    for number, offset in enumerate(range(0, volume["bytes"], max_part_bytes), start=1):
        expected = min(max_part_bytes, volume["bytes"] - offset)
        name = volume["name"] + ".part{:03d}".format(number)
        target = destination / name
        partial = destination / (name + ".partial")
        no_links(target); no_links(partial)
        if os.path.exists(disk_path(target)):
            if not file_exists(target) or size(target) != expected:
                raise ValueError("Existing finished part size/type conflict; preserved: " + str(target))
            _, digest = compare_segment(source, target, offset, expected)
        else:
            if os.path.exists(disk_path(partial)) and not file_exists(partial):
                raise ValueError("Partial must be a regular file")
            prior, digest = compare_segment(source, partial, offset, expected) if file_exists(partial) else (0, hashlib.sha256())
            with open(disk_path(source), "rb") as raw, open(disk_path(partial), "ab") as chunk:
                raw.seek(offset + prior)
                remaining = expected - prior
                while remaining:
                    block = raw.read(min(BUFFER, remaining))
                    if not block:
                        raise ValueError("Verified source truncated while splitting")
                    chunk.write(block)
                    digest.update(block)
                    remaining -= len(block)
                chunk.flush()
                os.fsync(chunk.fileno())
            expected_part_hash = digest.hexdigest()
            if size(partial) != expected or sha256(partial) != expected_part_hash:
                raise ValueError("Part destination readback mismatch; partial preserved")
            if os.path.exists(disk_path(target)):
                raise ValueError("Finished part appeared during writing; both files preserved")
            os.rename(disk_path(partial), disk_path(target))
        parts.append({"number": number, "name": name, "offset": offset,
                      "bytes": expected, "sha256": digest.hexdigest()})
    # Independently read the finished parts in order. This verifies both every
    # destination part and the exact concatenated raw TAR, without creating a
    # second assembled copy or loading the TAR into memory.
    whole = hashlib.sha256()
    for row in parts:
        digest = hashlib.sha256()
        with open(disk_path(destination / row["name"]), "rb") as handle:
            for block in iter(lambda: handle.read(BUFFER), b""):
                digest.update(block); whole.update(block)
        if digest.hexdigest() != row["sha256"]:
            raise ValueError("Finished part changed during final verification")
    if whole.hexdigest() != volume["sha256"].lower() or fingerprint(source) != before:
        raise ValueError("Reconstructed raw hash/source stability check failed")
    return {"name": volume["name"], "bytes": volume["bytes"], "sha256": volume["sha256"].lower(),
            "status": "complete", "source_whole_hash_verified": True,
            "concatenated_raw_hash_verified": True, "verified_utc": now(), "parts": parts}


def run(manifest_path, expected_manifest_sha256, source_directory, destination,
        *, max_part_bytes=MAX_PART_BYTES, selected=None):
    if type(max_part_bytes) is not int or not 1 <= max_part_bytes <= MAX_PART_BYTES:
        raise ValueError("Part limit must be1..1,900,000,000 bytes")
    manifest_path = Path(manifest_path).absolute()
    source_directory, destination = Path(source_directory).absolute(), Path(destination).absolute()
    no_links(source_directory); no_links(destination)
    manifest, manifest_hash = load_manifest(manifest_path, expected_manifest_sha256)
    known = {row["name"] for row in manifest["volumes"]}
    if selected and (len(set(selected)) != len(selected) or not set(selected) <= known):
        raise ValueError("Unknown or duplicate selected volume")
    os.makedirs(disk_path(destination), exist_ok=True)
    index_path = destination / "release_parts_index.json"
    with exclusive_lock(destination / ".release-split.lock"):
        prior = read_json(index_path) if file_exists(index_path) else None
        if prior and (prior.get("schema") != SCHEMA or prior.get("manifest_sha256") != manifest_hash
                      or prior.get("maximum_part_bytes") != max_part_bytes):
            raise ValueError("Existing index identifies a different manifest or part limit")
        by_name = {row["name"]: row for row in prior.get("volumes", [])} if prior else {}
        index = {"schema": SCHEMA, "kind": "local_raw_release_parts_not_uploaded",
                 "manifest_name": manifest_path.name, "manifest_sha256": manifest_hash,
                 "maximum_part_bytes": max_part_bytes, "created_utc": prior.get("created_utc", now()) if prior else now(),
                 "volumes": [by_name.get(v["name"], {"name": v["name"], "bytes": v["bytes"],
                               "sha256": v["sha256"].lower(), "status": "pending", "parts": []}) for v in manifest["volumes"]],
                 "network_accessed": False, "uploaded": False}
        write_json(index_path, index)
        reverified_this_invocation = []
        for volume in manifest["volumes"]:
            if selected and volume["name"] not in selected:
                continue
            source = source_directory / volume["name"]
            if not file_exists(source) or os.path.exists(disk_path(str(source) + ".partial")):
                if selected:
                    raise FileNotFoundError("Selected volume not fully downloaded: " + volume["name"])
                continue
            print("Verifying and splitting " + volume["name"], flush=True)
            # A failed resumed verification must not leave a stale complete
            # claim from an older index. Preserve its ordered part descriptions
            # for diagnostics, but withhold completion until all checks pass.
            for row in index["volumes"]:
                if row["name"] == volume["name"]:
                    row["status"] = "verifying"
            index["complete_volumes"] = sum(row["status"] == "complete" for row in index["volumes"])
            index["all_manifest_volumes_complete"] = False
            write_json(index_path, index)
            completed = split_volume(source, volume, destination, max_part_bytes)
            reverified_this_invocation.append(volume["name"])
            index["volumes"] = [completed if row["name"] == volume["name"] else row for row in index["volumes"]]
            index["updated_utc"] = now()
            index["complete_volumes"] = sum(row["status"] == "complete" for row in index["volumes"])
            index["all_manifest_volumes_complete"] = index["complete_volumes"] == len(index["volumes"])
            write_json(index_path, index)
        index["complete_volumes"] = sum(row["status"] == "complete" for row in index["volumes"])
        index["all_manifest_volumes_complete"] = index["complete_volumes"] == len(index["volumes"])
        index["reverified_volumes_this_invocation"] = reverified_this_invocation
        index["all_index_parts_reverified_this_invocation"] = (
            len(reverified_this_invocation) == len(manifest["volumes"]) and index["all_manifest_volumes_complete"])
        write_json(index_path, index)
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--source-directory", type=Path)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--max-part-bytes", type=int, default=MAX_PART_BYTES)
    parser.add_argument("--volumes", nargs="+")
    args = parser.parse_args()
    result = run(args.manifest, args.manifest_sha256, args.source_directory or args.manifest.parent,
                 args.destination, max_part_bytes=args.max_part_bytes, selected=args.volumes)
    print(json.dumps({"index": str(args.destination / "release_parts_index.json"),
                      "complete_volumes": result["complete_volumes"],
                      "planned_volumes": len(result["volumes"]), "all_complete": result["all_manifest_volumes_complete"],
                      "uploaded": False}, ensure_ascii=True))


if __name__ == "__main__":
    main()
