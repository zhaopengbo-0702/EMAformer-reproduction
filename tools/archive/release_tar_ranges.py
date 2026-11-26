"""Read-only TAR range hashing and bounded file bodies; no copied raw parts."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import release_split_local as splitter


def stable_source(path, expected_bytes, expected_fingerprint=None):
    path = Path(path)
    if not splitter.file_exists(path) or os.path.exists(splitter.disk_path(str(path) + ".partial")):
        raise ValueError("TAR source is incomplete or has download.partial sibling")
    fingerprint = splitter.fingerprint(path)
    if fingerprint[0] != expected_bytes:
        raise ValueError("TAR source size differs from export manifest")
    if expected_fingerprint is not None and tuple(fingerprint) != tuple(expected_fingerprint):
        raise ValueError("TAR source stat changed since range hash pass")
    return fingerprint


def hash_volume_ranges(source, volume, max_part_bytes):
    """One sequential source read computes every range and the full raw hash."""
    if type(max_part_bytes) is not int or not 1 <= max_part_bytes <= splitter.MAX_PART_BYTES:
        raise ValueError("Range size must be1..1,900,000,000 bytes")
    source = Path(source)
    before = stable_source(source, volume["bytes"])
    whole, parts = hashlib.sha256(), []
    with open(splitter.disk_path(source), "rb") as raw:
        info = os.fstat(raw.fileno())
        if (info.st_size, info.st_mtime_ns) != before:
            raise ValueError("TAR changed before range hashing began")
        for number, offset in enumerate(range(0, volume["bytes"], max_part_bytes), 1):
            length = min(max_part_bytes, volume["bytes"] - offset)
            digest, remaining = hashlib.sha256(), length
            while remaining:
                block = raw.read(min(splitter.BUFFER, remaining))
                if not block:
                    raise ValueError("TAR truncated during range hashing")
                whole.update(block); digest.update(block); remaining -= len(block)
            parts.append({"number":number, "name":volume["name"] + ".part{:03d}".format(number),
                          "offset":offset, "bytes":length, "sha256":digest.hexdigest()})
        if raw.read(1):
            raise ValueError("TAR grew during range hashing")
        info = os.fstat(raw.fileno())
        if (info.st_size, info.st_mtime_ns) != before:
            raise ValueError("Open TAR stat changed during range hashing")
    stable_source(source, volume["bytes"], before)
    if whole.hexdigest() != volume["sha256"].lower():
        raise ValueError("TAR whole SHA256 differs from trusted export manifest")
    return {"name":volume["name"], "bytes":volume["bytes"], "sha256":whole.hexdigest(),
            "status":"complete", "source_whole_hash_verified":True,
            "concatenated_raw_hash_verified":True, "source_fingerprint":list(before),
            "storage_method":"tar-ranges-no-part-copies", "verified_utc":splitter.now(), "parts":parts}


def validate_compatible_parts(existing, current):
    """Preserve raw asset identity across local-parts and TAR-range methods."""
    if existing.get("status") != "complete":
        return
    fields = ("number", "name", "offset", "bytes", "sha256")
    left = [{key:p[key] for key in fields} for p in existing.get("parts", [])]
    right = [{key:p[key] for key in fields} for p in current["parts"]]
    if (existing.get("name"),existing.get("bytes"),existing.get("sha256"),left) != (
            current["name"],current["bytes"],current["sha256"],right):
        raise ValueError("Existing completed split/range index differs from current verified TAR ranges")


def compatible_identity(existing, expected):
    return all(existing.get(key)==expected.get(key) for key in ("name","bytes","sha256"))


def prepare_index(manifest_path, expected_manifest_hash, source_directory, metadata_directory,
                  *, max_part_bytes, selected):
    """Writes only a small range index; never mutates the local-parts index."""
    manifest, manifest_hash = splitter.load_manifest(manifest_path, expected_manifest_hash)
    if len(set(selected)) != len(selected) or not set(selected) <= {v["name"] for v in manifest["volumes"]}:
        raise ValueError("Unknown or duplicate selected range volume")
    metadata_directory = Path(metadata_directory)
    os.makedirs(splitter.disk_path(metadata_directory), exist_ok=True)
    index_path = metadata_directory / "release_ranges_index.json"
    with splitter.exclusive_lock(metadata_directory / ".release-split.lock"):
        prior = splitter.read_json(index_path) if splitter.file_exists(index_path) else None
        copied_path = metadata_directory / "release_parts_index.json"
        copied = splitter.read_json(copied_path) if splitter.file_exists(copied_path) else None
        for old in (prior, copied):
            if old is not None and (old.get("manifest_sha256") != manifest_hash or old.get("maximum_part_bytes") != max_part_bytes):
                raise ValueError("Existing index manifest or part size regime differs")
        rows = {row["name"]:row for row in prior["volumes"]} if prior else {}
        index = {"schema":splitter.SCHEMA,"kind":"local_tar_range_index_no_part_copies",
                 "manifest_name":Path(manifest_path).name,"manifest_sha256":manifest_hash,
                 "maximum_part_bytes":max_part_bytes,"storage_method":"tar-ranges-no-part-copies",
                 "network_accessed":False,"uploaded":False,
                 "volumes":[rows.get(v["name"],{"name":v["name"],"bytes":v["bytes"],"sha256":v["sha256"],"status":"pending","parts":[]}) for v in manifest["volumes"]]}
        for volume in manifest["volumes"]:
            if volume["name"] not in selected:
                continue
            # Remove stale complete state before a fresh source read can fail.
            previous = next(row for row in index["volumes"] if row["name"]==volume["name"])
            old_previous = dict(previous)
            previous["status"]="verifying"
            index["all_manifest_volumes_complete"]=False
            splitter.write_json(index_path,index)
            current = hash_volume_ranges(Path(source_directory)/volume["name"],volume,max_part_bytes)
            if not compatible_identity(old_previous,current):
                raise ValueError("Prior range index volume identity differs")
            validate_compatible_parts(old_previous,current)
            if copied is not None:
                match = next((row for row in copied["volumes"] if row["name"]==volume["name"]),None)
                if match is not None:validate_compatible_parts(match,current)
            index["volumes"]=[current if row["name"]==volume["name"] else row for row in index["volumes"]]
        index["complete_volumes"]=sum(row["status"]=="complete" for row in index["volumes"])
        index["all_manifest_volumes_complete"]=index["complete_volumes"]==len(index["volumes"])
        index["reverified_volumes_this_invocation"]=list(selected)
        index["all_index_parts_reverified_this_invocation"]=len(selected)==len(manifest["volumes"])
        splitter.write_json(index_path,index)
    return index


class BoundedTarReader:
    """Exact range body for requests; tell/seek are range-relative, len is fixed.

    No fileno is exposed, so HTTP libraries cannot infer the full TAR size.
    Every positive read is clipped to remaining bytes. read(-1) is bounded to
    BUFFER to prevent accidental 1.9GB allocation by a consuming client.
    """
    mode = "rb"

    def __init__(self, source, offset, length, source_bytes, expected_fingerprint):
        if offset < 0 or length <= 0 or offset+length > source_bytes:
            raise ValueError("Range is outside the verified TAR")
        self.source=Path(source);self.offset=offset;self.length=length;self.source_bytes=source_bytes
        self.expected_fingerprint=tuple(expected_fingerprint)
        stable_source(self.source,source_bytes,self.expected_fingerprint)
        self.raw=open(splitter.disk_path(self.source),"rb")
        info=os.fstat(self.raw.fileno())
        if (info.st_size,info.st_mtime_ns)!=self.expected_fingerprint:
            self.raw.close();raise ValueError("Opened TAR changed since hash pass")
        self.raw.seek(offset);self.position=0;self.digest=hashlib.sha256();self.hash_position=0;self.closed=False

    def __len__(self):return self.length
    def tell(self):return self.position
    def readable(self):return True
    def seekable(self):return True

    def seek(self, offset, whence=0):
        target=offset if whence==0 else self.position+offset if whence==1 else self.length+offset if whence==2 else None
        if target is None or not 0<=target<=self.length:raise ValueError("Seek outside upload range")
        if target==0:
            self.digest=hashlib.sha256();self.hash_position=0
        elif target!=self.position:
            # Only rewind-to-start is safe for integrity of a streamed body.
            raise ValueError("Nonsequential range seek prohibited")
        self.raw.seek(self.offset+target);self.position=target
        return target

    def read(self, amount=-1):
        if self.closed:raise ValueError("Read on closed TAR range")
        remaining=self.length-self.position
        if not remaining or amount==0:return b""
        amount=min(remaining,splitter.BUFFER if amount is None or amount<0 else amount)
        block=self.raw.read(amount)
        if not block:raise ValueError("TAR truncated while streaming upload")
        if self.hash_position!=self.position:raise ValueError("Upload body was read nonsequentially")
        self.digest.update(block);self.position+=len(block);self.hash_position=self.position
        return block

    def verify_consumed(self, expected_sha256):
        stable_source(self.source,self.source_bytes,self.expected_fingerprint)
        info=os.fstat(self.raw.fileno())
        if (info.st_size,info.st_mtime_ns)!=self.expected_fingerprint:raise ValueError("Open TAR stat changed during upload")
        if self.position!=self.length or self.digest.hexdigest()!=expected_sha256:
            raise ValueError("Uploaded TAR range consumption/hash mismatch")

    def close(self):
        if not self.closed:self.raw.close();self.closed=True
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
