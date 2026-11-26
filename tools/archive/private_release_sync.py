"""Sequential resumable backup to one existing unpublished draft GitHub Release.

Script-only preparation: invoking main performs network writes. This module does
not publish releases, change repository visibility, remove uploaded assets, or
touch the cloud source. Credentials stay in memory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.parse import quote, urlsplit

import release_split_local as splitter

EXPECTED_USER = "zhaopengbo-0702"
EXPECTED_REPOSITORY = "zhaopengbo-0702/EMAformer-full-archive"
EXPECTED_RELEASE_ID = 459258716
EXPECTED_TAG = "emaformer-main288-20261126"
EXPECTED_MANIFEST = "fd048a8b7e4559ab847210e36e9af55250824aa99cd1fb3869f7f3cff84c7d0b"
ALLOWED_GITHUB_PROXY = "http://127.0.0.1:7890"
MERGED_REPOSITORY = "zhaopengbo-0702/EMAformer-reproduction"
MERGED_REPOSITORY_ID = 1471776241
ALLOWED_REPOSITORIES = (EXPECTED_REPOSITORY, MERGED_REPOSITORY)


class BackupError(RuntimeError):
    pass


def checked_destination(repository, release_id):
    if repository not in ALLOWED_REPOSITORIES:
        raise BackupError("Repository is not an approved backup destination")
    if release_id is None and repository == EXPECTED_REPOSITORY:
        release_id = EXPECTED_RELEASE_ID
    if type(release_id) is not int or release_id <= 0:
        raise BackupError("An explicit positive Release ID is required for the selected repository")
    if repository == EXPECTED_REPOSITORY and release_id != EXPECTED_RELEASE_ID:
        raise BackupError("Original archive repository must use its exact approved Release ID")
    return repository, release_id


def credential_token(repository=EXPECTED_REPOSITORY):
    if repository not in ALLOWED_REPOSITORIES:
        raise BackupError("Repository is not an approved credential destination")
    # Credential helper supplies the existing GitHub credential. Do not print
    # stdout/stderr, pass it to a child argv, or persist it to a temporary file.
    request = "protocol=https\nhost=github.com\npath=" + repository + ".git\n\n"
    result = subprocess.run(["git", "credential", "fill"], input=request, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            env={**os.environ, "GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "Never"},
                            timeout=60)
    if result.returncode:
        raise BackupError("Existing GitHub credential helper did not return credentials")
    fields = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    token = fields.get("password")
    if not token:
        raise BackupError("GitHub credential has no password/token field")
    return token


def asset_matches(asset, expected_size, expected_sha256):
    return (asset.get("state") == "uploaded" and asset.get("size") == expected_size
            and asset.get("digest") == "sha256:" + expected_sha256)


def acceptance_passed(value):
    backup = value.get("local_backup", {})
    return (value.get("kind") == "main288_only_closeout"
            and value.get("verified_main_runs") == 288
            and value.get("training_complete") is True
            and value.get("table1_EMAformer_roles_complete") is True
            and value.get("table5_EMAformer_roles_complete") is True
            and value.get("main_closeout_complete") is True
            and value.get("full_paper_reproduction_claim") is False
            and backup.get("verified_unique_runs") == 288
            and backup.get("selected_output_fully_backed_up") is True)


class GitHub:
    def __init__(self, token, expected_tag, upload_timeout=1800, session=None, *, repository_visibility="private",
                 github_proxy=None, repository=EXPECTED_REPOSITORY, release_id=None):
        import requests
        self.repository, self.release_id = checked_destination(repository, release_id)
        self.repository_id = MERGED_REPOSITORY_ID if repository == MERGED_REPOSITORY else None
        if expected_tag != EXPECTED_TAG:
            raise BackupError("Only the exact approved draft release tag is supported")
        if repository_visibility not in ("private", "public"):
            raise BackupError("Repository visibility must be private or public")
        if repository == MERGED_REPOSITORY and repository_visibility != "public":
            raise BackupError("Merged reproduction destination must be explicitly public")
        if github_proxy is not None and github_proxy != ALLOWED_GITHUB_PROXY:
            raise BackupError("GitHub proxy must be the exact approved local proxy")
        self.session = session or requests.Session()
        self.session.trust_env = False
        # Explicit process-local routing only; never inherit environment,
        # registry proxy discovery, netrc credentials or certificate overrides.
        self.session.proxies = {"http": github_proxy, "https": github_proxy} if github_proxy else {}
        self.token = token
        self.expected_tag = expected_tag
        self.repository_visibility = repository_visibility
        self.upload_timeout = upload_timeout
        self.base = "https://api.github.com/repos/" + self.repository
        self.upload_base = ("https://uploads.github.com/repos/" + self.repository
                            + "/releases/" + str(self.release_id) + "/assets")

    def request(self, method, url, *, accept="application/vnd.github+json", allow_asset_redirect=False, **kwargs):
        import requests
        parsed = urlsplit(url)
        if parsed.scheme != "https" or parsed.hostname not in ("api.github.com", "uploads.github.com"):
            raise BackupError("Refused authenticated request to unexpected host")
        headers = {"Authorization": "Bearer " + self.token, "Accept": accept,
                   "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "EMAformer-private-release-backup"}
        headers.update(kwargs.pop("headers", {}))
        try:
            response = self.session.request(method, url, headers=headers, allow_redirects=False,
                                            timeout=kwargs.pop("timeout", (30, 180)), **kwargs)
        except requests.RequestException as error:
            # Exception strings can contain signed URLs. Report class only.
            raise BackupError("GitHub transport failed: " + type(error).__name__) from None
        if allow_asset_redirect and response.status_code in (301,302,303,307,308):
            return response
        if response.status_code == 413:
            response.close()
            raise BackupError("Upload ingress rejected size (413). Rerun with a smaller --max-part-bytes and NEW --parts-directory; existing assets remain untouched.")
        if not 200 <= response.status_code < 300:
            code = response.status_code
            response.close()
            raise BackupError("GitHub API returned HTTP" + str(code))
        return response

    def verify_identity(self):
        user = self.request("GET", "https://api.github.com/user").json()
        if user.get("login") != EXPECTED_USER:
            raise BackupError("Authenticated GitHub account is not the exact expected user")
        repo = self.request("GET", self.base).json()
        if (repo.get("full_name") != self.repository
                or (self.repository_id is not None and repo.get("id") != self.repository_id)
                or repo.get("private") is not (self.repository_visibility == "private")
                or repo.get("visibility") != self.repository_visibility):
            raise BackupError("Expected exact " + self.repository_visibility.upper() + " repository; refusing upload")
        release = self.request("GET", self.base + "/releases/" + str(self.release_id)).json()
        if release.get("id") != self.release_id or release.get("tag_name") != self.expected_tag:
            raise BackupError("Release ID/tag mismatch")
        if release.get("draft") is not True or release.get("published_at") is not None:
            raise BackupError("Release must remain unpublished draft")
        if release.get("upload_url", "").split("{", 1)[0] != self.upload_base:
            raise BackupError("Release upload URL is not the expected endpoint")
        return release

    def assets(self):
        result = []
        page = 1
        while True:
            rows = self.request("GET", self.base + "/releases/" + str(self.release_id)
                                + "/assets?per_page=100&page=" + str(page)).json()
            if not isinstance(rows, list):
                raise BackupError("Malformed release asset listing")
            result.extend(rows)
            if len(rows) < 100:
                break
            page += 1
        names = [asset.get("name") for asset in result]
        if len(names) != len(set(names)):
            raise BackupError("Duplicate remote asset names; reconcile before continuing")
        return {asset["name"]: asset for asset in result}

    def downloaded_digest(self, asset_id):
        """Fallback when GitHub omits digest; authenticated redirect handled safely."""
        response = self.request("GET", self.base + "/releases/assets/" + str(asset_id),
                                accept="application/octet-stream", stream=True, allow_asset_redirect=True)
        if response.status_code in (301,302,303,307,308):
            redirect = response.headers.get("Location", "")
            response.close()
            parsed = urlsplit(redirect)
            if parsed.scheme != "https" or parsed.hostname not in ("release-assets.githubusercontent.com", "objects.githubusercontent.com", "github-releases.githubusercontent.com"):
                raise BackupError("Refused release asset readback redirect host")
            # Separate unauthenticated request: never forward the credential
            # header to the signed download endpoint or persist/log its URL.
            import requests
            try:
                response = self.session.get(redirect, stream=True, allow_redirects=False, timeout=(30,180))
            except requests.RequestException as error:
                raise BackupError("Remote asset readback failed: " + type(error).__name__) from None
            if response.status_code != 200:
                response.close()
                raise BackupError("Remote asset readback non200 response")
        digest, count = hashlib.sha256(), 0
        try:
            for block in response.iter_content(chunk_size=splitter.BUFFER):
                if block:
                    digest.update(block); count += len(block)
        finally:
            response.close()
        return count, digest.hexdigest()

    def verify_asset(self, asset, expected_size, expected_sha256):
        if asset.get("state") != "uploaded" or asset.get("size") != expected_size:
            raise BackupError("Existing remote asset has wrong size/state: " + str(asset.get("name")))
        if asset.get("digest"):
            if not asset_matches(asset, expected_size, expected_sha256):
                raise BackupError("Remote SHA256 digest mismatch: " + str(asset.get("name")))
        else:
            count, digest = self.downloaded_digest(asset["id"])
            if count != expected_size or digest != expected_sha256:
                raise BackupError("Remote readback hash mismatch: " + str(asset.get("name")))

    def clear_failed_upload(self, asset, name, expected_size, expected_sha256):
        """Clear a GitHub failed-upload placeholder during an exclusive retry."""
        if (self.repository != MERGED_REPOSITORY or asset.get("name") != name
                or asset.get("state") != "starter" or asset.get("digest")
                or asset.get("size") not in (0, expected_size)
                or type(asset.get("id")) is not int or asset["id"] <= 0):
            self.verify_asset(asset, expected_size, expected_sha256)
            return False
        response = self.request("DELETE", self.base + "/releases/assets/" + str(asset["id"]))
        response.close()
        return True

    def ensure_asset(self, path, remote_name, expected_size, expected_sha256, *, retries=3):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", remote_name):
            raise BackupError("Asset name must be ASCII-safe basename")
        if splitter.size(path) != expected_size or splitter.sha256(path) != expected_sha256:
            raise BackupError("Local upload asset size/hash mismatch")
        for attempt in range(retries):
            self.verify_identity()  # Reject visibility/draft changes immediately.
            existing = self.assets().get(remote_name)
            if existing and not self.clear_failed_upload(existing, remote_name, expected_size, expected_sha256):
                return {"name": remote_name, "asset_id": existing["id"], "bytes": expected_size,
                        "sha256": expected_sha256, "remote_digest_verified": True}
            before = splitter.fingerprint(path)
            try:
                with open(splitter.disk_path(path), "rb") as stream:
                    response = self.request("POST", self.upload_base + "?name=" + quote(remote_name),
                        data=stream, headers={"Content-Type": "application/octet-stream", "Content-Length": str(expected_size)},
                        timeout=(30, self.upload_timeout))
                    response.close()
            except BackupError as error:
                if "(413)" in str(error):
                    raise
                # A timed-out POST may actually have completed. Re-read API
                # Check state before retrying an upload with a lost response.
                existing = self.assets().get(remote_name)
                if existing and existing.get("state") == "uploaded":
                    self.verify_asset(existing, expected_size, expected_sha256)
                elif attempt + 1 == retries:
                    raise
                else:
                    time.sleep(min(30, 5 * (attempt + 1)))
                    continue
            if splitter.fingerprint(path) != before:
                raise BackupError("Local file changed during upload")
            remote = self.assets().get(remote_name)
            if not remote:
                raise BackupError("Uploaded asset missing from release listing")
            self.verify_asset(remote, expected_size, expected_sha256)
            return {"name": remote_name, "asset_id": remote["id"], "bytes": expected_size,
                    "sha256": expected_sha256, "remote_digest_verified": True}
        raise BackupError("Upload retry limit exhausted")

    def ensure_tar_range_asset(self, source, part, volume, remote_name, *, retries=3):
        """Opt-in raw TAR range stream; same asset bytes/names as local parts."""
        from release_tar_ranges import BoundedTarReader, stable_source
        if not re.fullmatch(r"[A-Za-z0-9_.-]+",remote_name):
            raise BackupError("Asset name must be ASCII-safe basename")
        stable_source(source,volume["bytes"],volume["source_fingerprint"])
        expected_size,expected_hash=part["bytes"],part["sha256"]
        for attempt in range(retries):
            self.verify_identity()
            existing=self.assets().get(remote_name)
            if existing and not self.clear_failed_upload(existing,remote_name,expected_size,expected_hash):
                stable_source(source,volume["bytes"],volume["source_fingerprint"])
                return {"name":remote_name,"asset_id":existing["id"],"bytes":expected_size,
                        "sha256":expected_hash,"remote_digest_verified":True}
            try:
                with BoundedTarReader(source,part["offset"],expected_size,volume["bytes"],volume["source_fingerprint"]) as body:
                    response=self.request("POST",self.upload_base+"?name="+quote(remote_name),
                        data=body,headers={"Content-Type":"application/octet-stream","Content-Length":str(expected_size)},
                        timeout=(30,self.upload_timeout))
                    response.close()
                    body.verify_consumed(expected_hash)
            except BackupError as error:
                stable_source(source,volume["bytes"],volume["source_fingerprint"])
                if "(413)" in str(error):raise
                existing=self.assets().get(remote_name)
                if existing and existing.get("state")=="uploaded":
                    self.verify_asset(existing,expected_size,expected_hash)
                elif attempt+1==retries:raise
                else:
                    time.sleep(min(30,5*(attempt+1)));continue
            stable_source(source,volume["bytes"],volume["source_fingerprint"])
            remote=self.assets().get(remote_name)
            if not remote:raise BackupError("Uploaded TAR range asset missing from listing")
            self.verify_asset(remote,expected_size,expected_hash)
            return {"name":remote_name,"asset_id":remote["id"],"bytes":expected_size,
                    "sha256":expected_hash,"remote_digest_verified":True}
        raise BackupError("TAR range upload retry limit exhausted")


def immutable_json_asset(path, value):
    raw = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf8")
    digest = hashlib.sha256(raw).hexdigest()
    target = path.with_name(path.stem + "_" + digest[:16] + path.suffix)
    if splitter.file_exists(target):
        if splitter.sha256(target) != digest:
            raise BackupError("Immutable metadata filename conflict")
    else:
        with open(splitter.disk_path(target), "xb") as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    return target, len(raw), digest


def find_acceptance(explicit, project_root):
    candidates = [explicit] if explicit else list(project_root.glob("验收报告_*/acceptance.json"))
    for path in sorted((p for p in candidates if p is not None), reverse=True):
        if splitter.file_exists(path):
            value = splitter.read_json(path)
            if acceptance_passed(value):
                return path, value
    return None, None


def run(args, github):
    manifest, manifest_hash = splitter.load_manifest(args.manifest, EXPECTED_MANIFEST)
    if len(manifest["volumes"]) != 25:
        raise BackupError("Expected exact25-volume main export manifest")
    github.verify_identity()
    os.makedirs(splitter.disk_path(args.parts_directory), exist_ok=True)
    state_path = args.parts_directory / "private_release_sync_state.json"
    with splitter.exclusive_lock(args.parts_directory / ".private-release-sync.lock"):
        state = splitter.read_json(state_path) if splitter.file_exists(state_path) else {
            "kind": "private_draft_release_backup", "repository": github.repository,
            "repository_id": github.repository_id, "release_id": github.release_id, "tag": args.expected_tag,
            "manifest_sha256": manifest_hash, "maximum_part_bytes": args.max_part_bytes,
            "uploaded_parts": {}, "verified_volumes": [], "created_utc": splitter.now()}
        if (state.get("repository") != github.repository or state.get("release_id") != github.release_id
                or (github.repository_id is not None and state.get("repository_id") != github.repository_id)
                or state.get("tag") != args.expected_tag or state.get("manifest_sha256") != manifest_hash
                or state.get("maximum_part_bytes") != args.max_part_bytes):
            raise BackupError("Existing worker state differs; use a fresh parts directory")
        # State has no token, private credential URL, or source absolute paths.
        def save(status):
            state.update(status=status, updated_utc=splitter.now(), release_published=False,
                         repository_visibility=github.repository_visibility)
            splitter.write_json(state_path, state)
        # Upload the ORIGINAL manifest bytes, preserving the trusted whole-file
        # hash (JSON reserialization would create a different byte identity).
        manifest_name = "export_manifest_" + manifest_hash[:16] + ".json"
        github.ensure_asset(args.manifest, manifest_name, splitter.size(args.manifest), manifest_hash)
        save("waiting_for_completed_local_volumes")
        for volume in manifest["volumes"]:
            source = args.source_directory / volume["name"]
            while (not splitter.file_exists(source)
                   or splitter.size(source) != volume["bytes"]
                   or os.path.exists(splitter.disk_path(str(source) + ".partial"))):
                if args.stop_request.exists():
                    save("stopped_at_safe_asset_boundary"); return state
                if args.once:
                    save("waiting_for_completed_local_volumes"); return state
                time.sleep(args.poll_seconds)
            upload_method=getattr(args,"upload_method","local-parts")
            state["current_upload_method"]=upload_method
            if upload_method=="tar-ranges":
                from release_tar_ranges import prepare_index
                index=prepare_index(args.manifest,EXPECTED_MANIFEST,args.source_directory,args.parts_directory,
                                    max_part_bytes=args.max_part_bytes,selected=[volume["name"]])
                # Include prior physical split rows in the remote reconstruction
                # index. They share exact offsets/names/hashes and need not be
                # rewritten when switching methods at a volume boundary.
                physical_index_path=args.parts_directory/"release_parts_index.json"
                if splitter.file_exists(physical_index_path):
                    physical=splitter.read_json(physical_index_path)
                    physical_rows={r["name"]:r for r in physical["volumes"] if r.get("status")=="complete"}
                    index["volumes"]=[physical_rows.get(r["name"],r) if r.get("status")!="complete" else r for r in index["volumes"]]
                    index["complete_volumes"]=sum(r.get("status")=="complete" for r in index["volumes"])
                    index["all_manifest_volumes_complete"]=index["complete_volumes"]==len(index["volumes"])
            else:
                # Existing default: physical part files and destination reread.
                index = splitter.run(args.manifest, EXPECTED_MANIFEST, args.source_directory, args.parts_directory,
                                     max_part_bytes=args.max_part_bytes, selected=[volume["name"]])
            row = next(v for v in index["volumes"] if v["name"] == volume["name"])
            if row["status"] != "complete":
                raise BackupError("Local split did not complete")
            for part in row["parts"]:
                if args.stop_request.exists():
                    save("stopped_at_safe_asset_boundary"); return state
                name = part["name"] if args.max_part_bytes == splitter.MAX_PART_BYTES else (
                    "raw" + str(args.max_part_bytes) + "-" + part["name"])
                save("uploading_sequential_asset")
                if upload_method=="tar-ranges":
                    receipt=github.ensure_tar_range_asset(source,part,row,name)
                else:
                    receipt = github.ensure_asset(args.parts_directory / part["name"], name, part["bytes"], part["sha256"])
                state["uploaded_parts"][name] = receipt
                save("asset_remote_size_digest_verified")
                print(json.dumps({"asset":name,"bytes":part["bytes"],"remote_digest_verified":True}), flush=True)
            if volume["name"] not in state["verified_volumes"]:
                state["verified_volumes"].append(volume["name"])
            public_index = {"schema": "emaformer-release-backup-index-v1", "repository": github.repository,
                "repository_id": github.repository_id, "release_id": github.release_id, "manifest_sha256": manifest_hash,
                "maximum_part_bytes": args.max_part_bytes, "volume_order": [v["name"] for v in manifest["volumes"]],
                "complete_uploaded_volumes": state["verified_volumes"], "local_split_index": index,
                "remote_parts": list(state["uploaded_parts"].values()), "release_remains_draft": True}
            target, count, digest = immutable_json_asset(args.parts_directory / "backup_index.json", public_index)
            github.ensure_asset(target, target.name, count, digest)
            save("volume_remote_backup_verified")
        while True:
            path, acceptance = find_acceptance(args.acceptance, args.project_root)
            if acceptance is not None:
                # Final remote re-list verifies ALL expected asset sizes and
                # server digests. Source-scientific acceptance stays separate.
                assets = github.assets()
                for part in state["uploaded_parts"].values():
                    if part["name"] not in assets:
                        raise BackupError("Previously uploaded remote part is missing")
                    github.verify_asset(assets[part["name"]], part["bytes"], part["sha256"])
                target, count, digest = immutable_json_asset(args.parts_directory / "local_strict_acceptance.json", acceptance)
                github.ensure_asset(target, target.name, count, digest)
                github.verify_identity()
                save("all25_remote_backups_and_local_main288_acceptance_verified_draft_ready")
                return state
            if args.stop_request.exists() or args.once:
                save("all25_remote_backups_waiting_for_local_strict_acceptance"); return state
            save("all25_remote_backups_waiting_for_local_strict_acceptance")
            time.sleep(args.poll_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", choices=ALLOWED_REPOSITORIES, default=EXPECTED_REPOSITORY,
                        help="Exact approved repository; merged reproduction target additionally checks repository ID")
    parser.add_argument("--release-id", type=int,
                        help="Exact existing draft Release ID; required for merged reproduction destination")
    parser.add_argument("--expected-tag", default=EXPECTED_TAG, help="Exact existing draft release tag")
    parser.add_argument("--repository-visibility", choices=["private", "public"], default="private",
                        help="Required existing repository visibility; public requires explicit opt-in; release must remain draft")
    parser.add_argument("--github-proxy", choices=[ALLOWED_GITHUB_PROXY],
                        help="Opt-in existing local proxy for GitHub API/assets; default is direct; environment settings remain disabled")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--source-directory", required=True, type=Path)
    parser.add_argument("--parts-directory", required=True, type=Path)
    parser.add_argument("--project-root", type=Path, default=Path(r"E:\EMAformer复现"))
    parser.add_argument("--acceptance", type=Path)
    parser.add_argument("--max-part-bytes", type=int, default=splitter.MAX_PART_BYTES)
    parser.add_argument("--upload-method",choices=["local-parts","tar-ranges"],default="local-parts",
                        help="Opt-in tar-ranges hashes raw ranges and streams them directly, writing no copied parts")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--upload-timeout", type=int, default=1800)
    parser.add_argument("--stop-request", type=Path)
    parser.add_argument("--once", action="store_true", help="Process ready volumes; return instead of waiting")
    args = parser.parse_args()
    if not 1 <= args.max_part_bytes <= splitter.MAX_PART_BYTES:
        parser.error("max-part-bytes must be1..1,900,000,000")
    if not 5 <= args.poll_seconds <= 300 or not 60 <= args.upload_timeout <= 7200:
        parser.error("poll5..300seconds; upload timeout60..7200seconds")
    if args.expected_tag != EXPECTED_TAG:
        parser.error("Only exact approved draft release tag is supported")
    try:
        args.repository, args.release_id = checked_destination(args.repository, args.release_id)
        if args.repository == MERGED_REPOSITORY and args.repository_visibility != "public":
            parser.error("Merged reproduction destination requires --repository-visibility public")
    except BackupError as error:
        parser.error(str(error))
    args.stop_request = args.stop_request or args.parts_directory / "private_release_stop.request"
    github = GitHub(credential_token(args.repository), args.expected_tag, args.upload_timeout,
                    repository_visibility=args.repository_visibility, github_proxy=args.github_proxy,
                    repository=args.repository, release_id=args.release_id)
    result = run(args, github)
    print(json.dumps({"status": result["status"], "uploaded_parts": len(result["uploaded_parts"]),
                      "verified_volumes": len(result["verified_volumes"]), "release_published": False}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except (BackupError, ValueError, OSError, subprocess.SubprocessError) as error:
        # Never emit response bodies, credential fields or raw network errors.
        message = str(error) if isinstance(error, (BackupError, ValueError)) else type(error).__name__
        print("STOPPED: " + message, file=sys.stderr)
        raise SystemExit(1)
