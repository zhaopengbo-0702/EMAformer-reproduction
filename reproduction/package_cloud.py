"""Create source-only upload archive; public data are transferred separately."""
from pathlib import Path
import hashlib
import json
import tarfile


def main():
    root = Path(__file__).resolve().parent.parent
    sources = []
    for directory in ("EMAformer", "reproduction", "reproduction_extensions"):
        for path in (root / directory).rglob("*"):
            relative = path.relative_to(root)
            if not path.is_file() or any(part in (".git", "__pycache__", "dataset") for part in relative.parts):
                continue
            if path.suffix in (".py", ".sh", ".txt", ".json", ".md") or path.name == "LICENSE":
                sources.append(path)
    for name in ("outputs/paper_reference_results.json", "outputs/paper_experiment_inventory.md",
                 "outputs/code_audit.md", "datasets/manifest.json", "datasets/manifest.csv", "datasets/README.md"):
        path = root / name
        if path.exists():
            sources.append(path)
    sources += [p for p in (root / "tools").glob("*.py")]
    archive = root / "outputs/cloud_code.tar.gz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz", compresslevel=6) as tar:
        for path in sorted(set(sources)):
            tar.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    record = {"file": str(archive), "bytes": archive.stat().st_size, "sha256": digest,
              "members": len(set(sources)), "excludes": ["data files", ".git", "__pycache__", "local dependency environments"]}
    (root / "outputs/cloud_code_package.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
