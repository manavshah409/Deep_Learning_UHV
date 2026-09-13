"""Package the portable progress deliverable without data, weights or local paths."""

from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "deliverables/UVH26_Faculty_Review_Project.zip"


def eligible(path):
    rel = path.relative_to(ROOT)
    if "__pycache__" in rel.parts or path.suffix in {".pyc", ".pt", ".log", ".cache"}:
        return False
    if rel.as_posix() == "configs/paths.local.yaml":
        return False
    return path.is_file() and not path.is_symlink()


def main():
    files = []
    for name in [
        "README.md",
        "README_STREAMLIT.md",
        "app.py",
        ".gitattributes",
        "requirements-dashboard.txt",
        "CHANGELOG.md",
        ".gitignore",
        "requirements.txt",
        "pyproject.toml",
        "data/README.md",
        "models/README.md",
    ]:
        files.append(ROOT / name)
    for name in [
        "src",
        "components",
        "utils",
        "assets",
        "tests",
        "configs",
        "notebooks",
        "docs",
        "scripts",
        "reports/audit",
        "reports/tables",
        "reports/figures",
        "reports/error_analysis",
        "output/pdf",
    ]:
        files.extend(p for p in (ROOT / name).rglob("*") if eligible(p))
    files = sorted(set(files))
    manifest = []
    for path in files:
        if not eligible(path):
            raise ValueError(f"Ineligible file: {path}")
        content = path.read_bytes()
        if (
            path.suffix in {".py", ".md", ".yaml", ".json", ".toml", ".txt", ".csv"}
            and (str(Path.home()) + "/").encode() in content
        ):
            raise ValueError(f"Machine-specific path in {path.relative_to(ROOT)}")
        manifest.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    DEST.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(DEST, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in files:
            z.write(
                path,
                "UVH26_Faculty_Review_Project/" + path.relative_to(ROOT).as_posix(),
            )
        z.writestr(
            "UVH26_Faculty_Review_Project/PACKAGE_MANIFEST.json",
            json.dumps(manifest, indent=2) + "\n",
        )
        z.write(
            ROOT / "docs/faculty_review/START_HERE.md",
            "UVH26_Faculty_Review_Project/START_HERE.md",
        )
    with zipfile.ZipFile(DEST) as z:
        if z.testzip() is not None:
            raise ValueError("ZIP verification failed")
        forbidden = [
            n
            for n in z.namelist()
            if "/data/raw/" in n
            or "/data/processed/" in n
            or "/runs/" in n
            or n.endswith(".pt")
            or "/.venv/" in n
            or n.endswith("paths.local.yaml")
        ]
        if forbidden:
            raise ValueError(f"Forbidden package entries: {forbidden}")
    digest = hashlib.sha256(DEST.read_bytes()).hexdigest()
    (DEST.parent / (DEST.name + ".sha256")).write_text(f"{digest}  {DEST.name}\n")
    print(
        f"{DEST.name}: {len(files)} files + manifest/start guide; {DEST.stat().st_size:,} bytes; SHA-256 {digest}"
    )


if __name__ == "__main__":
    main()
