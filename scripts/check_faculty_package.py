"""Extract and verify the portable faculty package, including its offline demo."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    package = ROOT / "deliverables/UVH26_Faculty_Review_Project.zip"
    with tempfile.TemporaryDirectory(prefix="uvh26-faculty-") as tmp:
        with zipfile.ZipFile(package) as z:
            assert z.testzip() is None
            for name in z.namelist():
                assert not Path(name).is_absolute() and ".." not in Path(name).parts
                assert not any(
                    part in {"runs", ".venv", "raw", "processed", "__pycache__"}
                    for part in Path(name).parts
                )
                assert not name.endswith(
                    (".pt", ".onnx", ".log", ".cache", "paths.local.yaml")
                )
            z.extractall(tmp)
        root = Path(tmp) / "UVH26_Faculty_Review_Project"
        manifest = json.loads((root / "PACKAGE_MANIFEST.json").read_text())
        for row in manifest:
            data = (root / row["path"]).read_bytes()
            assert (
                len(data) == row["bytes"]
                and hashlib.sha256(data).hexdigest() == row["sha256"]
            )
        demo = subprocess.run(
            [sys.executable, "scripts/show_progress.py"],
            cwd=root,
            text=True,
            capture_output=True,
            check=True,
        )
        assert (
            "0.560049" in demo.stdout
            and "0.458407" in demo.stdout
            and "30 epochs" in demo.stdout
        )
        code = "from streamlit.testing.v1 import AppTest; a=AppTest.from_file('app.py').run(timeout=60); assert len(a.exception)==0, [e.message for e in a.exception]; assert any(m.value=='0.5600' for m in a.metric); print('Portable dashboard passed')"
        subprocess.run(
            [sys.executable, "-c", code],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        tests = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"],
            cwd=root,
            text=True,
            capture_output=True,
            check=True,
        )
        assert "58 passed" in tests.stdout
    print(
        json.dumps(
            dict(
                zip_crc="passed",
                manifest_hashes_verified=len(manifest),
                offline_demo="passed",
                portable_dashboard="passed",
                portable_tests="58 passed",
                zip_sha256=hashlib.sha256(package.read_bytes()).hexdigest(),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
