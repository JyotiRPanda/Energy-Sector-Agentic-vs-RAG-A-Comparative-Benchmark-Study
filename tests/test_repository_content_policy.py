from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

# Hard policy: no image/media artifacts in this repository.
DISALLOWED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".svg",
    ".ico",
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
}

EXCLUDED_DIRS = {
    ".git",
    ".venv",
    ".pytest_cache",
    "__pycache__",
}


def test_no_disallowed_media_files_in_repo() -> None:
    violations = []

    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue

        if any(part in EXCLUDED_DIRS for part in path.parts):
            continue

        if path.suffix.lower() in DISALLOWED_EXTENSIONS:
            violations.append(path.relative_to(REPO_ROOT).as_posix())

    assert not violations, "Disallowed media files found: " + ", ".join(sorted(violations))