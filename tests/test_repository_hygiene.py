from pathlib import Path
import re


TEXT_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".toml", ".txt", ".csv", ".ipynb", ".gitignore", ".python-version"}
FORBIDDEN_PATTERNS = [
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"(?i)(api[_-]?key|secret[_-]?key|access[_-]?token)\s*[:=]\s*['\"][^'\"]+"),
    re.compile(r"/home/[^/\s]+/"),
    re.compile(r"[A-Za-z]:\\Users\\[^\\\s]+\\"),
]


def test_no_obvious_secrets_or_private_absolute_paths() -> None:
    root = Path(".")
    offenders: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file() or ".pytest_cache" in path.parts or "__pycache__" in path.parts:
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        if path.name in {".gitignore", ".python-version"} or path.suffix in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8", errors="ignore")
            for pattern in FORBIDDEN_PATTERNS:
                if pattern.search(text):
                    offenders.append(f"{path}: {pattern.pattern}")
    assert not offenders, offenders


def test_readme_excludes_internal_registry_ids() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    for token in ("DATA-", "SPLIT-", "EXP-"):
        assert token not in readme
