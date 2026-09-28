"""Dependency direction between the retrieval layer and the framework (plan §3.1, §11)."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "snn_hft.data.lob.schema"


def imported_modules(path: Path, source: str | None = None) -> set[str]:
    """Absolute module names imported by a file (relative imports resolved)."""
    tree = ast.parse(path.read_text() if source is None else source, filename=str(path))
    package = ".".join(path.relative_to(ROOT).with_suffix("").parts[:-1])
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".")[: len(package.split(".")) - node.level + 1]
                mod = ".".join(base + ([node.module] if node.module else []))
            else:
                mod = node.module or ""
            out.add(mod)
            out.update(f"{mod}.{a.name}" for a in node.names)  # `from pkg import module`
    return out


def py_files(*parts: str) -> list[Path]:
    files = sorted((ROOT.joinpath(*parts)).rglob("*.py"))
    assert files, parts
    return files


def test_framework_never_imports_retrieval_or_scripts():
    for f in py_files("snn_hft"):
        bad = {m for m in imported_modules(f) if m.split(".")[0] in ("retrieval", "scripts")}
        assert not bad, f"{f.relative_to(ROOT)} imports {bad}"


def test_retrieval_imports_only_the_schema_from_the_framework():
    for f in py_files("retrieval"):
        framework = {m for m in imported_modules(f) if m.split(".")[0] == "snn_hft"}
        bad = {m for m in framework if m != SCHEMA and not m.startswith(SCHEMA + ".")}
        assert not bad, f"{f.relative_to(ROOT)} imports {bad}"


def test_schema_depends_on_no_other_framework_module():
    for f in [ROOT / "snn_hft/data/lob/schema.py", ROOT / "snn_hft/data/lob/__init__.py"]:
        framework = {m for m in imported_modules(f) if m.split(".")[0] == "snn_hft"}
        assert not framework, f"{f.relative_to(ROOT)} imports {framework}"


def test_import_scanner_sees_both_import_forms():
    fake = ROOT / "snn_hft" / "signals" / "lob" / "probe.py"
    mods = imported_modules(fake, "import retrieval.book\nfrom ...data.lob import schema\n")
    assert "retrieval.book" in mods and SCHEMA in mods
