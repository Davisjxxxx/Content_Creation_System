from __future__ import annotations

from pathlib import Path


class ContextLoader:
    def __init__(self, package_root: Path | None = None):
        self.package_root = package_root or Path(__file__).resolve().parent

    def read(self, relative_path: str) -> str:
        path = (self.package_root / relative_path).resolve()
        root = self.package_root.resolve()
        if root not in path.parents and path != root:
            raise ValueError("context path escapes package root")
        return path.read_text(encoding="utf-8")

    def bundle(self, paths: tuple[str, ...]) -> str:
        sections = []
        for relative_path in paths:
            sections.append(f"\n# CONTEXT: {relative_path}\n{self.read('context/' + relative_path)}")
        return "\n".join(sections)
