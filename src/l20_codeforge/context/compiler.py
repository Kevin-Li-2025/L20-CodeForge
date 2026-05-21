from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field


class ContextFile(BaseModel):
    path: str
    score: float
    chars: int
    reason: str
    content: str


class ContextPack(BaseModel):
    repo: str
    query: str
    budget_chars: int
    used_chars: int
    files: list[ContextFile] = Field(default_factory=list)


@dataclass(frozen=True)
class ContextCompiler:
    repo: Path
    budget_chars: int = 12000
    max_file_chars: int = 5000

    def compile(self, query: str) -> ContextPack:
        repo = self.repo.resolve()
        candidates = []
        for path in self._iter_files(repo):
            rel = path.relative_to(repo).as_posix()
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            score, reason = self._score(rel, text, query)
            if score <= 0:
                continue
            candidates.append((score, reason, rel, text[: self.max_file_chars]))

        candidates.sort(key=lambda item: (-item[0], item[2]))
        files: list[ContextFile] = []
        used = 0
        for score, reason, rel, text in candidates:
            if used >= self.budget_chars:
                break
            remaining = self.budget_chars - used
            snippet = text[:remaining]
            if not snippet:
                break
            used += len(snippet)
            files.append(
                ContextFile(
                    path=rel,
                    score=score,
                    chars=len(snippet),
                    reason=reason,
                    content=snippet,
                )
            )

        return ContextPack(
            repo=str(repo),
            query=query,
            budget_chars=self.budget_chars,
            used_chars=used,
            files=files,
        )

    def _iter_files(self, repo: Path) -> list[Path]:
        ignored_parts = {
            ".git",
            ".venv",
            "venv",
            "__pycache__",
            "node_modules",
            "artifacts",
            ".mypy_cache",
            ".pytest_cache",
        }
        allowed_suffixes = {
            ".py",
            ".toml",
            ".yaml",
            ".yml",
            ".json",
            ".md",
            ".txt",
            ".sh",
            ".js",
            ".ts",
            ".tsx",
            ".jsx",
        }
        files = []
        for path in repo.rglob("*"):
            if not path.is_file():
                continue
            if any(part in ignored_parts for part in path.parts):
                continue
            if path.suffix not in allowed_suffixes:
                continue
            files.append(path)
        return files

    def _score(self, rel: str, text: str, query: str) -> tuple[float, str]:
        query_terms = {term.lower() for term in query.replace("/", " ").replace("_", " ").split()}
        haystack = f"{rel}\n{text[:20000]}".lower()
        matches = sum(1 for term in query_terms if term and term in haystack)

        score = float(matches)
        reasons = []
        if matches:
            reasons.append(f"{matches} query term matches")
        if "test" in rel.lower():
            score += 1.5
            reasons.append("test file")
        if rel.endswith(("pyproject.toml", "package.json", "requirements.txt")):
            score += 1.0
            reasons.append("project config")
        if rel.lower().endswith(("readme.md", "contributing.md")):
            score += 0.5
            reasons.append("repo guide")

        return score, ", ".join(reasons) if reasons else "low relevance"

