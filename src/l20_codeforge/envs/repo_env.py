from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CommandResult:
    command: str
    exit_code: int
    stdout: str
    stderr: str
    elapsed_timeout: bool = False


@dataclass(frozen=True)
class RepoEnv:
    repo: Path
    timeout_seconds: int = 120

    def run(self, command: str, cwd: Path | None = None) -> CommandResult:
        proc_cwd = cwd or self.repo
        try:
            completed = subprocess.run(
                command,
                cwd=proc_cwd,
                shell=True,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            return CommandResult(
                command=command,
                exit_code=completed.returncode,
                stdout=completed.stdout[-20000:],
                stderr=completed.stderr[-20000:],
            )
        except subprocess.TimeoutExpired as exc:
            return CommandResult(
                command=command,
                exit_code=124,
                stdout=(exc.stdout or "")[-20000:] if isinstance(exc.stdout, str) else "",
                stderr=(exc.stderr or "")[-20000:] if isinstance(exc.stderr, str) else "",
                elapsed_timeout=True,
            )

    def isolated_copy(self) -> Path:
        tmp = Path(tempfile.mkdtemp(prefix="l20-codeforge-"))
        dest = tmp / self.repo.name
        ignore = shutil.ignore_patterns(".git", ".venv", "venv", "__pycache__", "artifacts")
        shutil.copytree(self.repo, dest, ignore=ignore)
        return dest

