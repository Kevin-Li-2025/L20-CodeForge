from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from l20_codeforge.context.compiler import ContextCompiler
from l20_codeforge.evals.eval_card import EvalCard
from l20_codeforge.gpu.profile import L20Profile
from l20_codeforge.utils.paths import ensure_project_dirs

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command()
def profile() -> None:
    """Print the static L20 memory policy."""
    console.print_json(data=L20Profile.default().model_dump())


@app.command("init-dirs")
def init_dirs(root: Path = Path(".")) -> None:
    """Create standard local data and artifact directories."""
    created = ensure_project_dirs(root)
    for path in created:
        console.print(str(path))


@app.command("pack-context")
def pack_context(
    repo: Path,
    query: str,
    budget_chars: int = 12000,
    output: Path | None = None,
) -> None:
    """Create a token-budget proxy context pack for a repository."""
    compiler = ContextCompiler(repo=repo, budget_chars=budget_chars)
    pack = compiler.compile(query=query)
    payload = pack.model_dump()
    if output:
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"wrote {output}")
    else:
        console.print_json(data=payload)


@app.command("eval-card")
def eval_card(
    name: str,
    status: str,
    artifact_dir: Path = Path("artifacts/eval_cards"),
) -> None:
    """Write a small eval card for a completed smoke or experiment run."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    card = EvalCard(name=name, status=status)
    out = artifact_dir / f"{name}.json"
    out.write_text(card.model_dump_json(indent=2), encoding="utf-8")
    console.print(f"wrote {out}")


if __name__ == "__main__":
    app()

