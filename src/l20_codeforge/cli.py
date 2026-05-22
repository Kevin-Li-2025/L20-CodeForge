from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from l20_codeforge.agents.mini_swe import convert_mini_trajectory_file, export_mini_task_records
from l20_codeforge.context.compiler import ContextCompiler
from l20_codeforge.data.preferences import build_preference_pairs
from l20_codeforge.data.real_datasets import fetch_hf_real_dataset, list_real_dataset_specs
from l20_codeforge.data.real_sft import build_real_sft_jsonl
from l20_codeforge.data.report import write_trajectory_report
from l20_codeforge.data.sft import build_sft_jsonl
from l20_codeforge.data.smoke_tasks import write_smoke_tasks
from l20_codeforge.evals.eval_card import EvalCard
from l20_codeforge.evals.patch_eval import evaluate_patch, load_task
from l20_codeforge.evals.real_exec import evaluate_real_patch
from l20_codeforge.evals.sft_eval import evaluate_real_sft_model
from l20_codeforge.gpu.profile import L20Profile
from l20_codeforge.training.sft import train_real_sft
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


@app.command("list-real-sources")
def list_real_sources() -> None:
    """List real-world coding repair datasets supported by the registry."""
    console.print_json(data=[spec.model_dump() for spec in list_real_dataset_specs()])


@app.command("fetch-real-tasks")
def fetch_real_tasks(
    dataset: str,
    output: Path = Path("data/raw/real_tasks.jsonl"),
    split: str | None = None,
    limit: int | None = 100,
    streaming: bool = False,
) -> None:
    """Fetch real GitHub issue/PR repair records from a Hugging Face dataset."""
    try:
        report = fetch_hf_real_dataset(
            key=dataset,
            output_path=output,
            split=split,
            limit=limit,
            streaming=streaming,
        )
    except Exception as exc:
        console.print(f"[red]failed to fetch real dataset:[/red] {exc}")
        raise typer.Exit(1) from exc
    console.print_json(data=report.model_dump())


@app.command("build-real-sft")
def build_real_sft(
    real_tasks: Path,
    output: Path = Path("data/processed/real_sft.jsonl"),
    limit: int | None = None,
    min_patch_chars: int = 20,
) -> None:
    """Convert real issue/PR gold patches into chat SFT JSONL."""
    count = build_real_sft_jsonl(
        real_tasks_path=real_tasks,
        output_path=output,
        limit=limit,
        min_patch_chars=min_patch_chars,
    )
    console.print_json(data={"records": count, "output": str(output)})


@app.command("train-real-sft")
def train_real_sft_command(
    model: str,
    train_jsonl: Path,
    output_dir: Path = Path("artifacts/checkpoints/real-sft-smoke"),
    max_steps: int = 5,
    max_length: int = 2048,
    limit: int | None = 64,
    learning_rate: float = 2e-4,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    load_in_4bit: bool = False,
    bf16: bool = True,
) -> None:
    """Run LoRA/QLoRA SFT on real gold-patch chat data."""
    report = train_real_sft(
        model_name_or_path=model,
        train_jsonl=train_jsonl,
        output_dir=output_dir,
        max_steps=max_steps,
        max_length=max_length,
        limit=limit,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        load_in_4bit=load_in_4bit,
        bf16=bf16,
    )
    console.print_json(data=report)


@app.command("eval-real-sft")
def eval_real_sft_command(
    model: str,
    eval_jsonl: Path,
    output: Path = Path("artifacts/evals/real_sft_eval.json"),
    adapter_path: str | None = None,
    exclude_jsonl: Path | None = None,
    limit: int | None = 50,
    max_length: int = 4096,
    max_new_tokens: int = 384,
    generate_samples: int = 3,
    load_in_4bit: bool = True,
    bf16: bool = True,
) -> None:
    """Evaluate assistant-token NLL and sample generations on real SFT data."""
    report = evaluate_real_sft_model(
        model_name_or_path=model,
        eval_jsonl=eval_jsonl,
        output=output,
        adapter_path=adapter_path,
        exclude_jsonl=exclude_jsonl,
        limit=limit,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        generate_samples=generate_samples,
        load_in_4bit=load_in_4bit,
        bf16=bf16,
    )
    console.print_json(data={key: value for key, value in report.items() if key != "per_record"})


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


@app.command("generate-smoke-tasks")
def generate_smoke_tasks(
    output_dir: Path = Path("data/raw/smoke_tasks"),
    overwrite: bool = False,
) -> None:
    """Generate small executable repo-repair tasks."""
    task_files = write_smoke_tasks(output_dir=output_dir, overwrite=overwrite)
    for task_file in task_files:
        console.print(str(task_file))


@app.command("eval-patch")
def eval_patch(
    task_file: Path,
    patch_file: Path,
    output: Path = Path("artifacts/trajectories/patch_eval.jsonl"),
    keep_worktree: bool = False,
    run_hidden: bool = False,
    timeout_seconds: int = 120,
) -> None:
    """Apply a patch to an isolated task repo, run tests, and append a trajectory."""
    task = load_task(task_file)
    patch = patch_file.read_text(encoding="utf-8")
    result = evaluate_patch(
        task=task,
        patch=patch,
        keep_worktree=keep_worktree,
        run_hidden=run_hidden,
        timeout_seconds=timeout_seconds,
    )
    result.trajectory.write_jsonl(output)
    console.print_json(
        data={
            "task_id": task.task_id,
            "status": result.trajectory.status,
            "reward": result.trajectory.reward.model_dump(),
            "trajectory_output": str(output),
            "worktree": result.worktree if keep_worktree else None,
        }
    )


@app.command("eval-real-patch")
def eval_real_patch_command(
    real_tasks: Path,
    instance_id: str,
    patch_file: Path,
    output: Path = Path("artifacts/real_eval/reports/real_patch_eval.json"),
    repos_dir: Path = Path("artifacts/real_eval/repos"),
    repo_dir: Path | None = None,
    test_command: str | None = None,
    candidate_name: str = "candidate",
    timeout_seconds: int = 120,
    skip_fetch: bool = False,
    max_output_chars: int = 20000,
    keep_worktree_state: bool = False,
) -> None:
    """Apply a candidate patch to a real SWE task and run base/gold/candidate tests."""
    try:
        report = evaluate_real_patch(
            real_tasks_path=real_tasks,
            instance_id=instance_id,
            patch_file=patch_file,
            output=output,
            repos_dir=repos_dir,
            repo_dir=repo_dir,
            test_command=test_command,
            candidate_name=candidate_name,
            timeout_seconds=timeout_seconds,
            fetch_existing=not skip_fetch,
            max_output_chars=max_output_chars,
            keep_worktree_state=keep_worktree_state,
        )
    except Exception as exc:
        console.print(f"[red]failed to evaluate real patch:[/red] {exc}")
        raise typer.Exit(1) from exc

    console.print_json(
        data={
            "instance_id": report.instance_id,
            "verdict": report.verdict,
            "base_status": report.base_result.status,
            "gold_status": report.gold_result.status,
            "candidate_status": report.candidate_result.status,
            "notes": report.notes,
            "output": str(output),
            "worktree": report.worktree,
        }
    )


@app.command("build-sft")
def build_sft(
    trajectories: Path,
    output: Path = Path("data/processed/soft_verified_trajectories.jsonl"),
    min_reward: float = 1.0,
    include_partial: bool = False,
) -> None:
    """Convert verified trajectories into chat SFT JSONL."""
    count = build_sft_jsonl(
        trajectories_path=trajectories,
        output_path=output,
        min_reward=min_reward,
        include_partial=include_partial,
    )
    console.print_json(data={"records": count, "output": str(output)})


@app.command("smoke-loop")
def smoke_loop(
    task_dir: Path = Path("data/raw/smoke_tasks"),
    trajectories: Path = Path("artifacts/trajectories/smoke_reference.jsonl"),
    sft_output: Path = Path("data/processed/smoke_sft.jsonl"),
    report_output: Path = Path("artifacts/reports/smoke_reference_report.json"),
    run_hidden: bool = True,
    overwrite: bool = True,
) -> None:
    """Run the full local loop: generate tasks, evaluate reference patches, build SFT."""
    task_files = write_smoke_tasks(output_dir=task_dir, overwrite=overwrite)
    if overwrite and trajectories.exists():
        trajectories.unlink()
    success = 0
    for task_file in task_files:
        task = load_task(task_file)
        patch_path = Path(task.metadata["reference_patch"])
        result = evaluate_patch(
            task=task,
            patch=patch_path.read_text(encoding="utf-8"),
            run_hidden=run_hidden,
        )
        result.trajectory.write_jsonl(trajectories)
        if result.trajectory.status == "success":
            success += 1

    report = write_trajectory_report(trajectories, report_output)
    records = build_sft_jsonl(
        trajectories_path=trajectories,
        output_path=sft_output,
        min_reward=1.0,
    )
    console.print_json(
        data={
            "tasks": len(task_files),
            "success": success,
            "trajectories": str(trajectories),
            "report": str(report_output),
            "status_counts": report.status_counts,
            "sft_records": records,
            "sft_output": str(sft_output),
        }
    )


@app.command("report-trajectories")
def report_trajectories(
    trajectories: Path,
    output: Path = Path("artifacts/reports/trajectory_report.json"),
) -> None:
    """Summarize trajectory status, rewards, and tags."""
    report = write_trajectory_report(trajectories, output)
    console.print_json(data={"output": str(output), **report.model_dump()})


@app.command("export-mini-tasks")
def export_mini_tasks(
    task_dir: Path = Path("data/raw/smoke_tasks"),
    output: Path = Path("artifacts/mini_swe/mini_task_records.jsonl"),
    mini_output_dir: Path = Path("artifacts/mini_swe/trajectories"),
    overwrite_tasks: bool = False,
) -> None:
    """Write mini-SWE-agent prompts and suggested commands for generated tasks."""
    task_files = write_smoke_tasks(output_dir=task_dir, overwrite=overwrite_tasks)
    count = export_mini_task_records(
        task_files=task_files,
        output_path=output,
        output_dir=mini_output_dir,
    )
    console.print_json(data={"records": count, "output": str(output)})


@app.command("convert-mini")
def convert_mini(
    task_file: Path,
    mini_trajectory: Path,
    output: Path = Path("artifacts/trajectories/mini_swe_converted.jsonl"),
    run_hidden: bool = True,
    timeout_seconds: int = 120,
) -> None:
    """Convert a mini-SWE-agent trajectory into L20 CodeForge trajectory JSONL."""
    result = convert_mini_trajectory_file(
        task_file=task_file,
        mini_trajectory_file=mini_trajectory,
        run_hidden=run_hidden,
        timeout_seconds=timeout_seconds,
    )
    result.trajectory.write_jsonl(output)
    console.print_json(
        data={
            "task_id": result.trajectory.task.task_id,
            "status": result.trajectory.status,
            "patch_found": result.patch_found,
            "mini_exit_status": result.mini_exit_status,
            "reward": result.trajectory.reward.model_dump(),
            "output": str(output),
        }
    )


@app.command("build-dpo")
def build_dpo(
    trajectories: Path,
    output: Path = Path("data/processed/preference_pairs.jsonl"),
    min_reward_gap: float = 0.25,
) -> None:
    """Build task-level chosen/rejected patch pairs from trajectory JSONL."""
    count = build_preference_pairs(
        trajectories_path=trajectories,
        output_path=output,
        min_reward_gap=min_reward_gap,
    )
    console.print_json(data={"pairs": count, "output": str(output)})


if __name__ == "__main__":
    app()
