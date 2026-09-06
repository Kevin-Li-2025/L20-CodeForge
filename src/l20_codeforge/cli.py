from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from l20_codeforge.agents.mini_swe import convert_mini_trajectory_file, export_mini_task_records
from l20_codeforge.context.compiler import ContextCompiler
from l20_codeforge.data.code_bench_sft import build_mbpp_sft_jsonl
from l20_codeforge.data.code_rlvr import materialize_rstar_code_rlvr
from l20_codeforge.data.preferences import build_preference_pairs
from l20_codeforge.data.real_datasets import fetch_hf_real_dataset, list_real_dataset_specs
from l20_codeforge.data.real_sft import build_real_sft_jsonl
from l20_codeforge.data.report import write_trajectory_report
from l20_codeforge.data.retention import (
    build_lcb_verified_trajectory_sft,
    compose_retention_sft_mixture,
    materialize_mbpp_replay,
)
from l20_codeforge.data.sft import build_sft_jsonl
from l20_codeforge.data.smoke_tasks import write_smoke_tasks
from l20_codeforge.evals.code_rlvr import (
    build_verified_sft_from_rollouts,
    generate_code_rollouts,
    merge_code_rollouts,
    select_mixed_reward_tasks,
)
from l20_codeforge.evals.eval_card import EvalCard
from l20_codeforge.evals.evalplus_runner import (
    generate_evalplus_repairs,
    generate_evalplus_samples,
    run_evalplus_official,
    select_evalplus_by_base_tests,
    select_evalplus_by_prompt_doctests,
    select_evalplus_by_public_consensus,
)
from l20_codeforge.evals.function_retention import generate_function_retention_rollouts
from l20_codeforge.evals.patch_eval import evaluate_patch, load_task
from l20_codeforge.evals.real_exec import evaluate_real_patch
from l20_codeforge.evals.sft_eval import evaluate_real_sft_model
from l20_codeforge.evals.verifier_audit import VerifierAuditGates, audit_verifier_dataset
from l20_codeforge.gpu.profile import L20Profile
from l20_codeforge.rewards.code_execution import CodeExecutionConfig
from l20_codeforge.training.grpo import train_code_grpo
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


@app.command("build-mbpp-sft")
def build_mbpp_sft(
    output: Path = Path("data/processed/code_bench/mbpp_train_sft.jsonl"),
    split: str = "train",
    limit: int | None = None,
    exclude_evalplus_mbpp: bool = True,
) -> None:
    """Build non-EvalPlus MBPP split SFT data for public benchmark post-training."""
    try:
        count = build_mbpp_sft_jsonl(
            output_path=output,
            split=split,
            limit=limit,
            exclude_evalplus_mbpp=exclude_evalplus_mbpp,
        )
    except Exception as exc:
        console.print(f"[red]failed to build MBPP SFT data:[/red] {exc}")
        raise typer.Exit(1) from exc
    console.print_json(
        data={
            "records": count,
            "split": split,
            "output": str(output),
            "exclude_evalplus_mbpp": exclude_evalplus_mbpp,
        }
    )


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
    completion_only_loss: bool = False,
    load_in_4bit: bool = False,
    bf16: bool = True,
    seed: int = 42,
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
        completion_only_loss=completion_only_loss,
        load_in_4bit=load_in_4bit,
        bf16=bf16,
        seed=seed,
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


@app.command("audit-code-verifier")
def audit_code_verifier_command(
    input_jsonl: Path,
    output: Path = Path("artifacts/verifier/audit.json"),
    timeout_seconds: float = 2.0,
    compile_timeout_seconds: float = 5.0,
    memory_limit_mb: int = 512,
    max_output_chars: int = 100000,
    comparison: str = "tokens",
    min_reference_solutions: int = 1,
    min_reference_accept_rate: float = 1.0,
    min_faulty_kill_rate: float = 0.7,
    max_false_positive_rate: float = 0.05,
    max_false_negative_rate: float = 0.05,
    limit: int | None = None,
    fail_on_gates: bool = False,
) -> None:
    """Audit labeled algorithmic-code tests before using them as RL rewards."""

    if comparison not in {"tokens", "exact"}:
        console.print("[red]comparison must be 'tokens' or 'exact'[/red]")
        raise typer.Exit(2)
    report = audit_verifier_dataset(
        input_path=input_jsonl,
        output_path=output,
        execution_config=CodeExecutionConfig(
            timeout_seconds=timeout_seconds,
            compile_timeout_seconds=compile_timeout_seconds,
            memory_limit_mb=memory_limit_mb,
            max_output_chars=max_output_chars,
            comparison=comparison,
        ),
        gates=VerifierAuditGates(
            min_reference_accept_rate=min_reference_accept_rate,
            min_faulty_kill_rate=min_faulty_kill_rate,
            max_false_positive_rate=max_false_positive_rate,
            max_false_negative_rate=max_false_negative_rate,
        ),
        min_reference_solutions=min_reference_solutions,
        limit=limit,
    )
    console.print_json(
        data={
            "status": report["status"],
            "summary": report["summary"],
            "gates": report["gates"],
            "output": str(output),
        }
    )
    if fail_on_gates and report["status"] != "PASS":
        raise typer.Exit(2)


@app.command("generate-evalplus")
def generate_evalplus_command(
    model: str,
    dataset: str = "humaneval",
    output: Path = Path("artifacts/evalplus/samples.jsonl"),
    adapter_path: str | None = None,
    n_samples: int = 1,
    limit: int | None = None,
    id_start: int | None = None,
    id_end: int | None = None,
    shard_index: int | None = None,
    shard_count: int | None = None,
    task_ids: str | None = None,
    prompt_style: str = "default",
    temperature: float = 0.0,
    top_p: float = 0.95,
    max_new_tokens: int = 512,
    sample_batch_size: int = 1,
    load_in_4bit: bool = True,
    bf16: bool = True,
    seed: int = 42,
    overwrite: bool = False,
) -> None:
    """Generate official EvalPlus-compatible JSONL samples with 4-bit model loading."""
    try:
        report = generate_evalplus_samples(
            model_name_or_path=model,
            dataset=dataset,
            output=output,
            adapter_path=adapter_path,
            n_samples=n_samples,
            limit=limit,
            id_start=id_start,
            id_end=id_end,
            shard_index=shard_index,
            shard_count=shard_count,
            task_ids=parse_csv(task_ids),
            prompt_style=prompt_style,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            sample_batch_size=sample_batch_size,
            load_in_4bit=load_in_4bit,
            bf16=bf16,
            seed=seed,
            overwrite=overwrite,
        )
    except Exception as exc:
        console.print(f"[red]failed to generate EvalPlus samples:[/red] {exc}")
        raise typer.Exit(1) from exc
    console.print_json(data=report.model_dump())


def parse_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


@app.command("repair-evalplus")
def repair_evalplus_command(
    model: str,
    samples: Path,
    eval_results: Path,
    dataset: str = "humaneval",
    output: Path = Path("artifacts/evalplus/repairs.samples.jsonl"),
    n_repairs: int = 10,
    task_ids: str | None = None,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_new_tokens: int = 512,
    sample_batch_size: int = 1,
    load_in_4bit: bool = True,
    bf16: bool = True,
    seed: int = 42,
    overwrite: bool = False,
) -> None:
    """Generate repair candidates from failed EvalPlus samples and base-test feedback."""
    try:
        report = generate_evalplus_repairs(
            model_name_or_path=model,
            dataset=dataset,
            samples=samples,
            eval_results=eval_results,
            output=output,
            n_repairs=n_repairs,
            task_ids=parse_csv(task_ids),
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            sample_batch_size=sample_batch_size,
            load_in_4bit=load_in_4bit,
            bf16=bf16,
            seed=seed,
            overwrite=overwrite,
        )
    except Exception as exc:
        console.print(f"[red]failed to generate EvalPlus repairs:[/red] {exc}")
        raise typer.Exit(1) from exc
    console.print_json(data=report.model_dump())


@app.command("eval-evalplus")
def eval_evalplus_command(
    dataset: str,
    samples: Path,
    output: Path = Path("artifacts/evalplus/evalplus_report.json"),
    base_only: bool = False,
    parallel: int | None = None,
    test_details: bool = True,
    mini: bool = False,
    i_just_wanna_run: bool = False,
) -> None:
    """Run official evalplus.evaluate on generated samples and save stdout/stderr."""
    report = run_evalplus_official(
        dataset=dataset,
        samples=samples,
        output=output,
        base_only=base_only,
        parallel=parallel,
        test_details=test_details,
        mini=mini,
        i_just_wanna_run=i_just_wanna_run,
    )
    console.print_json(data=report.model_dump())
    if report.exit_code != 0:
        raise typer.Exit(report.exit_code)


@app.command("select-evalplus")
def select_evalplus_command(
    samples: Path,
    eval_results: Path,
    output: Path = Path("artifacts/evalplus/selected.samples.jsonl"),
    tie_breaker: str = "first",
) -> None:
    """Select one sample per EvalPlus task using only base-test pass/fail results."""
    report = select_evalplus_by_base_tests(
        samples=samples,
        eval_results=eval_results,
        output=output,
        tie_breaker=tie_breaker,
    )
    console.print_json(data=report.model_dump())


@app.command("select-evalplus-prompt")
def select_evalplus_prompt_command(
    samples: Path,
    output: Path = Path("artifacts/evalplus/prompt-selected.samples.jsonl"),
    dataset: str = "humaneval",
    timeout_seconds: float = 2.0,
) -> None:
    """Select one sample per EvalPlus task using only prompt doctest examples."""
    report = select_evalplus_by_prompt_doctests(
        samples=samples,
        output=output,
        dataset=dataset,
        timeout_seconds=timeout_seconds,
    )
    console.print_json(data=report.model_dump())


@app.command("select-evalplus-consensus")
def select_evalplus_consensus_command(
    samples: Path,
    eval_results: Path,
    output: Path = Path("artifacts/evalplus/consensus-selected.samples.jsonl"),
    dataset: str = "humaneval",
    max_synthetic_inputs: int = 32,
    timeout_seconds: float = 2.0,
    tie_breaker: str = "longest",
) -> None:
    """Select with base-test filtering plus public-input consensus reranking."""
    report = select_evalplus_by_public_consensus(
        samples=samples,
        eval_results=eval_results,
        output=output,
        dataset=dataset,
        max_synthetic_inputs=max_synthetic_inputs,
        timeout_seconds=timeout_seconds,
        tie_breaker=tie_breaker,
    )
    console.print_json(data=report.model_dump())


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


@app.command("build-rstar-code-rlvr")
def build_rstar_code_rlvr_command(
    output_dir: Path,
    train_tasks: int = 800,
    dev_tasks: int = 200,
    retention_tasks: int = 0,
    final_tasks: int = 0,
    seed: int = 20260829,
    min_tests: int = 8,
    max_tests: int = 24,
    max_case_input_chars: int = 131_072,
    max_case_output_chars: int = 65_536,
    cache_dir: Path | None = None,
    use_rows_api: bool = False,
    rows_api_limit: int = 4000,
    rows_api_batch_size: int = 20,
) -> None:
    """Freeze deterministic rStar executable train/dev data from converted parquet."""
    report = materialize_rstar_code_rlvr(
        output_dir,
        train_tasks=train_tasks,
        dev_tasks=dev_tasks,
        retention_tasks=retention_tasks,
        final_tasks=final_tasks,
        seed=seed,
        min_tests=min_tests,
        max_tests=max_tests,
        max_case_input_chars=max_case_input_chars,
        max_case_output_chars=max_case_output_chars,
        cache_dir=cache_dir,
        use_rows_api=use_rows_api,
        rows_api_limit=rows_api_limit,
        rows_api_batch_size=rows_api_batch_size,
    )
    console.print_json(data=report)


@app.command("build-mbpp-retention-replay")
def build_mbpp_retention_replay_command(output_dir: Path) -> None:
    """Freeze official MBPP train replay and validation retention tasks."""
    console.print_json(data=materialize_mbpp_replay(output_dir))


@app.command("build-lcb-trajectory-sft")
def build_lcb_trajectory_sft_command(
    eval_all_json: Path,
    output: Path,
    max_contest_date: str,
    min_contest_date: str | None = None,
    max_records: int = 138,
    seed: int = 20260830,
) -> None:
    """Build training-only SFT from historical full-harness-passing L20 outputs."""
    console.print_json(
        data=build_lcb_verified_trajectory_sft(
            eval_all_json,
            output,
            max_contest_date=max_contest_date,
            min_contest_date=min_contest_date,
            max_records=max_records,
            seed=seed,
        )
    )


@app.command("compose-retention-sft")
def compose_retention_sft_command(
    target_sft: Path,
    lcb_replay_sft: Path,
    mbpp_replay_sft: Path,
    output: Path,
    target_records: int = 414,
    lcb_records: int = 138,
    mbpp_records: int = 138,
    seed: int = 20260830,
) -> None:
    """Compose a deterministic target/trajectory/function replay SFT mixture."""
    console.print_json(
        data=compose_retention_sft_mixture(
            target_sft,
            lcb_replay_sft,
            mbpp_replay_sft,
            output,
            target_records=target_records,
            lcb_records=lcb_records,
            mbpp_records=mbpp_records,
            seed=seed,
        )
    )


@app.command("generate-code-rollouts")
def generate_code_rollouts_command(
    model: str,
    tasks_jsonl: Path,
    output: Path,
    adapter_path: str | None = None,
    n_samples: int = 1,
    temperature: float = 0.0,
    top_p: float = 0.95,
    max_prompt_length: int = 6144,
    max_new_tokens: int = 1536,
    batch_size: int = 2,
    load_in_4bit: bool = True,
    bf16: bool = True,
    seed: int = 42,
    shard_index: int = 0,
    shard_count: int = 1,
    timeout_seconds: float = 2.0,
    execution_workers: int = 1,
    overwrite: bool = False,
) -> None:
    """Generate and execute code rollouts for one deterministic GPU shard."""
    report = generate_code_rollouts(
        model,
        tasks_jsonl,
        output,
        adapter_path=adapter_path,
        n_samples=n_samples,
        temperature=temperature,
        top_p=top_p,
        max_prompt_length=max_prompt_length,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        load_in_4bit=load_in_4bit,
        bf16=bf16,
        seed=seed,
        shard_index=shard_index,
        shard_count=shard_count,
        timeout_seconds=timeout_seconds,
        execution_workers=execution_workers,
        overwrite=overwrite,
    )
    console.print_json(data=report)


@app.command("merge-code-rollouts")
def merge_code_rollouts_command(
    output: Path,
    inputs: list[Path],
    expected_tasks: int | None = None,
) -> None:
    """Merge disjoint rollout shards and write an executable-accuracy report."""
    console.print_json(data=merge_code_rollouts(inputs, output, expected_tasks=expected_tasks))


@app.command("generate-function-retention-rollouts")
def generate_function_retention_rollouts_command(
    model: str,
    tasks_jsonl: Path,
    output: Path,
    adapter_path: str | None = None,
    n_samples: int = 1,
    temperature: float = 0.0,
    top_p: float = 0.95,
    max_prompt_length: int = 4096,
    max_new_tokens: int = 1024,
    batch_size: int = 2,
    load_in_4bit: bool = True,
    bf16: bool = True,
    seed: int = 42,
    shard_index: int = 0,
    shard_count: int = 1,
    timeout_seconds: float = 4.0,
    overwrite: bool = False,
) -> None:
    """Generate and execute official MBPP-validation retention rollouts."""
    report = generate_function_retention_rollouts(
        model,
        tasks_jsonl,
        output,
        adapter_path=adapter_path,
        n_samples=n_samples,
        temperature=temperature,
        top_p=top_p,
        max_prompt_length=max_prompt_length,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        load_in_4bit=load_in_4bit,
        bf16=bf16,
        seed=seed,
        shard_index=shard_index,
        shard_count=shard_count,
        timeout_seconds=timeout_seconds,
        overwrite=overwrite,
    )
    console.print_json(data=report)


@app.command("build-verified-code-sft")
def build_verified_code_sft_command(
    rollouts_jsonl: Path,
    output: Path,
    min_distinct_passing: int = 1,
    max_records: int | None = None,
) -> None:
    """Build SFT chat records only from all-tests-passed base rollouts."""
    console.print_json(
        data=build_verified_sft_from_rollouts(
            rollouts_jsonl,
            output,
            min_distinct_passing=min_distinct_passing,
            max_records=max_records,
        )
    )


@app.command("select-mixed-code-rlvr")
def select_mixed_code_rlvr_command(
    tasks_jsonl: Path,
    rollouts_jsonl: Path,
    output: Path,
    max_tasks: int | None = None,
) -> None:
    """Select RL tasks with both passing and failing base rollouts."""
    console.print_json(
        data=select_mixed_reward_tasks(
            tasks_jsonl,
            rollouts_jsonl,
            output,
            max_tasks=max_tasks,
        )
    )


@app.command("train-code-grpo")
def train_code_grpo_command(
    model: str,
    train_jsonl: Path,
    output_dir: Path,
    adapter_path: str | None = None,
    replay_jsonl: Path | None = None,
    replay_loss_weight: float = 0.0,
    replay_max_length: int = 3072,
    max_steps: int = 100,
    max_completion_length: int = 1024,
    limit: int | None = None,
    learning_rate: float = 1e-6,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    num_generations: int = 4,
    temperature: float = 0.7,
    top_p: float = 0.95,
    beta: float = 0.0,
    loss_type: str = "dr_grpo",
    reward_type: str = "dense",
    timeout_seconds: float = 2.0,
    execution_workers: int = 6,
    load_in_4bit: bool = True,
    bf16: bool = True,
    seed: int = 42,
) -> None:
    """Train QLoRA GRPO with frozen stdin/stdout execution rewards."""
    console.print_json(
        data=train_code_grpo(
            model,
            train_jsonl,
            output_dir,
            adapter_path=adapter_path,
            replay_jsonl=replay_jsonl,
            replay_loss_weight=replay_loss_weight,
            replay_max_length=replay_max_length,
            max_steps=max_steps,
            max_completion_length=max_completion_length,
            limit=limit,
            learning_rate=learning_rate,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_generations=num_generations,
            temperature=temperature,
            top_p=top_p,
            beta=beta,
            loss_type=loss_type,
            reward_type=reward_type,
            timeout_seconds=timeout_seconds,
            execution_workers=execution_workers,
            load_in_4bit=load_in_4bit,
            bf16=bf16,
            seed=seed,
        )
    )


if __name__ == "__main__":
    app()
