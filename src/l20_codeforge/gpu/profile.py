from __future__ import annotations

from pydantic import BaseModel


class TrainingPlan(BaseModel):
    model_size: str
    method: str
    max_seq_length: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    notes: str


class L20Profile(BaseModel):
    name: str
    vram_gib: int
    preferred_dtype: str
    system_ram_warning: str
    plans: list[TrainingPlan]

    @classmethod
    def default(cls) -> "L20Profile":
        return cls(
            name="NVIDIA L20",
            vram_gib=45,
            preferred_dtype="bf16",
            system_ram_warning=(
                "The observed host has about 15 GiB RAM; keep dataset workers low and avoid "
                "large in-memory preprocessing."
            ),
            plans=[
                TrainingPlan(
                    model_size="7B",
                    method="QLoRA SFT",
                    max_seq_length=8192,
                    micro_batch_size=1,
                    gradient_accumulation_steps=16,
                    notes="Primary path for trajectory SFT.",
                ),
                TrainingPlan(
                    model_size="7B",
                    method="GRPO guided retry",
                    max_seq_length=6144,
                    micro_batch_size=1,
                    gradient_accumulation_steps=8,
                    notes="Start with num_generations=4 and cheap executable rewards.",
                ),
                TrainingPlan(
                    model_size="14B",
                    method="QLoRA SFT",
                    max_seq_length=4096,
                    micro_batch_size=1,
                    gradient_accumulation_steps=16,
                    notes="Only after 7B pipeline is stable.",
                ),
            ],
        )

