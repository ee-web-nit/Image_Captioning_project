from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:

    mode: str

    epochs: int

    batch_size: int

    shuffle_buffer: int

    learning_rate: float

    checkpoint_dir: Path

    model_dir: Path

    tensorboard_dir: Path

    history_dir: Path