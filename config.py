from dataclasses import dataclass


@dataclass
class ModelConfig:
    in_features: int = 768
    out_features: int = 768
    lora_rank: int = 8
    volterra_rank: int = 8
    lora_alpha: float = 1.0
    volterra_alpha: float = 1.0
    dropout: float = 0.0
    freeze_base: bool = True


@dataclass
class ExperimentConfig:
    batch_size: int = 2
    num_tokens: int = 16
    seed: int = 42
