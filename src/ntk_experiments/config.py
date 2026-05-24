from dataclasses import dataclass

@dataclass
class Config:
    INPUT_DIM: int = 2
    OUTPUT_DIM: int = 1
    BETA: float = 0.1
    DATASET: str = 'synthetic'  # Options: 'synthetic', 'mnist'
    SEED: int = 42

config = Config()