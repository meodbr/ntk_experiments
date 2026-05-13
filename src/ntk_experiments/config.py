from dataclasses import dataclass

@dataclass
class Config:
    INPUT_DIM: int = 1
    OUTPUT_DIM: int = 1
    BETA: float = 0.1
    DATASET: str = 'synthetic'  # Options: 'synthetic', 'mnist'
    SEED: int = 52

config = Config()