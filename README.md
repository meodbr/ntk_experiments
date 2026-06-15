# Experiments with NTK

Small repo where I use Neural Tangent Kernels to study infinite-width behavior of CNN and MLP

The paper : [preprint.pdf](preprint.pdf)

## Usage

```
uv sync --extra ["cpu", "cu126"]
uv run -m ntk_experiments.main
```