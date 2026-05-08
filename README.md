# VoRA

VoRA is a research-oriented PyTorch scaffold for PEFT modules that extend LoRA with efficient low-rank nonlinear Volterra adaptation.

Starting point:

```text
y = W0 x + B_l A_l x
```

VoRA-v1 adds a quadratic low-rank branch:

```text
y = W0 x + B_l A_l x + B_v(A_v x \odot A_v x)
```

The codebase is intentionally simple at first so later variants can be added cleanly:

- `models/lora_linear.py`: LoRA baseline.
- `models/vora_linear.py`: VoRA linear layer wrapper.
- `models/volterra.py`: reusable Volterra branches.
- `models/token_interaction.py`: token-neighborhood interaction modules.
- `models/routing.py`: input-conditioned expert routing.

## Quick Check

```bash
python test.py
```

This compares forward shapes and trainable parameter counts for `LoRALinear` and `VoRALinear` on random input.

## Project Layout

```text
VoRA/
├── config.py
├── train.py
├── test.py
├── models/
│   ├── lora_linear.py
│   ├── vora_linear.py
│   ├── volterra.py
│   ├── routing.py
│   ├── token_interaction.py
│   ├── vit/
│   └── restoration/
├── losses/
├── datasets/
├── engine/
├── utils/
├── experiments/
├── checkpoints/
├── outputs/
└── papers/
```

## Roadmap

1. LoRA baseline.
2. VoRA-v1 with efficient quadratic Volterra branch.
3. Token interaction-aware VoRA.
4. Dynamic routing with multiple Volterra experts.
5. ViT and restoration backbone integration.
