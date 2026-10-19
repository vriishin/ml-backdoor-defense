# ML Backdoor Defense

Research implementation exploring training-time backdoor defenses in neural networks, including Anti-Backdoor Learning (ABL) and Adversarial Neuron Pruning (ANP).

This is a **collaborative research codebase**, shared here as a technical portfolio and reproducibility reference. A companion paper and quantitative evaluation can be added later.

## Core components

- `backdoor_isolation.py` — isolate potentially poisoned training samples.
- `backdoor_unlearning.py` — fine-tuning and unlearning stages.
- `mask_optimization.py` — neuron mask optimization.
- `neuron_pruning.py` — neuron-level pruning.
- `data_loader.py`, `config.py` — experiment configuration and data handling.
- `models/` — architectures and normalization layers.
- `create_backdoor_data/`, `trigger/` — research data-generation and trigger assets.

## Research workflows

The experiments evaluate different sequences of sample isolation, fine-tuning, neuron-mask optimization, pruning, and unlearning to study trade-offs between attack success rate and clean model accuracy.

For the published research paper and reproducible benchmarks, see the project materials when they are added.

## Status

Experimental code from a collaborative research project; not a validated production security tool. Dataset/checkpoint paths and library versions may need adjustment before experiments can be reproduced.

## Experiment commands

See [detailed command sequences](docs/experiment-commands.md) for the baseline ABL workflow and the ABL/ANP variations used in the project. These commands require the original experimental datasets and checkpoint configuration.
