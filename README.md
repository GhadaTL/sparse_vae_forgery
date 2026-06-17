# Internship CRNS — Anomaly Detection Project

This repository contains code for an anomaly detection / adaptive control internship project. It includes training and evaluation pipelines, model definitions, loss functions, utilities for adaptive controllers, and scripts to run experiments.

## Repository structure

- [train.py](train.py) — training entrypoint
- [evaluate.py](evaluate.py) — evaluation / scoring entrypoint
- [configs/default.yaml](configs/default.yaml) — default configuration
- [requirements.txt](requirements.txt) — Python dependencies
- [checkpoints/](checkpoints/) — saved model checkpoints (`best_model.pth`, `last_model.pth`)
- [data/](data/) — dataset loader(s); see [data/dataset.py](data/dataset.py)
- [models/](models/) — model implementations (e.g. `full_model.py`, `multiscale_decoder.py`, `projection_head.py`, `sparse_latent.py`)
- [losses/](losses/) — loss implementations (see [losses/total_loss.py](losses/total_loss.py))
- [evaluation/](evaluation/) — evaluation utilities and metrics (see [evaluation/anomaly_scorer.py](evaluation/anomaly_scorer.py), [evaluation/metrics.py](evaluation/metrics.py), [evaluation/heatmap.py](evaluation/heatmap.py))
- [adaptive/](adaptive/) — adaptive controller utilities (e.g. `scheduler.py`, `logger.py`)
- [utils/](utils/) — helper modules (e.g. `beta_controller.py`, `k_controller.py`)
- [outputs/](outputs/) — example outputs and intermediate artifacts
- [results/README.md](results/README.md) — results notes

## Requirements

- Python 3.8+ recommended
- Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick start

1. Prepare your dataset and update `configs/default.yaml` if needed.
2. Train a model:

```bash
python train.py --config configs/default.yaml
```

Checkpoints will be written to the `checkpoints/` directory.

3. Run evaluation / anomaly scoring:

```bash
python evaluate.py --config configs/default.yaml --checkpoint checkpoints/last_model.pth
```

The evaluation pipeline uses the utilities in [evaluation/](evaluation/) to compute anomaly scores, heatmaps, and metrics.

## Configuration

Most runtime options are controlled by `configs/default.yaml`. Edit it to change dataset paths, training hyperparameters, logging, and evaluation options.

## Training details

- Training logic is implemented in `train.py` and leverages model definitions in `models/` and loss functions in `losses/`.
- Adaptive components and scheduling (e.g. dynamic controllers) live in `adaptive/` and `utils/`.

## Evaluation details

- Scoring and metric aggregation are implemented in [evaluation/anomaly_scorer.py](evaluation/anomaly_scorer.py) and [evaluation/metrics.py](evaluation/metrics.py).
- Visualizations and heatmap generation are in [evaluation/heatmap.py](evaluation/heatmap.py).

## Checkpoints and outputs

- Models are saved to `checkpoints/`. Keep `best_model.pth` for the best validation run and `last_model.pth` for the most recent.
- Generated outputs (for example, patches or intermediate arrays) may be stored in `outputs/`.

## Extending the codebase

- Add new model architectures under `models/` and expose them to the training script.
- Add or update loss functions in `losses/` and integrate them into the training loop.
- Implement new dataset loaders in `data/` and point `configs/default.yaml` to the proper paths.

## Troubleshooting

- If you hit dependency issues, ensure the Python version matches and reinstall packages:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

- Check `configs/default.yaml` for incorrect paths or hyperparameters.

## Notes

- This repository is intended for research and experimentation. Expect to adapt configs and scripts for your environment and dataset.

If you want, I can also:
- run tests or a dry training run, or
- add a short example `configs/experiment.yaml` and a minimal README section with expected hyperparameters.
