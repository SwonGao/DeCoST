# DeCoST: Decoupled Discrete-Continuous Optimization with Service-Time-Guided Trajectory
[![ICLR-Brazil](https://img.shields.io/badge/ICLR-Brazil-green)](https://iclr.cc/Conferences/2026)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Conference](https://img.shields.io/badge/ICLR-Poster-blue)](#paper)
[![Paper](https://img.shields.io/badge/Paper-OpenReview-orange)](https://openreview.net/forum?id=koIbbsfKSf)

Hi! Thanks for your interest in our work.

This repository provides the PyTorch implementation of DeCoST (Decoupled Discrete-Continuous Optimization with Service-Time-Guided Trajectory) for the Orienteering Problem with Time Windows and Variable Profits (OPTWVP). Our work, Learning to Solve Orienteering Problem with Time Windows and Variable Profits (ICLR 2026), tackles this setting with a two-stage learning framework that decouples discrete routing and continuous service-time decisions. The first stage predicts routes and initial service times with parallel decoding, while the second stage refines service times via a linear program with provable global optimality. Experiments show DeCoST outperforms state-of-the-art constructive and meta-heuristic solvers in solution quality and efficiency, with up to 6.6x faster inference on instances under 500 nodes.

Project page: https://swongao.github.io/DeCoST_iclr2026/

If you find our work useful, please cite:

```bibtex
@inproceedings{gao2026decost,
  title={Learning to Solve Orienteering Problem with Time Windows and Variable Profits},
  author={Songqun Gao and Zanxi Ruan and Patrick Floor and Marco Roveri and Luigi Palopoli and Daniele Fontanelli},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
}
```

## Setup

Recommended environment setup with conda:

```bash
git clone https://github.com/SwonGao/DeCoST
conda create -n decost python=3.12
conda activate decost
pip install torch torchvision torchaudio
pip install typeguard matplotlib tqdm pytz scikit-learn tensorflow tensorboard_logger pandas gurobipy wandb
```

If you are using CUDA, install the PyTorch build that matches your CUDA version from the official PyTorch site.

Make sure the Python working directory is `~/DeCoST/DeCoST`. For example, in Visual Studio Code, add:

```json
"cwd": "${workspaceFolder}/DeCoST",
```

to `.vscode/launch.json`.

## Project Structure

- `DeCoST/`: source code and scripts
- `data/`: datasets and generated instances

## Data Generation
```bash
python generate_data.py --problem OPTWVP --problem_size 50 --timewindows 100
```

## Training
```bash
python train.py --problem OPTWVP --problem-size 50 --timewindows 100
```

## Evaluation
```bash
python test.py
```

## License

This project is released under the MIT License. See `LICENSE` for details.

## Acknowledgments

- POMO: https://github.com/yd-kwon/POMO
- GFACS: https://github.com/ai4co/gfacs
- PIP-constraint: https://github.com/jieyibi/PIP-constraint
- ILS: https://codeocean.com/capsule/7061648/tree/v1
- Gurobi: https://www.gurobi.com/
