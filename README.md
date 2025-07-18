# Evolution

This project contains a simple agent-based evolution simulation implemented in Python. Agents move around a 2D map to eat food, reproduce and survive. The simulation uses PyTorch tensors so it can take advantage of GPU acceleration when available.

## Features

- Specify the number of starting agents with `--agents`.
- Agents reproduce to create babies; no new agents can be added manually after the simulation starts.
- Slider control to speed up the simulation time.
- Optionally runs on GPU when PyTorch detects CUDA support.

## Requirements

- Python 3.8+
- `pygame`
- `torch`

Install the dependencies with pip:

```bash
pip install pygame torch torchvision
```

## Usage

Run the simulation with a chosen number of agents (for example 200) and GPU when available:

```bash
python evolution_sim.py --agents 200 --device cuda
```

Move the slider at the bottom of the window to accelerate or slow down the simulation.
