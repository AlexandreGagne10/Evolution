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

### Configuration

Key constants controlling the simulation are defined near the top of `evolution_sim.py`:

- `FOOD_PER_ROUND` – number of food dots spawned each round.
- `MAX_AGE` – maximum lifetime (in rounds) of an agent.
- `SURVIVAL_FOOD` – minimum food required per round to stay alive.
- `REPRODUCTION_THRESHOLD` – energy an agent needs before it can reproduce.
- `REPRODUCTION_COST` – energy cost paid by each parent on reproduction.

Modifying these values allows you to tune the difficulty of survival. Each agent starts with `SURVIVAL_FOOD` energy so it can survive the first round. You can override the starting energy when using the `Simulation` class directly via the `initial_energy` argument.

## License

This project is licensed under the terms of the MIT License. See [LICENSE](LICENSE) for details.
