# Dynamic cyber investment model with insurance

This repository contains the Python implementation used in the numerical analysis of a dynamic model of optimal investment in cybersecurity with cyber insurance.

The model combines:

- a dynamic security level controlled by investment,
- a Hawkes process describing clustered cyberattacks,
- a stop-loss cyber insurance mechanism,
- a Hamilton-Jacobi-Bellman integro-partial differential equation (HJB-PIDE),
- numerical procedures for computing the optimal investment strategy.

## Repository structure

```text
configs/    Parameter configurations used in numerical experiments
src/        Core implementation of the model, solver and simulation tools
scripts/    Scripts used to reproduce selected numerical experiments
```

Generated numerical results and plots are not included in the repository because of their size. They can be reproduced locally by running the scripts.

## Requirements

The code was developed in Python. The main dependencies are listed in `requirements.txt`.

## Usage

After installing the required packages, the scripts in the `scripts/` directory can be used to reproduce the main numerical experiments.

## Notes

This repository accompanies a master's thesis on a stochastic extension of the Gordon-Loeb model to a dynamic setting with Hawkes cyberattacks and cyber insurance.
