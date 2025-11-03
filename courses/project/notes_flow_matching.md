
## Diffusion as a Special Case of FM

Shi et al. (2023); Liu et al. (2023) show that the linear version of Flow Matching with **Gaussian trajectories** (as path) can be seen as a certain limiting case of bridge matching.

### Diffusion Bridge

- Ordinary diffusion: starts from a noisy prior (e.g. Gaussian noise) and ends anywhere in the data distribution.

- **Diffusion bridge**: starts at one known point (e.g. a data sample or boundary condition) and must end at another known point (e.g. a specific value at time 1).
