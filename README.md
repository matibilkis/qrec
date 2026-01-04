# QREC & MAREK

**Reinforcement Learning for Optimal Quantum State Discrimination**

[![arXiv](https://img.shields.io/badge/arXiv-2404.10726-b31b1b.svg)](https://arxiv.org/abs/2404.10726)
[![arXiv](https://img.shields.io/badge/arXiv-2203.09807-b31b1b.svg)](https://arxiv.org/abs/2203.09807)
[![arXiv](https://img.shields.io/badge/arXiv-2001.10283-b31b1b.svg)](https://arxiv.org/abs/2001.10283)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Model-free calibration of quantum receivers through trial and error.

---

## Overview

This repository provides a comprehensive framework for implementing **reinforcement learning (RL)** techniques to achieve **optimal quantum state discrimination** over unknown channels. The codebase enables real-time calibration and optimization of quantum devices—particularly **coherent-state receivers**—without requiring prior knowledge of system parameters.

### The Challenge

Quantum devices are particularly challenging to operate: their functionality relies on precisely tuning parameters, yet environmental conditions constantly shift, causing detuning. Traditional approaches require detailed modeling of environmental behavior, which is often computationally unaffordable, while direct parameter measurements introduce extra noise.

### Our Solution

We frame quantum receiver calibration as a **reinforcement learning problem**, where an agent learns optimal discrimination strategies through trial and error—without any prior knowledge of experimental details.

---

## Key Features

**Q-Learning Framework**
- ε-greedy exploration with configurable parameters
- Adaptive learning rates (1/N decay or fixed)
- Real-time Q-value updates for optimal policy discovery
- Support for change-point detection scenarios

**Quantum Physics Engine**
- Born rule probability calculations
- Coherent state displacement operations
- Kennedy receiver simulation
- Variable-loss optical channel modeling

**Analysis & Visualization**
- Learning curve generation
- Q-function landscape plotting
- Noise sensitivity analysis
- Comparative performance metrics

**Dynamic Calibration**
- Model-free control loops
- Continuous parameter re-calibration
- Adaptation to environmental drift
- Optimal β displacement learning

---

## Repository Structure

```
qrec/
├── qrec/                         # Core module
│   └── utils.py                  # Q-learning utilities, physics functions
├── experiments/                  # Experimental configurations
│   ├── 0/                        # Full exploration (ε=1.0)
│   ├── 1/                        # Low exploration, 1/N learning rate
│   ├── 2/                        # Change-point: α = 1.5 → 0.25
│   ├── 3/                        # Fixed lr = 0.005
│   ├── 4/                        # Fixed lr = 0.05
│   ├── 5/                        # Change-point with fixed lr (best)
│   └── 6/                        # Noise inspection
├── paper/                        # Publication figures
├── basic_inspection.py           # Error landscape visualization
├── index_experiments             # Experiment documentation
└── requirements.txt              # Dependencies
```

**Related Repository:** [matibilkis/marek](https://github.com/matibilkis/marek)

```
marek/
├── main_programs/                # Core RL algorithms
├── dynamic_programming/          # DP optimization modules
├── bounds_optimals_and_limits/   # Theoretical bounds computation
├── plotting_programs/            # Visualization tools
├── appendix_A/                   # Supplementary materials
├── tests/                        # Validation suite
├── agent.py                      # RL agent implementation
├── environment.py                # Quantum channel simulation
├── training.py                   # Training loop
└── basics.py                     # Core physics functions
```

---

## Mathematical Framework

### Coherent State Discrimination

The goal is to discriminate between coherent states |±α⟩ using a Kennedy-like receiver with displacement β:

```
P(n|α) = exp(-|α|²) · δₙ₀ + (1 - exp(-|α|²)) · δₙ₁
```

The **success probability** for a given displacement β:

```
Pₛ(β) = ½ Σₙ max_{s∈{-1,+1}} P(n | sα + β)
```

### Q-Learning Update Rules

**Action-value updates:**
```
Q₁(β, n, g) ← Q₁(β, n, g) + α · [r - Q₁(β, n, g)]
Q₀(β) ← Q₀(β) + α · [max_g Q₁(β, n, g) - Q₀(β)]
```

**Policy (ε-greedy):**
```
π(β) = { random      with probability ε
       { argmax Q₀   with probability 1-ε
```

---

## Quick Start

### Installation

```bash
git clone https://github.com/matibilkis/qrec.git
cd qrec
pip install -r requirements.txt
```

### Basic Usage

```python
from qrec.utils import *

# Initialize Q-tables with 25 discretized β values
betas_grid, [q0, q1, n0, n1] = define_q(nbetas=25)

# Find model-aware optimal (for comparison)
mmin, p_star, beta_star = model_aware_optimal(betas_grid, alpha=0.4)

# Run Q-learning episode
hidden_phase = np.random.choice([0, 1])  # Nature chooses ±α
indb, beta = ep_greedy(q0, betas_grid, ep=0.01)  # Agent chooses β
n = give_outcome(hidden_phase, beta, alpha=0.4)  # Photon detection
indg, guess = ep_greedy(q1[indb, n, :], [0, 1], ep=0.01)  # Agent guesses
reward = give_reward(guess, hidden_phase)  # Success/failure
```

### Running Experiments

```bash
cd experiments/5
python change_point.py
```

---

## Results

The RL agent successfully learns near-optimal receiver configurations:

| Experiment | Configuration | Key Finding |
|:-----------|:--------------|:------------|
| 0 | ε = 1.0 (full exploration) | Baseline uniform sampling |
| 1 | ε = 0.01, lr = 1/N | Convergent but slow adaptation |
| 2 | Change-point, lr = 1/N | Cannot adapt to α changes |
| 3 | ε = 0.01, lr = 0.005 | Stable but slow learning |
| 4 | ε = 0.01, lr = 0.05 | Good balance |
| 5 | Change-point, lr = 0.05 | **Successful re-calibration** |

<p align="center">
  <img src="paper/image.png" alt="RL agent learning curves" height="200"/>
  <img src="paper/image copy.png" alt="Quantum receiver setup schematic" height="200"/>
</p>
<p align="center"><em>Our agents in action: learning curves for sensor calibration</em></p>

> **Key Insight:** Fixed learning rates enable adaptation to changing channel conditions, while decaying rates (1/N) lock the agent to initial configurations.

---

## API Reference

### Physics Functions

| Function | Description |
|:---------|:------------|
| `p(alpha, n)` | Born rule probability P(n\|α) |
| `Perr(beta, alpha)` | Error probability for displacement β |
| `give_outcome(phase, beta, alpha)` | Sample photon detection outcome |
| `model_aware_optimal(betas, alpha)` | Compute theoretical optimum |

### Q-Learning Functions

| Function | Description |
|:---------|:------------|
| `define_q(nbetas)` | Initialize Q-tables and counters |
| `ep_greedy(qvals, actions, ep)` | ε-greedy action selection |
| `greedy(arr)` | Greedy selection (ties broken randomly) |
| `give_reward(guess, phase)` | Binary reward function |
| `Psq(q0, q1, betas, alpha)` | Evaluate current policy |

---

## Dependencies

```
numpy
matplotlib
scipy
tqdm
numba
```

---

## Contributing

Contributions are welcome. Please open an issue first to discuss proposed changes.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Publications

This framework has enabled three peer-reviewed publications in quantum machine learning:

<table>
<tr valign="top">
<td width="33%">

### Automatic Re-calibration of Quantum Devices by RL

**T. Crosta, L. Rebón, F. Vilariño, J.M. Matera, M. Bilkis**

[arXiv:2404.10726](https://arxiv.org/abs/2404.10726) (2024)

Model-free RL framework for continuous recalibration of quantum device parameters. Demonstrated on Kennedy receiver-based long-distance quantum communication.

- Continuous recalibration
- No prior knowledge needed
- Environmental drift adaptation

</td>
<td width="33%">

### RL Calibration on Variable-Loss Optical Channels

**M. Bilkis, M. Fraas, A. Acín, G. Sentís**

[arXiv:2203.09807](https://arxiv.org/abs/2203.09807) (2022)

Calibration of quantum receivers for optical coherent states over channels with variable transmissivity using reinforcement learning.

- Variable loss channels
- Error probability optimization
- No channel tomography

</td>
<td width="33%">

### Real-Time Calibration: Learning by Trial and Error

**M. Bilkis, M. Rosati, R. Muñoz-Tapia, J. Calsamiglia**

[Phys. Rev. Research 2, 033295](https://arxiv.org/abs/2001.10283) (2020)

Foundational work: RL agents learn near-optimal coherent-state receivers through real-time trial and error experimentation.

- First RL quantum receiver
- Trial and error learning
- Linear optics + detectors

</td>
</tr>
</table>

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{crosta2024automatic,
  title={Automatic re-calibration of quantum devices by reinforcement learning},
  author={Crosta, T. and Reb{\'o}n, L. and Vilari{\~n}o, F. and Matera, J. M. and Bilkis, M.},
  journal={arXiv preprint arXiv:2404.10726},
  year={2024}
}

@article{bilkis2022reinforcement,
  title={Reinforcement-learning calibration of coherent-state receivers on variable-loss optical channels},
  author={Bilkis, M. and Fraas, M. and Ac{\'i}n, A. and Sent{\'i}s, G.},
  journal={arXiv preprint arXiv:2203.09807},
  year={2022}
}

@article{bilkis2020realtime,
  title={Real-time calibration of coherent-state receivers: Learning by trial and error},
  author={Bilkis, M. and Rosati, M. and Mu{\~n}oz-Tapia, R. and Calsamiglia, J.},
  journal={Physical Review Research},
  volume={2},
  pages={033295},
  year={2020}
}
```
