# Active Cell Balancing in Batteries Under Dynamic Driving Conditions

This repository provides an advanced simulation environment for active battery cell balancing using a reinforcement learning framework. It implements a custom Gymnasium environment that models the dynamics of a lithium-ion battery pack, focusing on energy transfer between adjacent cells to achieve optimal voltage and state-of-charge (SOC) balance.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Environment Overview](#environment-overview)
- [Development](#development)
- [License](#license)
- [Citing](#Citing)
- [References](#references)

## Features
- **Realistic Battery Modeling:**  
  Simulates individual cell properties including capacity, internal resistance, and voltage-SOC mapping based on experimental Li-NMC data.
- **Driving Cycle Integration:**  
  Converts real-world driving cycle data into discharge current profiles, integrating vehicle dynamics and load conditions.
- **Discrete Action Space:**  
  Offers a multi-dimensional discrete action space where each action controls energy transfer intensity and direction between adjacent cells.
- **Custom Reward Function:**  
  Incentivizes effective cell balancing by rewarding the minimization of SOC imbalances.
- **MDP Framework:**  
  The environment is modeled as a Markov Decision Process (MDP) with well-defined state and action spaces, transition dynamics, and long-term discounting.

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/messlem99/Battery_Cell_Balancing.git
   cd Battery_Cell_Balancing
## Usage
To test the simulation environment, follow these steps:
1. **Initialize the Environment:**
   ```bash
   import gymnasium as gym
   from Battery_Cell_Balancing import BatteryBalancingEnv
   env = BatteryBalancingEnv(driving_cycle_filepath="data.csv")
   observation, info = env.reset(seed=42)
2. **Run a Simulation Episode:**
   ```bash
   done = False
   while not done:
    action = env.action_space.sample()  # Sample random actions
    observation, reward, done, truncated, info = env.step(action)
    print(f"Reward: {reward}, Done: {done}")
## Environment Overview
The environment is modeled as an MDP with the following components:
- **State Space:**
Each state includes per-cell voltage, SOC values, and the instantaneous load current derived from the driving cycle.
- **Action Space:**
A discrete multi-dimensional vector controlling energy transfers between adjacent cells.
- **Transition Dynamics:**
Updates based on realistic battery dynamics including load current effects and energy balancing operations.
- **Reward Function:**
Rewards effective balancing by minimizing SOC differences between adjacent cells.
- **Termination Conditions:**
Episodes end when voltage or SOC imbalances exceed defined thresholds, or if all cells are fully discharged.
## Development
Contributions and enhancements are welcome. To get started:
- Fork the repository and create your feature branch.
- Submit pull requests for review.
## License
- This project is licensed under the MIT License. See the LICENSE file for details.
## Citing
To cite this project in publications:
```bash
@misc{EVBatteryBalancing2025,
  author       = {Abdelkader Messlem and Youcef Messlem and Ahmed Safa},
  title        = {A Simulation Environment for Active Battery Cell Balancing Using Gymnasium},
  year         = {2025},
  howpublished = {\url{https://github.com/messlem99/Battery_Cell_Balancing}},
}
```
## References
- F. Baronti, W. Zamboni, N. Femia, R. Roncella, and R. Saletti, “Experimental analysis of open-circuit voltage hysteresis in lithiumiron-phosphate batteries,” in IECON 2013 - 39th Annual Conference of the IEEE Industrial Electronics Society. Vienna, Austria: IEEE, Nov. 2013, pp. 6728–6733. [Online]. Available: http://ieeexplore.ieee. org/document/6700246/
- P. Ramadass, B. Haran, R. White, and B. N. Popov, “Mathematical modeling of the capacity fade of Li-ion cells,” Journal of Power Sources, vol. 123, no. 2, pp. 230–240, Sep. 2003. [Online]. Available: https://linkinghub.elsevier.com/retrieve/pii/S0378775303005317
- G. Ning, B. Haran, and B. N. Popov, “Capacity fade study of lithium-ion batteries cycled at high discharge rates,” Journal of Power Sources, vol. 117, no. 1-2, pp. 160–169, May 2003. [Online]. Available: https://linkinghub.elsevier.com/retrieve/pii/S0378775303000296
