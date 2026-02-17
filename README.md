# Robustness in Reinforcement Learning under Task-Uncertainty

**Master's Thesis** | Universitat Pompeu Fabra | Master in Intelligent Interactive Systems

**Author**: Nikodim Aleksandrovich Svetlichnyi  
**Supervisor**: Dr. Miguel Calvo-Fullana  
**Date**: July 2025

---

## Abstract

This repository contains the implementation of a modular framework for achieving robustness in reinforcement learning agents operating under task-uncertainty. Instead of training a single monolithic meta-policy, we propose a committee-based approach utilizing pre-trained expert policies, each specialized for a distinct known task. The framework develops two online adaptation mechanisms for dynamic task inference and policy selection based solely on reward feedback.

## Problem Formulation

### Multi-Task MDP with Task-Uncertainty

We consider an agent operating in an environment modeled as a multi-task Markov Decision Process (MDP), where:

- State space: $s \in \mathcal{S}$
- Action space: $a \in \mathcal{A}$
- Transition dynamics: $p(s_{t+1} | s_t, a_t)$
- Reward functions: $\\{r_i : \mathcal{S} \times \mathcal{A} \to \mathbb{R}\\}_{i=1}^m$

The agent must find a policy $\pi$ that performs well across all potential tasks without knowing which reward function is active at execution time.

### Robust Optimization Objective

The core optimization problem is formulated as a max-min robust reinforcement learning objective:

$$P^* = \max_{\pi \in \Pi} \min_{i \in [m]} V_i(\pi)$$

where $V_i(\pi)$ denotes the expected return under reward function $r_i$ and policy $\pi$.

Using Lagrangian duality theory, this non-convex problem can be reformulated as:

$$D^* = \min_{\lambda \in \Delta_m} \max_{\pi \in \Pi} \sum_{i=1}^m \lambda_i V_i(\pi)$$

where $\Delta_m$ is the probability simplex satisfying $\sum_{i=1}^m \lambda_i = 1, \lambda_i \geq 0$.

A key theoretical result establishes that strong duality holds for this class of problems, ensuring $P^* = D^*$.

## Methodology

### Two-Phase Approach

#### Phase 1: Training Expert Policies

For each task $i \in \\{1, \ldots, m\\}$, independently train an expert policy using Proximal Policy Optimization (PPO):

$$\pi_i^* \in \arg\max_{\pi \in \Pi} V_i(\pi)$$

This results in a committee of specialized policies $\{\hat{\pi}_1,\hat{\pi}_2,\ldots,\hat{\pi}_m^{*}\}$..

#### Phase 2: Online Adaptation Mechanisms

We develop two distinct algorithms for online policy selection:

### Algorithm 1: Dual-Lambda Algorithm

**Theoretical Foundation**: Derived from the dual formulation using Lagrangian relaxation and primal-dual optimization.

**Core Update Rule**:

At each epoch $k$, the dual variables $\lambda$ are updated using gradient descent on the dual objective:

$$\lambda_{i,k+1/2} = \lambda_{i,k} - \frac{\eta_\lambda}{T_0} \sum_{t=kT_0}^{(k+1)T_0-1} r_i(s_t, a_t)$$

followed by projection onto the simplex:

$$\lambda_{k+1} = \Pi_{\Delta_m}(\lambda_{k+1/2})$$

where $\Pi_{\Delta_m}$ denotes the Euclidean projection operator onto the probability simplex.

**Properties**:
- Provides formal convergence guarantees to the minimax optimum
- Exhibits inherent conservatism due to worst-case optimization
- On-policy and model-free

### Algorithm 2: Predictive Control Algorithm

**Theoretical Foundation**: Inspired by Model Predictive Control (MPC) principles, employing receding horizon optimization.

**Core Selection Rule**:

At each decision step $k$ with current state $s_k$:

1. Simulate forward trajectories for each expert policy over horizon $T_0$:

$$\tau_i(s_k) = \\{s'_{k,0}, a'_{k,0}, s'_{k,1}, a'_{k,1}, \ldots, s'_{k,T_0-1}, a'_{k,T_0-1}\\}$$

where $s'\_{k,0}=s\_k$ and $a'\_{k,t}=\pi\_i(s'\_{k,t})$.

2. Evaluate cumulative reward using the true environment reward:

$$R_i(s_k) = \sum_{t=0}^{T_0-1} r_{\text{true}}(s'_{k,t}, a'_{k,t})$$

3. Select the best policy and execute only its first action:

$$i_k^* = \arg\max_{i \in [m]} R_i(s_k)$$

**Properties**:
- Pragmatic and highly adaptive
- Exhibits emergent role specialization
- No formal guarantees, but superior empirical performance

## Experimental Environment

### 2D Navigation Task

**State Space**: $s_t = (x_t, y_t, \theta_t)$ representing position and orientation

**Action Space**: $a_t = (v_t, \omega_t)$ representing linear and angular velocities

**Dynamics**:

$$\begin{cases}
x_{t+1} = x_t + v_t \cos(\theta_t) \Delta t \\\\
y_{t+1} = y_t + v_t \sin(\theta_t) \Delta t \\\\
\theta_{t+1} = \theta_t + \omega_t \Delta t
\end{cases}$$

**Observation Vector**:

$$o_t = [d_t^{\text{target}}, \phi_t^{\text{target}}, d_t^{\text{obs}}, \phi_t^{\text{obs}}]^\top$$

The agent perceives the environment through a 60-degree forward-facing sensor dome with range $r_{\text{sensor}} = 2.0$.

**Reward Function**:

$$R(s_t, a_t, s_{t+1}) = \begin{cases}
-C_{\text{penalty}} & \text{if collision with obstacle} \\\\
-k \cdot d(s_{t+1}, \text{goal}) & \text{otherwise}
\end{cases}$$

where $C_{\text{penalty}} = 15$ and $k = 1.0$.

## Key Results

### Three-Phase Evaluation Protocol

1. **Phase 1**: Baseline environments with circular obstacles $r \in \\{1, 2, 3\\}$
2. **Phase 2**: Complex composite environment with multiple goals and heterogeneous obstacles
3. **Phase 3**: Zero-shot generalization to novel obstacle geometries (triangles, squares)

### Performance Comparison (Phase 2)

Averaged over 30 experimental runs with 2-4 goals:

| Algorithm | Avg. Reward | Avg. Steps | Obstacle Collisions | Collision Rate |
|-----------|-------------|------------|---------------------|----------------|
| Predictive Control | -9304.62 | 965.0 | 27.4 | 2.79% |
| Dual-Lambda | -13014.19 | 1127.4 | 120.8 | 11.51% |
| MetaRL Baseline | -41168.39 | 2849.9 | 423.8 | 16.62% |

### Key Findings

1. Both proposed algorithms significantly outperform classical MetaRL by 3-4× in complex environments
2. Predictive Control demonstrates 71.8% fewer obstacle collisions than Dual-Lambda
3. Dual-Lambda provides inherent conservatism valuable for safety-critical applications
4. Performance is fundamentally bounded by the diversity of the expert policy set
5. Zero-shot generalization to novel geometries is partially successful but reveals limitations

## Installation

```bash
# Clone the repository
git clone https://github.com/nikodim-working-on-code/robust-rl-task-uncertainty.git
cd robust-rl-task-uncertainty

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Requirements

- Python 3.8+
- gymnasium==0.29.1
- stable-baselines3==2.3.2
- numpy==1.26.0
- scipy==1.11.4
- matplotlib==3.8.2
- shapely==2.0.7


## Citation

If you use this work in your research, please cite:

```bibtex
@mastersthesis{svetlichnyi2025robust,
  title={Robustness in Reinforcement Learning under Task-Uncertainty},
  author={Svetlichnyi, Nikodim Aleksandrovich},
  year={2025},
  school={Universitat Pompeu Fabra},
  type={Master's Thesis},
  supervisor={Miguel Calvo-Fullana}
}
```

## Acknowledgments

This research was conducted under the supervision of **Dr. Miguel Calvo-Fullana** at the Universitat Pompeu Fabra. I express my sincere gratitude for his invaluable guidance, insightful discussions, and unwavering support throughout this work. His expertise in robust control and optimization theory was instrumental in developing the theoretical foundations of this thesis.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

Nikodim Aleksandrovich Svetlichnyi  
Master in Intelligent Interactive Systems  
Universitat Pompeu Fabra  

For questions or collaborations, please open an issue on GitHub.

---

**Keywords**: Reinforcement Learning, Task-Uncertainty, Robust Optimization, Meta-Learning, Multi-Task Learning, Lagrangian Duality, Model Predictive Control, Robot Navigation, Expert Policy Committees
