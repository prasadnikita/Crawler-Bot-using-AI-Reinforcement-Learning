# 🤖 Crawler Bot using AI Reinforcement Learning

> Training a two-armed crawler bot to walk efficiently using three reinforcement learning algorithms — Monte Carlo, Q-Learning, and Temporal Difference (TD) Learning.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Reinforcement Learning](https://img.shields.io/badge/RL-Q--Learning%20%7C%20Monte%20Carlo%20%7C%20TD-green)
![NumPy](https://img.shields.io/badge/NumPy-1.x-orange)

---

## Overview

This project implements and compares three classic reinforcement learning algorithms to optimize the locomotion of a simulated crawler bot. The bot is equipped with two arms and must learn — purely through trial and reward — how to move forward as efficiently as possible.

Each algorithm takes a different approach to learning the optimal policy from environment interactions, making this an ideal comparison study of foundational RL methods.

---

## Algorithms Implemented

| Algorithm | Approach | Update Strategy |
|-----------|----------|----------------|
| **Monte Carlo** | Episode-based learning | Updates after full episode completes |
| **Q-Learning** | Off-policy TD control | Updates at every step using Bellman equation |
| **Temporal Difference (TD)** | Online learning | Updates using bootstrapped value estimates |

---

## How It Works

```
Crawler Bot (2 arms, multiple joint angles)
          │
          ▼
Agent observes current state (arm positions)
          │
          ▼
Selects action (move arm up/down) via ε-greedy policy
          │
          ▼
Environment returns reward (forward movement)
          │
          ▼
Agent updates Q-values using selected RL algorithm
          │
          ▼
Repeat until optimal walking policy is learned
```

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core implementation |
| NumPy | State/action value computation |
| Matplotlib | Reward and convergence plots |
| Custom Environment | Crawler bot physics simulation |

---

## Getting Started

### Installation

```bash
git clone https://github.com/prasadnikita/Crawler-Bot-using-AI-Reinforcement-Learning.git
cd Crawler-Bot-using-AI-Reinforcement-Learning
pip install numpy matplotlib
```

### Run

```bash
# Run the RL training
python RLearning.py

# Run the crawler bot simulation
python crawler.py
```

---

## Project Structure

```
Crawler-Bot-using-AI-Reinforcement-Learning/
├── crawler.py       # Crawler bot environment and physics
├── RLearning.py     # Monte Carlo, Q-Learning, and TD implementations
└── README.md
```

---

## Key Concepts

**State Space** — Joint angles of the two arms define the bot's current configuration.

**Action Space** — Each arm can move to discrete angle positions at each time step.

**Reward Signal** — Positive reward for forward displacement; encourages efficient locomotion.

**Policy** — ε-greedy exploration balances exploitation of known good actions with exploration of new ones.

---

## License

MIT
