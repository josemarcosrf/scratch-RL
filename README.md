# RL from scratch

This repo contains my from-scratch implementations of some common RL algorithms.

The main goal is to do a deep dive by implementing, to go beyond the theoretical
understanding.

Therefore it is structured with simplicity in mind, with each algorithm being
self contained and in a single file aand with as little abstraction as possible.

<!--ts-->
   * [RL from scratch](#rl-from-scratch)
      * [How To](#how-to)
         * [Install](#install)
      * [Run](#run)
         * [Tabular first-visit Monte Carlo](#tabular-first-visit-monte-carlo)
         * [Tabular SARSA](#tabular-sarsa)
         * [Tabular Q-learning](#tabular-q-learning)

<!-- Added by: jose, at: mar 01 nov 2022 19:07:41 CET -->

<!--te-->


## How To

### Install

1. Install mesa glx drivers:

```bash
sudo apt-get install -y mesa-utils and libgl1-mesa-glx
sudo apt-get install mesa-common-dev libgl1-mesa-dev libglu1-mesa-dev
```

2. Create a virtual env with `python>=3.8`

```bash
python3.8 -m venv .venv
source .venv/bin/activate
```

3. Then install with:

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Run

> If at run time, when running and environtment with `render_mode="human"` you see:
> `libGL error: MESA-LOADER`, you might need to run like:
> ```bash
> MESA_LOADER_DRIVER_OVERRIDE=i965 python <your-script.py>
> ```
> For more info check [this blogpost](https://devcodetutorial.com/faq/libgl-error-failed-to-load-drivers-iris-and-swrast-in-ubuntu-20-04)



### Tabular first-visit Monte Carlo

**Valid environments**:

 - `Blackjack-v1`


### Tabular SARSA

**Valid environments:**

 - `CliffWalking-v0`
 - `FrozenLake-v1`

**Examples:**

```bash
 # SARSA on Cliff World
python -m algorithms.tabular.sarsa -e CliffWalking-v0
```
### Tabular Q-learning

**Valid environments:**

 - `CliffWalking-v0`
 - `FrozenLake-v1`


**Examples:**

```bash
# Q-learning on Frozen lake
python -m algorithms.tabular.q_learning -e FrozenLake-v1 -ep 0.5 -ne 5000
```
