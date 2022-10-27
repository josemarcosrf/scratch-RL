# RL from scratch

This repo contains my from-scratch implementations of some common RL algorithms.

The main goal is to do a deep dive by implementing, to go beyond the theoretical
understanding.

<!--ts-->
   * [RL from scratch](#rl-from-scratch)
      * [How To](#how-to)
         * [Install](#install)
         * [Run](#run)

<!-- Added by: jose, at: jue 27 oct 2022 18:32:28 CEST -->

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


### Run

> If at run time, when running and environtment with `render_mode="human"` you see:
> `libGL error: MESA-LOADER`, you might need to run like:
> ```bash
> MESA_LOADER_DRIVER_OVERRIDE=i965 python <your-script.py>
> ```
> For more info check [this blogpost](https://devcodetutorial.com/faq/libgl-error-failed-to-load-drivers-iris-and-swrast-in-ubuntu-20-04)
