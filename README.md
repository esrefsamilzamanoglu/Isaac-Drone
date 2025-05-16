# Isaac-Drone

Custom reinforcement-learning tasks for **Isaac Lab**.

---

## 🚀 Quick Installation

1. **Install Isaac Sim *and* Isaac Lab** by following the official guide: [https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip\_installation.html](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html)
2. Clone this repository **inside** your Isaac Lab task directory:

   ```bash
   cd /path/to/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/
   git clone https://github.com/esrefsamilzamanoglu/Isaac-Drone.git
   ```

Isaac Lab automatically discovers any task placed under `isaaclab_tasks`, so no extra setup is needed.

---

## 🏋️‍♂️ Training Tasks

### Quadcopter RSSI — Simple Environment

```bash
./isaaclab.sh \
  -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Isaac-Drone-Simple-RSSI-Seeker \
  --max_iterations=500 \
  --seed=99 \
  --headless
```

*(Add or remove `--headless` to toggle GUI on/off.)*

### Quadcopter RSSI — Sionna Transfer Learning

```bash
./isaaclab.sh \
  -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Isaac-Drone-Sionna-RSSI-Seeker \
  --checkpoint=model_400.pt \
  --resume \
  --max_iterations=500 \
  --seed=99 \
  --headless \
  --load_run=2025-05-16_17-15-43
```

* `--checkpoint` should point to your pretrained `model_400.pt`.
* `--load_run` is the dirname of the log folder containing that checkpoint.
* `--resume` instructs the trainer to restore optimizer/scheduler state and continue training.

---

## 🔍 Evaluate a Trained Model

To roll out a **Simple-RSSI** policy in a single environment:

```bash
./isaaclab.sh \
  -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task=Isaac-Drone-Simple-RSSI-Seeker \
  --num_envs=1 \
  --checkpoint=logs/rsl_rl/quadcopter_rssi/2025-05-15_12-53-51/model_400.pt
```

* Set `--headless=false` if you want an interactive GUI.

---

## 📊 Visualize Training Logs

Launch TensorBoard to inspect rewards, losses, and other metrics:

```bash
tensorboard --logdir=logs/rsl_rl/quadcopter_rssi/2025-05-16_17-15-43
```

Replace the timestamp with your actual run folder.

> **Tip:** Use `tensorboard --logdir logs/rsl_rl` to see all runs at once.

---

## 📅 Roadmap

* ⬜ Additional quadcopter variants (hover, waypoint navigation)
* ⬜ Fixed‑wing UAV tasks
* ⬜ Batch evaluation scripts & plotting utilities
* ⬜ Community contributions and pretrained models

---

## 📄 License

BSD-3-Clause
