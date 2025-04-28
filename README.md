# Isaac-Drone

Custom reinforcement-learning tasks for **Isaac Lab**.

---

## 🚀 Quick Installation

1. **Install Isaac Sim _and_ Isaac Lab** by following the official guide: <https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html>
2. Clone this repository **inside** your Isaac Lab task directory:

```bash
cd /path/to/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/
git clone https://github.com/esrefsamilzamanoglu/Isaac-Drone.git
```

Isaac Lab automatically discovers any task placed under `isaaclab_tasks`, so no extra setup is needed.

---

## 🏋️‍♂️ Training Tasks

### Quadcopter RSSI — direct control

```bash
./isaaclab.sh \
    -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Isaac-Quadcopter-RSSI-Direct-v0 \
    --max_iterations 1000 \
    --headless            # remove this flag to open a GUI window
```

*(More training examples will be added as new tasks land in this repo.)*

---

## 🔍 Evaluate a Trained Model

After training finishes you will find checkpoints in
`logs/rsl_rl/<task‑name>/`. To roll out the **best-performing** policy:

```bash
./isaaclab.sh \
    -p scripts/reinforcement_learning/rsl_rl/rollout.py \
    --task=Isaac-Quadcopter-RSSI-Direct-v0 \
    --checkpoint=logs/rsl_rl/Isaac-Quadcopter-RSSI-Direct-v0/best.pt \
    --episodes 5 \
    --headless=false      # set true for off‑screen evaluation
```

> **Tip** Use `tensorboard --logdir logs/rsl_rl` to monitor rewards while the
> training job is running.

---

## 📅 Roadmap
- ⬜ Additional quadcopter variants (hover, waypoint nav)
- ⬜ Fixed‑wing UAV tasks
- ⬜ Pre‑trained checkpoints + batch evaluation scripts

---

## License
BSD‑3‑Clause

