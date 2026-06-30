# BC vs DAgger

Comparison of Behavior Cloning (BC) and Dataset Aggregation (DAgger) for imitation learning.

## Structure

```
src/
  experts.py      # train PPO expert; load + query interface
  data.py         # rollout collection, demo dataset, dagger buffer
  policy.py       # imitation MLP shared by BC and DAgger
  bc.py           # behavior cloning train loop
  dagger.py       # DAgger loop (rollout learner → query expert → aggregate)
  evaluate.py     # headless rollouts → mean/std return
  utils.py        # seeding, plotting, csv logging

scripts/
  01_train_expert.py        # train and save PPO expert
  02_collect_demos.py       # collect expert demonstrations
  03_train_bc.py            # train BC policy
  04_bc_data_ablation.py    # sweep #demos → show return collapse
  05_train_dagger.py        # train DAgger policy
  06_make_plots.py          # return curves: BC vs DAgger vs expert
```

## Usage

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python scripts/01_train_expert.py
python scripts/02_collect_demos.py
python scripts/03_train_bc.py
python scripts/04_bc_data_ablation.py
python scripts/05_train_dagger.py
python scripts/06_make_plots.py
```

Notes:
- `scripts/01_train_expert.py` defaults to `Hopper-v5`, which requires MuJoCo. That dependency is included via `gymnasium[mujoco]` in `requirements.txt`.
- Avoid hard-coding `/usr/local/bin/python3`; use the `.venv` interpreter so `numpy`, `torch`, `gymnasium`, and `stable-baselines3` all come from the same environment.

Results (CSV + figures) are written to `results/`.
