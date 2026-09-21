# Environment setup

```bash
bash environment/install.sh              # creates conda env "harl_marl" with CUDA torch
bash environment/install.sh my_env_name  # custom env name
bash environment/install.sh harl_marl --cpu   # CPU-only torch
```

`requirements.txt` is the curated set actually needed to run this project's `mujoco_marl` /
`safety_marl` experiments (`examples/train.py`, `examples/train_obs_attacker.py`,
`examples/eval_*_vs_pedm.py`, `examples/train_pedm_detector.py`). `requirements-lock.txt` is a
full `pip freeze` from the dev machine's own `mujoko` conda env, kept only as a reference for
exact-version debugging -- don't install from it directly, it includes a lot of unrelated
packages from this project's broader history (pandapower, sacred, pettingzoo, etc.).

**numpy must stay at 1.23.5** -- see the comment in `requirements.txt`. Checkpoints saved under
numpy>=2.0 fail to load under numpy<2.0 (`ModuleNotFoundError: No module named 'numpy._core'`).

After install, victim checkpoints and PEDM detectors are expected under `results/robust_victim/
<scenario>/...` and `results/obs_attackers/<scenario>/pedm_detector.pt` -- these are NOT part of
the git repo (`results/` is gitignored, checkpoints are large binaries) and must be copied over
separately from wherever the experiments were originally run, or regenerated:
- victim: this project's own `train_*_victim.py` / HARL's own `mappo` training on the target
  scenario (not scripted here -- was trained in earlier sessions, copy the checkpoint dir).
- PEDM detector: `python -m examples.train_pedm_detector --scenario <Scenario-v4> --victim_run
  <victim run dir> --out results/obs_attackers/<Scenario-v4>/pedm_detector.pt` (see that
  script's own docstring).
