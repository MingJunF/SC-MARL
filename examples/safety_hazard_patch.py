"""Monkeypatch for safety_gymnasium's Hazards geom (2026-09-12, extended 2026-09-14), per
explicit user request:

  1. Each hazard's radius is now independently RANDOM per episode (drawn uniformly from
     [--hazard_size_min, --hazard_size_max], default [0.1, 0.35], centered on the library's
     original fixed 0.2) instead of one fixed size shared by every hazard -- the agent must
     learn to judge each hazard's actual extent (e.g. from lidar) rather than memorize a
     constant boundary.
  2. Cost is now FLAT per violating step: `cost_hazards += self.cost` (1.0) whenever the agent
     is within a hazard's (now-random) radius, instead of the library's original
     depth-proportional `self.cost * (size - h_dist)`. Stepping on the very edge of a hazard
     costs the same as standing at its center.
  3. (2026-09-14) Only `NUM_HAZARDS=1` hazard is actually placed, down from SafetyPointGoal1's
     built-in 8 -- per explicit user request ("只需要一个hazard，不需要那么多hazard"). The task's
     `GoalLevel1.__init__` hard-codes `Hazards(num=8, keepout=0.18)`, so a passed-in kwarg can't
     be overridden by changing the class's dataclass-field default (Python already bound the
     literal 8 at that call site) -- instead `Hazards.__init__` is wrapped to force `self.num`
     back down to 1 right after the original constructor runs, before anything else (lidar
     observation sizing, placement-dict building, world config) reads `self.num`.

Does NOT edit the installed `safety_gymnasium` package -- this module monkeypatches the
`Hazards` class's `__init__`/`process_config`/`get_config`/`cal_cost` methods at import time.
Every script that creates a SafetyPointGoal1/etc. env must `import examples.safety_hazard_patch`
(for its side effect) BEFORE calling `gym.make(...)`, exactly like `import safety_gymnasium`
itself is already required for env registration.

Usage: import examples.safety_hazard_patch  # noqa: F401  (apply once, before gym.make)
"""
import numpy as np
from safety_gymnasium.assets.geoms.hazards import Hazards

HAZARD_SIZE_MIN = 0.40
HAZARD_SIZE_MAX = 0.40
# Reverted back from [0.15,0.45] (2026-09-16): the bigger range was tried alongside a
# --time_penalty (later abandoned) and then a --speed_bonus_scale reward redesign, but BOTH
# attempts under the bigger hazard failed to converge (dual-ascent lambda never plateaued,
# deterministic eval got WORSE over training: v_ret -22->-40, v_cost 3.5->6.3) -- per explicit
# user diagnosis ("那就是hazard过大的问题，改回之前的能训练出来的版本再跑跑看"), reverted to this
# original range (the one config that DID converge cleanly: lambda plateaued at 6.497, v_ret
# 18.35, v_cost 0.90) to retest the speed-bonus reward design in isolation, without the bigger
# hazard as a confound.
NUM_HAZARDS = 1

_orig_init = Hazards.__init__


def _patched_init(self, *args, **kwargs):
    _orig_init(self, *args, **kwargs)
    self.num = NUM_HAZARDS


def _process_config(self, config, layout, rots):
    """Same loop as the original, but draws one independent random size per hazard first
    (stored on `self` as `hazard_sizes`, read by the patched `get_config`/`cal_cost` below) --
    replaces the single shared `self.size` for this call's world build."""
    if hasattr(self, "num"):
        assert len(rots) == self.num, "The number of rotations should be equal to the number of obstacles."
        rng = getattr(self, "random_generator", None)
        if rng is not None and hasattr(rng, "uniform"):
            self.hazard_sizes = [float(rng.uniform(HAZARD_SIZE_MIN, HAZARD_SIZE_MAX)) for _ in range(self.num)]
        else:
            self.hazard_sizes = list(np.random.uniform(HAZARD_SIZE_MIN, HAZARD_SIZE_MAX, size=self.num))
        for i in range(self.num):
            name = f"{self.name[:-1]}{i}"
            self._cur_hazard_size = self.hazard_sizes[i]
            config[self.type][name] = self.get_config(xy_pos=layout[name], rot=rots[i])
            config[self.type][name].update({"name": name})
    else:
        assert len(rots) == 1, "The number of rotations should be 1."
        config[self.type][self.name] = self.get_config(xy_pos=layout[self.name], rot=rots[0])


def _get_config(self, xy_pos, rot):
    """Identical to the original except the geom's radius comes from `self._cur_hazard_size`
    (this call's randomly-drawn size, set by `_process_config` right before calling this) --
    so each hazard is actually built at its own random size, not just scored differently."""
    size = getattr(self, "_cur_hazard_size", self.size)
    geom = {
        "name": self.name,
        "size": [size, 1e-2],
        "pos": np.r_[xy_pos, 2e-2],
        "rot": rot,
        "type": "cylinder",
        "contype": 0,
        "conaffinity": 0,
        "group": self.group,
        "rgba": self.color,
    }
    if self.is_meshed:
        geom.update({"type": "mesh", "mesh": "bush", "material": "bush", "euler": [np.pi / 2, 0, 0]})
    return geom


def _cal_cost(self):
    """Flat per-step cost (self.cost, default 1.0) whenever inside a hazard's OWN random
    radius -- replaces the original depth-proportional `self.cost * (size - h_dist)`."""
    cost = {}
    if not self.is_constrained:
        return cost
    cost["cost_hazards"] = 0
    sizes = getattr(self, "hazard_sizes", [self.size] * len(self.pos))
    for h_pos, size in zip(self.pos, sizes):
        h_dist = self.agent.dist_xy(h_pos)
        if h_dist <= size:
            cost["cost_hazards"] += self.cost
    return cost


Hazards.__init__ = _patched_init
Hazards.process_config = _process_config
Hazards.get_config = _get_config
Hazards.cal_cost = _cal_cost
