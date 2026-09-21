"""Monkeypatch for safety_gymnasium's placement sampling (2026-09-14, extended 2026-09-15 per
follow-up user requests), forcing a fixed spatial LAYOUT instead of fully-random placement:

  1. Agent always spawns in the BOTTOM-LEFT corner region.
  2. Goal ALTERNATES deterministically between the BOTTOM-LEFT and TOP-RIGHT corners based on the
     agent's CURRENT LIVE position (`task.agent.pos`, read straight from the mujoco sim, not a
     frozen spawn record) -- per explicit user correction of an earlier, over-complicated attempt
     at this ("应该是先robot和目标分别出现在地图对角...新目标继续生成在对角，正常第二次应该是生成在robot出生
     点附近"): the new goal is always placed in whichever of the two corners is FARTHER from
     where the agent physically is right now. The very first goal (agent still at its bottom-left
     spawn point) is therefore always top-right; once the agent reaches it and a new goal is
     needed (`continue_goal=True`, set in the 4 training/eval scripts, not here), the agent is now
     near top-right, so the new goal deterministically goes back to bottom-left, and so on --
     genuine back-and-forth shuttling, not a random coin flip (an earlier version of this patch
     tried a 50/50 random choice and got tangled in a subtle survivorship-bias skew from hazard
     placement failures silently discarding and re-drawing bottom-left-goal attempts more often;
     abandoned that approach entirely once the user pointed out the actual desired design is
     simpler and deterministic).
  3. The single hazard (see `safety_hazard_patch.py`'s NUM_HAZARDS=1) is NOT placed by the
     generic region-based sampler -- its (x,y) is computed from that EPISODE'S initial sampled
     agent/goal positions: a random point along the segment between them (fraction range widened
     to 25%-75%, from the original 35%-65%) with a random PER-EPISODE perpendicular jitter
     (0.05-0.30, up from a fixed 0.05) -- per explicit user request ("hazardregion有点小了，可以大
     一点，但是每个episode大小也是不要太固定，要有个方位"). Widening the jitter means the hazard no
     longer ALWAYS overlaps the exact straight-line path (jitter can now exceed the hazard's own
     min radius 0.10) -- it's a looser, more varied "roughly in the way" placement rather than a
     hard geometric guarantee. NOTE: this hazard placement is only computed ONCE at episode reset
     from the FIRST agent/goal pair -- it is NOT recomputed on later shuttle-goal respawns (each
     of which goes through `build_goal_position`/`sample_goal_position`, untouched by the hazard
     logic here), so the hazard may not sit on later shuttle legs' paths.
  4. The single decorative vase is left AS-IS (attempted forcing `Vases.num=0` per "可以不要有
     其他障碍物" but that crashed lidar observation building -- `_obs_lidar_pseudo` assumes at
     least one position exists once an obstacle is registered as lidar-observed; safe_gymnasium
     doesn't cleanly support a zero-count free-geom this way). Since the vase carries NO cost
     (`Vases(num=1, is_constrained=False)`, confirmed by reading `goal_level1.py`) it doesn't
     affect training either way, so this was left alone rather than risk a broken env.

Why this needs a `RandomGenerator.sample_layout` override (not just the placement-REGION
override in `_patched_placements_dict_from_object` below): region-based sampling for the
hazard can only constrain it to a box, not to the line between two OTHER objects' actual
per-episode sampled positions -- that requires reading `agent`'s and `goal`'s already-placed
(x,y) mid-way through the SAME sampling loop, which only `sample_layout`'s own local `layout`
dict has access to (`self.layout` is not updated until the whole loop succeeds).

Why the goal-alternation logic ALSO needs a `BaseTask.build_goal_position` override (not just
`RandomGenerator`): only the TASK object has `self.agent.pos` (the live mujoco position) --
`RandomGenerator` only knows about placement regions and an RNG, it has no simulator handle.

Does NOT edit the installed `safety_gymnasium` package -- patches `BaseTask.
_placements_dict_from_object`, `BaseTask.build_goal_position`, and `RandomGenerator.
sample_layout` at import time. Every script that creates a SafetyPointGoal1/etc. env must
`import examples.safety_layout_patch` (for its side effect), after `examples.safety_hazard_patch`
and before calling `gym.make(...)`.

Usage: import examples.safety_layout_patch  # noqa: F401
"""
import numpy as np
import mujoco
from safety_gymnasium.bases.base_task import BaseTask
from safety_gymnasium.utils.common_utils import ResamplingError
from safety_gymnasium.utils.random_generator import RandomGenerator

AGENT_REGION = [(-1.5, -1.5, -0.6, -0.6)]
BOTTOM_LEFT_CENTER = np.array([-1.05, -1.05])
TOP_RIGHT_CENTER = np.array([1.05, 1.05])
# The goal's bottom-left option must NOT be the same tight box as AGENT_REGION: agent's
# keepout=0.4 shrinks its 0.9-wide region down to a ~0.1-wide sliver right in the corner
# (empirically ~(-1.1,-1.0)), and goal+agent need >=0.705 separation (0.4+0.305 keepouts) -- a
# same-sized box in the exact same corner can never satisfy that. Extending the bottom-left
# option most of the way to the arena center gives it enough area far enough from the agent's
# tiny corner sliver for placement to actually succeed there.
BOTTOM_LEFT_REGION = (-1.5, -1.5, -0.1, -0.1)
TOP_RIGHT_REGION = (0.6, 0.6, 1.5, 1.5)
HAZARD_REGION = [(-0.9, -0.9, 0.9, 0.9)]  # fallback only, used if agent/goal aren't placed yet
# Vase banished far off the play area (2026-09-16, per explicit user request "只保留hazard，其他的
#障碍物什么的都去掉"): NOT set to num=0 -- that crashed lidar observation building (documented
# above), so instead placed in a tiny fixed region way outside the ~1.5-unit arena where it can
# never be reached or physically block the agent, without touching Vases.num.
VASE_REGION = [(4.5, 4.5, 5.5, 5.5)]  # wide enough to clear the vase's own 0.15 keepout margin
HAZARD_PATH_FRAC_LO = 0.4
HAZARD_PATH_FRAC_HI = 0.6
HAZARD_PATH_JITTER_MIN = 0.05
HAZARD_PATH_JITTER_MAX = 0.30  # perpendicular offset now randomized per episode in [MIN, MAX]


def _farther_corner(agent_xy):
    """Whichever of the two corner regions is farther from the agent's given (x,y)."""
    d_bl = np.linalg.norm(agent_xy - BOTTOM_LEFT_CENTER)
    d_tr = np.linalg.norm(agent_xy - TOP_RIGHT_CENTER)
    return BOTTOM_LEFT_REGION if d_bl > d_tr else TOP_RIGHT_REGION


_orig_placements_dict_from_object = BaseTask._placements_dict_from_object


def _patched_placements_dict_from_object(self, object_name):
    if object_name == 'agent':
        keepout = self.agent.keepout
        return {'agent': (AGENT_REGION, keepout)}
    if object_name == 'goal':
        keepout = self.goal.keepout
        # Placeholder region for the very first build_placements_dict call -- immediately
        # overwritten per-episode by `_patched_sample_layout` (first goal) and
        # `_patched_build_goal_position` (every subsequent shuttle goal) before it's ever
        # actually sampled from, since both recompute the correct single corner dynamically.
        return {'goal': ([TOP_RIGHT_REGION], keepout)}
    if object_name == 'hazards':
        keepout = self.hazards.keepout
        return {f'hazard{i}': (HAZARD_REGION, keepout) for i in range(self.hazards.num)}
    if object_name == 'vases':
        keepout = self.vases.keepout
        return {f'vase{i}': (VASE_REGION, keepout) for i in range(self.vases.num)}
    return _orig_placements_dict_from_object(self, object_name)


BaseTask._placements_dict_from_object = _patched_placements_dict_from_object


def _patched_sample_layout(self):
    """Same structure as the original, except: `goal`'s region is forced to whichever corner is
    farther from the just-placed `agent` (always top-right, since agent always starts
    bottom-left); `hazard0`'s (x,y) is computed from the already-sampled `agent`/`goal` positions
    in THIS loop's local `layout` dict, instead of being drawn independently from its own
    placement region."""

    def placement_is_valid(xy, layout, keepout):
        for other_name, other_xy in layout.items():
            other_keepout = self.placements[other_name][1]
            dist = np.sqrt(np.sum(np.square(xy - other_xy)))
            if dist < other_keepout + self.placements_margin + keepout:
                return False
        return True

    layout = {}
    for name, (placements, keepout) in self.placements.items():
        if name == 'goal' and 'agent' in layout:
            placements = [_farther_corner(layout['agent'])]

        if name == 'hazard0' and 'agent' in layout and 'goal' in layout:
            agent_xy, goal_xy = layout['agent'], layout['goal']
            d = goal_xy - agent_xy
            direction = d / (np.linalg.norm(d) + 1e-9)
            normal = np.array([-direction[1], direction[0]])
            jitter = self.random_generator.uniform(HAZARD_PATH_JITTER_MIN, HAZARD_PATH_JITTER_MAX)
            conflicted = True
            for _ in range(2000):
                frac = self.random_generator.uniform(HAZARD_PATH_FRAC_LO, HAZARD_PATH_FRAC_HI)
                perp = self.random_generator.uniform(-jitter, jitter)
                xy = agent_xy + frac * d + perp * normal
                if placement_is_valid(xy, layout, keepout):
                    conflicted = False
                    break
            if conflicted:
                return False
            layout[name] = xy
            continue

        conflicted = True
        for _ in range(2000):
            xy = self.draw_placement(placements, keepout)
            if placement_is_valid(xy, layout, keepout):
                conflicted = False
                break
        if conflicted:
            return False
        layout[name] = xy
    self.layout = layout
    return True


RandomGenerator.sample_layout = _patched_sample_layout


def _patched_build_goal_position(self) -> None:
    """Replaces the original (base_task.py): restricts the goal's placement region to whichever
    corner is farther from the agent's CURRENT LIVE position (`self.agent.pos`, read straight
    from the mujoco sim) before resampling, instead of drawing from both corners. This is what
    makes every subsequent `continue_goal` respawn alternate back to the corner the agent just
    came from, instead of picking a fresh random corner each time."""
    if 'goal' in self.world_info.layout:
        del self.world_info.layout['goal']

    keepout = self.placements_conf.placements['goal'][1]
    corner = _farther_corner(self.agent.pos[:2])
    self.placements_conf.placements['goal'] = ([corner], keepout)

    for _ in range(10000):
        if self.random_generator.sample_goal_position():
            break
    else:
        raise ResamplingError('Failed to generate goal')
    self.world_info.world_config_dict['geoms']['goal']['pos'][:2] = self.world_info.layout['goal']
    self._set_goal(self.world_info.layout['goal'])
    mujoco.mj_forward(self.model, self.data)


BaseTask.build_goal_position = _patched_build_goal_position
