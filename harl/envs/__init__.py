from absl import flags
from harl.envs.smac.smac_logger import SMACLogger
from harl.envs.smacv2.smacv2_logger import SMACv2Logger
from harl.envs.mamujoco.mamujoco_logger import MAMuJoCoLogger
from harl.envs.pettingzoo_mpe.pettingzoo_mpe_logger import PettingZooMPELogger
from harl.envs.gym.gym_logger import GYMLogger
from harl.envs.football.football_logger import FootballLogger
from harl.envs.dexhands.dexhands_logger import DexHandsLogger
from harl.envs.lag.lag_logger import LAGLogger
from harl.envs.safety_marl.safety_marl_logger import SafetyMARLLogger
from harl.envs.mujoco_marl.mujoco_marl_logger import MujocoMARLLogger

FLAGS = flags.FLAGS
FLAGS(["train_sc.py"])

LOGGER_REGISTRY = {
    "smac": SMACLogger,
    "mamujoco": MAMuJoCoLogger,
    "pettingzoo_mpe": PettingZooMPELogger,
    "gym": GYMLogger,
    "football": FootballLogger,
    "dexhands": DexHandsLogger,
    "smacv2": SMACv2Logger,
    "lag": LAGLogger,
    "safety_marl": SafetyMARLLogger,
    "safety_illusory": SafetyMARLLogger,
    "mujoco_marl": MujocoMARLLogger,
}
