"""Runner registry."""
from harl.runners.on_policy_ha_runner import OnPolicyHARunner
from harl.runners.on_policy_ma_runner import OnPolicyMARunner
from harl.runners.off_policy_ha_runner import OffPolicyHARunner
from harl.runners.off_policy_ma_runner import OffPolicyMARunner
from harl.runners.adversary_runner import AdversaryRunner
from harl.runners.on_policy_lagr_runner import OnPolicyLagrRunner
from harl.runners.on_policy_alternating_lagr_runner import OnPolicyAlternatingLagrRunner

RUNNER_REGISTRY = {
    "happo": OnPolicyHARunner,
    "hatrpo": OnPolicyHARunner,
    "haa2c": OnPolicyHARunner,
    "haddpg": OffPolicyHARunner,
    "hatd3": OffPolicyHARunner,
    "hasac": OffPolicyHARunner,
    "had3qn": OffPolicyHARunner,
    "maddpg": OffPolicyMARunner,
    "matd3": OffPolicyMARunner,
    "mappo": OnPolicyMARunner,
    "illusory": AdversaryRunner,
    "mappo_lagr": OnPolicyLagrRunner,
    "mappo_hard": OnPolicyLagrRunner,
    "mappo_alt": OnPolicyAlternatingLagrRunner,
}
