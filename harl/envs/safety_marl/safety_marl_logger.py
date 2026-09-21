from harl.common.base_logger import BaseLogger


class SafetyMARLLogger(BaseLogger):
    def get_task_name(self):
        return self.env_args.get("scenario", "SafetyPointGoal1Gymnasium-v0")
