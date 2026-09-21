from harl.common.base_logger import BaseLogger


class MujocoMARLLogger(BaseLogger):
    def get_task_name(self):
        return self.env_args.get("scenario", "HalfCheetah-v4")
