import torch
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import CnnPolicy, QNetwork
from stable_baselines3.common.torch_layers import FlattenExtractor
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecTransposeImage, VecFrameStack

class CustomCnnPolicy(CnnPolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        net_arch = [512,128,64],
        activation_fn = torch.nn.ReLU,
        features_extractor_class= FlattenExtractor,
        features_extractor_kwargs = None,
        normalize_images = True,
        optimizer_class = torch.optim.Adam,
        optimizer_kwargs = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )


    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the network and the optimizer.

        Put the target network into evaluation mode.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """

        self.q_net = self.make_q_net()
        self.q_net_target = self.make_q_net()
        self.q_net_target.load_state_dict(self.q_net.state_dict())
        self.q_net_target.set_training_mode(False)

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(  # type: ignore[call-arg]
            self.q_net.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )
    def make_q_net(self):
        # Make sure we always have separate networks for features extractors etc
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return QNetwork(**net_args).to(self.device)


if __name__=="__main__":
    env_str = "CarRacing-v3"
    log_dir = f"mask_logs/{env_str}"
    env_kwargs_dict={"continuous": False, "render_mode": "rgb_array"}
    n_stack = 4

    env = make_vec_env(env_str,
                    n_envs=1,
                    env_kwargs=env_kwargs_dict)

    env = VecFrameStack(env, n_stack=n_stack)
    env = VecTransposeImage(env)
    policy_kwargs = dict(
        net_arch=[900,64]
    )

    model = DQN(CustomCnnPolicy, env, policy_kwargs=policy_kwargs, verbose=0)
    print(model.q_net)