

from typing import Dict, Optional, Sequence, Tuple, Any
import torch
import math

from allenact.base_abstractions.sensor import Sensor, SensorSuite
from allenact.algorithms.onpolicy_sync.losses.imitation import Imitation
from allenact.embodiedai.sensors.vision_sensors import DepthSensor
from allenact.utils.system import get_logger
from allenact.utils.misc_utils import all_unique
from allenact.utils.experiment_utils import PipelineStage
from env.tasks import HomeServiceTaskSampler
from env.expert_sensors import HomeServiceGreedyActionExpertSensor
from experiments.home_service_base import HomeServiceBaseExperimentConfig


class StepwiseLinearDecay:
    def __init__(self, cumm_steps_and_values: Sequence[Tuple[int, float]]):
        assert len(cumm_steps_and_values) >= 1

        self.steps_and_values = list(sorted(cumm_steps_and_values))
        self.steps = [steps for steps, _ in cumm_steps_and_values]
        self.values = [value for _, value in cumm_steps_and_values]

        assert all_unique(self.steps)
        assert all(0 <= v <= 1 for v in self.values)

    def __call__(self, epoch: int) -> float:
        """Get the value for the input number of steps."""
        if epoch <= self.steps[0]:
            return self.values[0]
        elif epoch >= self.steps[-1]:
            return self.values[-1]
        else:
            # TODO: Binary search would be more efficient but seems overkill
            for i, (s0, s1) in enumerate(zip(self.steps[:-1], self.steps[1:])):
                if epoch < s1:
                    p = (epoch - s0) / (s1 - s0)
                    v0 = self.values[i]
                    v1 = self.values[i + 1]
                    return p * v1 + (1 - p) * v0


def il_training_params(label: str, training_steps: int, square_root_scaling=False):
    num_train_processes = int(label.split("proc")[0])
    num_steps = 64
    # num_mini_batch = 2 if torch.cuda.device_count() > 0 else 1
    num_mini_batch = 1
    prop = (num_train_processes / 40) * (num_steps / 64)  # / num_mini_batch
    if not square_root_scaling:
        lr = 3e-4 * prop
    else:
        lr = 3e-4 * min(math.sqrt(prop), prop)
    update_repeats = 3
    dagger_steps = min(int(2e6), training_steps // 10)
    bc_tf1_steps = min(int(2e5), training_steps // 10)

    get_logger().info(
        f"Using {training_steps // int(1e6)}M training steps and"
        f" {dagger_steps // int(1e6)}M Dagger steps,"
        f" {bc_tf1_steps // int(1e5)}00k BC with teacher forcing=1,"
        f" {num_train_processes} processes (per machine)",
    )

    return dict(
        lr=lr,
        num_steps=num_steps,
        num_mini_batch=num_mini_batch,
        update_repeats=update_repeats,
        use_lr_decay=False,
        num_train_processes=num_train_processes,
        dagger_steps=dagger_steps,
        bc_tf1_steps=bc_tf1_steps,
    )

class HomeServiceILBaseExperimentConfig(HomeServiceBaseExperimentConfig):
    IL_PIPELINE_TYPE: Optional[str] = None
    square_root_scaling = False

    def _training_pipeline_info(self, **kwargs) -> Dict[str, Any]:
        training_steps = self.TRAINING_STEPS
        params = self._use_label_to_get_training_params()
        bc_tf1_steps = params["bc_tf1_steps"]
        dagger_steps = params["dagger_steps"]

        return dict(
            named_losses=dict(imitation_loss=Imitation()),
            pipeline_stages=[
                PipelineStage(
                    loss_names=["imitation_loss"],
                    max_stage_steps=training_steps,
                    teacher_forcing=StepwiseLinearDecay(
                        cumm_steps_and_values=[
                            (bc_tf1_steps, 1.0),
                            (bc_tf1_steps + dagger_steps, 0.0),
                        ]
                    ),
                ),
            ],
            **params,
        )

    def _use_label_to_get_training_params(self):
        return il_training_params(
            label=self.IL_PIPELINE_TYPE.lower(),
            training_steps=self.TRAINING_STEPS,
            square_root_scaling=self.square_root_scaling,
        )

    def num_train_processes(self) -> int:
        return self._use_label_to_get_training_params()["num_train_processes"]

    def num_valid_processes(self) -> int:
        return 0

    def num_test_processes(self) -> int:
        return 1