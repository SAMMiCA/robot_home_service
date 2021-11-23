from typing import (
    Optional,
    Tuple,
    Sequence,
    Union,
    Dict,
    Any,
)

import gym
import gym.spaces
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from allenact.algorithms.onpolicy_sync.policy import (
    ActorCriticModel,
    DistributionType,
    LinearActorCriticHead,
)
from allenact.algorithms.onpolicy_sync.policy import (
    LinearCriticHead,
    LinearActorHead,
    ObservationType,
)
from allenact.base_abstractions.distributions import CategoricalDistr
from allenact.base_abstractions.misc import ActorCriticOutput, Memory
from allenact.embodiedai.mapping.mapping_models.active_neural_slam import (
    ActiveNeuralSLAM,
)
from allenact.embodiedai.models.basic_models import SimpleCNN, RNNStateEncoder
from allenact.utils.misc_utils import prepare_locals_for_super
from allenact.utils.model_utils import simple_conv_and_linear_weights_init


class HomeServiceActorCriticSimpleConvRNN(ActorCriticModel[CategoricalDistr]):
    """A CNN->RNN actor-critic model for rearrangement tasks."""

    def __init__(
        self,
        action_space: gym.spaces.Discrete,
        observation_space: gym.spaces.Dict,
        rgb_uuid: str,
        subtask_uuid: str,
        rel_position_change_uuid: str,
        ordered_object_types: Sequence[str],
        hidden_size=512,
        task_type_embedding_dim: int = 32,
        object_type_embedding_dim: int = 128,
        object_visible_embedding_dim: int = 16,
        position_dim: int = 32,
        num_rnn_layers=1,
        rnn_type="GRU",
    ):
        """
        # Parameters
        action_space : The action space of the agent.
            Should equal `gym.spaces.Discrete(# actions available to the agent)`.
        observation_space : The observation space available to the agent.
        rgb_uuid : The unique id of the RGB image sensor (see `RGBSensor`).
        unshuffled_rgb_uuid : The unique id of the `UnshuffledRGBRearrangeSensor` available to the agent.
        hidden_size : The size of the hidden layer of the RNN.
        num_rnn_layers: The number of hidden layers in the RNN.
        rnn_type : The RNN type, should be "GRU" or "LSTM".
        """
        super().__init__(action_space=action_space, observation_space=observation_space)
        self._hidden_size = hidden_size

        self.rgb_uuid = rgb_uuid
        self.subtask_uuid = subtask_uuid
        self.rel_position_change_uuid = rel_position_change_uuid

        self.task_type_embedder = nn.Embedding(
            num_embeddings=8, embedding_dim=task_type_embedding_dim
        )
        self.object_type_embedder = nn.Embedding(
            num_embeddings=len(ordered_object_types) + 1, embedding_dim=object_type_embedding_dim,
        )
        self.object_visible_embedder = nn.Embedding(
            num_embeddings=3, embedding_dim=object_visible_embedding_dim,
        )
        self.position_encoder = nn.Linear(11, position_dim)

        self.visual_encoder = self._create_visual_encoder()

        self.state_encoder = RNNStateEncoder(
            task_type_embedding_dim
            + object_type_embedding_dim * 2
            + object_visible_embedding_dim * 2
            + position_dim
            + self.recurrent_hidden_state_size,
            self._hidden_size,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
        )

        self.actor = LinearActorHead(self._hidden_size, action_space.n)
        self.critic = LinearCriticHead(self._hidden_size)

        self.train()

    def _create_visual_encoder(self) -> nn.Module:
        """Create the visual encoder for the model."""
        return SimpleCNN(
            observation_space=gym.spaces.Dict(
                {
                    self.rgb_uuid: self.observation_space[self.rgb_uuid]
                }
            ),
            output_size=self._hidden_size,
            rgb_uuid=self.rgb_uuid,
            depth_uuid=None,
        )

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    @property
    def recurrent_hidden_state_size(self):
        return self._hidden_size

    def _recurrent_memory_specification(self):
        return dict(
            rnn=(
                (
                    ("layer", self.num_recurrent_layers),
                    ("sampler", None),
                    ("hidden", self.recurrent_hidden_state_size),
                ),
                torch.float32,
            )
        )

    def forward(  # type:ignore
        self,
        observations: ObservationType,
        memory: Memory,
        prev_actions: torch.Tensor,
        masks: torch.FloatTensor,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        cur_img = observations[self.rgb_uuid]
        subtask = observations[self.subtask_uuid]
        import pdb; pdb.set_trace()

        x = self.visual_encoder({self.rgb_uuid: cur_img})
        x, rnn_hidden_states = self.state_encoder(x, memory.tensor("rnn"), masks)

        ac_output = ActorCriticOutput(
            distributions=self.actor(x), values=self.critic(x), extras={}
        )

        return ac_output, memory.set_tensor("rnn", rnn_hidden_states)


class HomeServiceResNetActorCriticRNN(HomeServiceActorCriticSimpleConvRNN):
    def __init__(
        self,
        action_space: gym.spaces.Discrete,
        observation_space: gym.spaces.Dict,
        rgb_uuid: str,
        subtask_uuid: str,
        rel_position_change_uuid: str,
        ordered_object_types: Sequence[str],
        hidden_size=512,
        task_type_embedding_dim: int = 32,
        object_type_embedding_dim: int = 128,
        object_visible_embedding_dim: int = 16,
        position_dim: int = 32,
        num_rnn_layers=1,
        rnn_type="GRU",
    ):
        """A CNN->RNN rearrangement model that expects ResNet features instead
        of RGB images.

        Nearly identical to `RearrangeActorCriticSimpleConvRNN` but
        `rgb_uuid` should now be the unique id of the ResNetPreprocessor
        used to featurize RGB images using a pretrained ResNet before
        they're passed to this model.
        """
        self.visual_attention: Optional[nn.Module] = None
        super().__init__(**prepare_locals_for_super(locals()))

    def _create_visual_encoder(self) -> nn.Module:
        a = self.observation_space[self.rgb_uuid].shape[0]
        self.visual_attention = nn.Sequential(
            nn.Conv2d(a, 32, 1,), nn.ReLU(inplace=True), nn.Conv2d(32, 1, 1,),
        )
        visual_encoder = nn.Sequential(
            nn.Conv2d(a, self._hidden_size, 1,), nn.ReLU(inplace=True),
        )
        self.visual_attention.apply(simple_conv_and_linear_weights_init)
        visual_encoder.apply(simple_conv_and_linear_weights_init)

        return visual_encoder

    def forward(  # type:ignore
        self,
        observations: ObservationType,
        memory: Memory,
        prev_actions: torch.Tensor,
        masks: torch.FloatTensor,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        cur_img_resnet = observations[self.rgb_uuid]
        batch_shape, features_shape = cur_img_resnet.shape[:-3], cur_img_resnet.shape[-3:]
        cur_img_resnet_reshaped = cur_img_resnet.view(-1, *features_shape)
        attention_probs = torch.softmax(
            self.visual_attention(cur_img_resnet_reshaped).view(
                cur_img_resnet_reshaped.shape[0], -1
            ),
            dim=-1,
        ).view(cur_img_resnet_reshaped.shape[0], 1, *cur_img_resnet_reshaped.shape[-2:])
        
        vis_x = (
            (self.visual_encoder(cur_img_resnet_reshaped) * attention_probs)
            .mean(-1)
            .mean(-1)
        )
        vis_x = vis_x.view(*batch_shape, -1)

        subtask = observations[self.subtask_uuid]
        task_type_x = self.task_type_embedder(subtask['type'])
        target_type_x = self.object_type_embedder(subtask['target_type'])
        target_visible_x = self.object_visible_embedder(subtask['target_visible'])
        place_type_x = self.object_type_embedder(subtask['place_type'])
        place_visible_x = self.object_visible_embedder(subtask['place_visible'])
        position_x = torch.cat([observations['rel_position_change']['agent_locs'], subtask['target_position'], subtask['place_position']], -1)
        position_x = self.position_encoder(position_x)
        x = torch.cat([vis_x, task_type_x, target_type_x, target_visible_x, place_type_x, place_visible_x, position_x], -1)
        # print(f'x.shape: {x.shape}')

        x, rnn_hidden_states = self.state_encoder(x, memory.tensor("rnn"), masks)

        ac_output = ActorCriticOutput(
            distributions=self.actor(x), values=self.critic(x), extras={}
        )

        return ac_output, memory.set_tensor("rnn", rnn_hidden_states)
