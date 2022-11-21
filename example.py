import copy
from PIL import Image
import numpy as np
import os

from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
import prior

from env.tasks import HomeServiceTaskSampler
from experiments.home_service_base import HomeServiceBaseExperimentConfig

# Load procthor-10k dataset
# dataset = prior.load_dataset('procthor-10k', revision="rearrangement-2022")
# dataset = prior.load_dataset('procthor-10k')


task_sampler_params = HomeServiceBaseExperimentConfig.stagewise_task_sampler_args(
    stage="train_seen", process_ind=0, total_processes=1, headless=False
)
