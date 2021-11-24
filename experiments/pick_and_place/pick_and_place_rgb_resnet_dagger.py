from experiments.pick_and_place.pick_and_place_rgb_il_base import PickAndPlaceRGBILBaseExperimentConfig


class PickAndPlaceRGBResNetDaggerExperimentConfig(PickAndPlaceRGBILBaseExperimentConfig):
    USE_RESNET_CNN = True
    IL_PIPELINE_TYPE = "8proc"

    @classmethod
    def tag(cls) -> str:
        return f"PickAndPlaceRGBResNetDagger_{cls.IL_PIPELINE_TYPE}"