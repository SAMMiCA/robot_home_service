from experiments.pick_and_place.pick_and_place_rgb_il_rl_combined_base import PickAndPlaceRGBILRLCombinedBaseExperimentConfig

class PickAndPlaceRGBResNetILRLCombinedExperimentConfig(PickAndPlaceRGBILRLCombinedBaseExperimentConfig):
    USE_RESNET_CNN = True
    IL_PIPELINE_TYPE = "20proc"

    @classmethod
    def tag(cls) -> str:
        return f"PickAndPlaceRGBResNetILRLCombined_{cls.IL_PIPELINE_TYPE}"