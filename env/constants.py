import os
from pathlib import Path

VISIBILITY_DISTANCE = 1.5
MAX_HAND_METERS = 0.5
FOV = 90

IOU_THRESHOLD = 0.5
OPENNESS_THRESHOLD = 0.2
POSITION_DIFF_BARRIER = 2.0

REQUIRED_THOR_VERSION = "4.2.0"
STARTER_HOME_SERVICE_DATA_DIR = os.path.join(
    os.path.abspath(os.path.dirname(Path(__file__))), "../data/home_service"
)

# Commit ID for AI2THOR-Rearrangement Challenge
THOR_COMMIT_ID = "eb93d0b6520e567bac8ad630462b5c0c4cea1f5f"
PROCTHOR_COMMIT_ID = "90eac925dc750818890069e3131f899998dc58b4"
# PROCTHOR_COMMIT_ID = "9561884d24ec7167a42b577e83bf5703c5d4584d"
# PROCTHOR_COMMIT_ID = "391b3fae4d4cc026f1522e5acf60953560235971"

STEP_SIZE = 0.25
ROTATION_ANGLE = 90
HORIZON_ANGLE = 30
SMOOTHING_FACTOR = 1

SCENE_TYPE_TO_LABEL = {
    "Kitchen": 1,
    "LivingRoom": 2,
    "Bedroom": 3,
    "Bathroom": 4,
}

# fmt: off
REARRANGE_SIM_OBJECTS = [
    # A
    "AlarmClock", "AluminumFoil", "Apple", "AppleSliced", "ArmChair",
    "BaseballBat", "BasketBall", "Bathtub", "BathtubBasin", "Bed", "Blinds", "Book", "Boots", "Bottle", "Bowl", "Box",
    # B
    "Bread", "BreadSliced", "ButterKnife",
    # C
    "Cabinet", "Candle", "CD", "CellPhone", "Chair", "Cloth", "CoffeeMachine", "CoffeeTable", "CounterTop", "CreditCard",
    "Cup", "Curtains",
    # D
    "Desk", "DeskLamp", "Desktop", "DiningTable", "DishSponge", "DogBed", "Drawer", "Dresser", "Dumbbell",
    # E
    "Egg", "EggCracked",
    # F
    "Faucet", "Floor", "FloorLamp", "Footstool", "Fork", "Fridge",
    # G
    "GarbageBag", "GarbageCan",
    # H
    "HandTowel", "HandTowelHolder", "HousePlant", "Kettle", "KeyChain", "Knife",
    # L
    "Ladle", "Laptop", "LaundryHamper", "Lettuce", "LettuceSliced", "LightSwitch",
    # M
    "Microwave", "Mirror", "Mug",
    # N
    "Newspaper",
    # O
    "Ottoman",
    # P
    "Painting", "Pan", "PaperTowel", "Pen", "Pencil", "PepperShaker", "Pillow", "Plate", "Plunger", "Poster", "Pot",
    "Potato", "PotatoSliced",
    # R
    "RemoteControl", "RoomDecor",
    # S
    "Safe", "SaltShaker", "ScrubBrush", "Shelf", "ShelvingUnit", "ShowerCurtain", "ShowerDoor", "ShowerGlass",
    "ShowerHead", "SideTable", "Sink", "SinkBasin", "SoapBar", "SoapBottle", "Sofa", "Spatula", "Spoon", "SprayBottle",
    "Statue", "Stool", "StoveBurner", "StoveKnob",
    # T
    "TableTopDecor", "TargetCircle", "TeddyBear", "Television", "TennisRacket", "TissueBox", "Toaster", "Toilet",
    "ToiletPaper", "ToiletPaperHanger", "Tomato", "TomatoSliced", "Towel", "TowelHolder", "TVStand",
    # V
    "VacuumCleaner", "Vase",
    # W
    "Watch", "WateringCan", "Window", "WineBottle",
]
# fmt: on
DEFAULT_COMPATIBLE_RECEPTACLES = {
    "AlarmClock": ["Box", "Dresser", "Desk", "SideTable", "DiningTable", "TVStand", "CoffeeTable", "CounterTop", "Shelf"], 
    "Apple": ["Pot", "Pan", "Bowl", "Microwave", "Fridge", "Plate", "Sink", "SinkBasin", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "Desk", "CounterTop", "GarbageCan", "Dresser"], 
    # "AppleSliced": ["Pot", "Pan", "Bowl", "Microwave", "Fridge", "Plate", "Sink", "SinkBasin", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "Desk", "CounterTop", "GarbageCan", "Dresser"], 
    "BaseballBat": ["Bed", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "Desk", "CounterTop"], 
    "BasketBall": ["Sofa", "ArmChair", "Dresser", "Desk", "Bed", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop"], 
    "Book": ["Sofa", "ArmChair", "Box", "Ottoman", "Dresser", "Desk", "Bed", "Cabinet", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Shelf", "Drawer"], 
    "Bottle": ["Fridge", "Box", "Dresser", "Desk", "Sink", "SinkBasin", "Cabinet", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Shelf", "GarbageCan"], 
    "Bowl": ["Microwave", "Fridge", "Dresser", "Desk", "Sink", "SinkBasin", "Cabinet", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Shelf"], 
    "Box": ["Sofa", "ArmChair", "Dresser", "Desk", "Cabinet", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Shelf", "Ottoman"], 
    "Bread": ["Microwave", "Fridge", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "Desk", "CounterTop", "GarbageCan", "Plate"], 
    # "BreadSliced": ["Microwave", "Fridge", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "Desk", "CounterTop", "GarbageCan", "Toaster", "Plate"], 
    "ButterKnife": ["Pot", "Pan", "Bowl", "Mug", "Plate", "Cup", "Sink", "SinkBasin", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "Desk", "CounterTop", "Drawer"], 
    "Candle": ["Box", "Dresser", "Desk", "Toilet", "Cart", "Bathtub", "Cabinet", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Shelf", "Drawer"], 
    "CD": ["Box", "Ottoman", "Dresser", "Desk", "Cabinet", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Shelf", "Drawer", "GarbageCan", "Safe", "Sofa", "ArmChair"], 
    "CellPhone": ["Sofa", "ArmChair", "Box", "Ottoman", "Dresser", "Desk", "Bed", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Shelf", "Drawer", "Safe"], 
    "Cloth": ["Sofa", "ArmChair", "Box", "Ottoman", "Dresser", "LaundryHamper", "Desk", "Toilet", "Cart", "BathtubBasin", "Bathtub", "Sink", "SinkBasin", "Cabinet", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Shelf", "Drawer", "GarbageCan"], 
    "CreditCard": ["Sofa", "ArmChair", "Box", "Ottoman", "Dresser", "Desk", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Shelf", "Drawer"], 
    "Cup": ["Microwave", "Fridge", "Dresser", "Desk", "Sink", "SinkBasin", "Cabinet", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Shelf"], 
    "DishSponge": ["Pot", "Pan", "Bowl", "Plate", "Box", "Toilet", "Cart", "Cart", "BathtubBasin", "Bathtub", "Sink", "SinkBasin", "Cabinet", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Shelf", "Drawer", "GarbageCan"], 
    "Egg": ["Pot", "Pan", "Bowl", "Microwave", "Fridge", "Plate", "Sink", "SinkBasin", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "GarbageCan"], 
    "EggCracked": ["Pot", "Pan", "Bowl", "Microwave", "Fridge", "Plate", "Sink", "SinkBasin", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "GarbageCan"], 
    "Fork": ["Pot", "Pan", "Bowl", "Mug", "Plate", "Cup", "Sink", "SinkBasin", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Drawer"], 
    "HandTowel": ["HandTowelHolder"], 
    "Kettle": ["DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Sink", "SinkBasin", "Cabinet", "StoveBurner", "Shelf"], 
    "KeyChain": ["Sofa", "ArmChair", "Box", "Ottoman", "Dresser", "Desk", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Shelf", "Drawer", "Safe"], 
    "Knife": ["Pot", "Pan", "Bowl", "Mug", "Plate", "Sink", "SinkBasin", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Drawer"], 
    "Ladle": ["Pot", "Pan", "Bowl", "Plate", "Sink", "SinkBasin", "Cabinet", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Drawer"], 
    "Laptop": ["Sofa", "ArmChair", "Ottoman", "Dresser", "Desk", "Bed", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop"], 
    "Lettuce": ["Pot", "Pan", "Bowl", "Fridge", "Plate", "Sink", "SinkBasin", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "GarbageCan"], 
    # "LettuceSliced": ["Pot", "Pan", "Bowl", "Fridge", "Plate", "Sink", "SinkBasin", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "GarbageCan"], 
    "Mug": ["SinkBasin", "Cabinet", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Shelf"], 
    "Newspaper": ["Sofa", "ArmChair", "Ottoman", "Dresser", "Desk", "Bed", "Toilet", "Cabinet", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Shelf", "Drawer", "GarbageCan"], 
    "Pan": ["DiningTable", "CounterTop", "TVStand", "CoffeeTable", "SideTable", "Sink", "SinkBasin", "Cabinet", "StoveBurner", "Fridge"], 
    "Pen": ["Mug", "Box", "Dresser", "Desk", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Shelf", "Drawer", "GarbageCan"], 
    "Pencil": ["Mug", "Box", "Dresser", "Desk", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Shelf", "Drawer", "GarbageCan"], 
    "PepperShaker": ["DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Drawer", "Cabinet", "Shelf"], 
    "Pillow": ["Sofa", "ArmChair", "Ottoman", "Bed"], 
    "Plate": ["Microwave", "Fridge", "Dresser", "Desk", "Sink", "SinkBasin", "Cabinet", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Shelf"], 
    "Plunger": ["Cart", "Cabinet"], 
    "Pot": ["StoveBurner", "Fridge", "Sink", "SinkBasin", "Cabinet", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Shelf"], 
    "Potato": ["Pot", "Pan", "Bowl", "Microwave", "Fridge", "Plate", "Sink", "SinkBasin", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "GarbageCan"], 
    # "PotatoSliced": ["Pot", "Pan", "Bowl", "Microwave", "Fridge", "Plate", "Sink", "SinkBasin", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "GarbageCan"], 
    "RemoteControl": ["Sofa", "ArmChair", "Box", "Ottoman", "Dresser", "Desk", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Shelf", "Drawer"], 
    "SaltShaker": ["DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Drawer", "Cabinet", "Shelf"], 
    "SoapBar": ["Toilet", "Cart", "Bathtub", "BathtubBasin", "Sink", "SinkBasin", "Cabinet", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Shelf", "Drawer", "GarbageCan"], 
    "SoapBottle": ["Dresser", "Desk", "Toilet", "Cart", "Bathtub", "Sink", "Cabinet", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Shelf", "Drawer", "GarbageCan"], 
    "Spatula": ["Pot", "Pan", "Bowl", "Plate", "Sink", "SinkBasin", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Drawer"], 
    "Spoon": ["Pot", "Pan", "Bowl", "Mug", "Plate", "Cup", "Sink", "SinkBasin", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Drawer"], 
    "SprayBottle": ["Dresser", "Desk", "Toilet", "Cart", "Cabinet", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Shelf", "Drawer", "GarbageCan"], 
    "Statue": ["Box", "Dresser", "Desk", "Cart", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Shelf", "Safe"], 
    "TeddyBear": ["Bed", "Sofa", "ArmChair", "Ottoman", "Dresser", "Desk", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Safe"], 
    "TennisRacket": ["Dresser", "Desk", "Bed", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop"], 
    "TissueBox": ["Box", "Dresser", "Desk", "Toilet", "Cart", "Cabinet", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Shelf", "Drawer", "GarbageCan"], 
    "ToiletPaper": ["Dresser", "Desk", "Toilet", "ToiletPaperHanger", "Cart", "Bathtub", "Cabinet", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Shelf", "Drawer", "GarbageCan"], 
    "Tomato": ["DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Sink", "SinkBasin", "Pot", "Bowl", "Fridge", "GarbageCan", "Plate"], 
    # "TomatoSliced": ["DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Sink", "SinkBasin", "Pot", "Bowl", "Fridge", "GarbageCan", "Plate"], 
    "Towel": ["TowelHolder"], 
    "Vase": ["Box", "Dresser", "Desk", "Cart", "Cabinet", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Shelf", "Safe"], 
    "Watch": ["Box", "Dresser", "Desk", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Shelf", "Drawer", "Safe"], 
    "WateringCan": ["Dresser", "Desk", "Cabinet", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Shelf", "Drawer"], 
    "WineBottle": ["Fridge", "Dresser", "Desk", "Cabinet", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Shelf", "GarbageCan"]
}
for k, v in DEFAULT_COMPATIBLE_RECEPTACLES.items():
    if 'Cart' in v:
        v.remove('Cart')

    DEFAULT_COMPATIBLE_RECEPTACLES[k] = v

# fmt: off
OBJECT_TYPES_WITH_PROPERTIES = {
    "StoveBurner": {"openable": False, "receptacle": True, "pickupable": False},
    "Drawer": {"openable": True, "receptacle": True, "pickupable": False},
    "CounterTop": {"openable": False, "receptacle": True, "pickupable": False},
    "Cabinet": {"openable": True, "receptacle": True, "pickupable": False},
    "StoveKnob": {"openable": False, "receptacle": False, "pickupable": False},
    "Window": {"openable": False, "receptacle": False, "pickupable": False},
    "Sink": {"openable": False, "receptacle": True, "pickupable": False},
    "Floor": {"openable": False, "receptacle": True, "pickupable": False},
    "Book": {"openable": True, "receptacle": False, "pickupable": True},
    "Bottle": {"openable": False, "receptacle": False, "pickupable": True},
    "Knife": {"openable": False, "receptacle": False, "pickupable": True},
    "Microwave": {"openable": True, "receptacle": True, "pickupable": False},
    "Bread": {"openable": False, "receptacle": False, "pickupable": True},
    "Fork": {"openable": False, "receptacle": False, "pickupable": True},
    "Shelf": {"openable": False, "receptacle": True, "pickupable": False},
    "Potato": {"openable": False, "receptacle": False, "pickupable": True},
    "HousePlant": {"openable": False, "receptacle": False, "pickupable": False},
    "Toaster": {"openable": False, "receptacle": True, "pickupable": False},
    "SoapBottle": {"openable": False, "receptacle": False, "pickupable": True},
    "Kettle": {"openable": True, "receptacle": False, "pickupable": True},
    "Pan": {"openable": False, "receptacle": True, "pickupable": True},
    "Plate": {"openable": False, "receptacle": True, "pickupable": True},
    "Tomato": {"openable": False, "receptacle": False, "pickupable": True},
    "Vase": {"openable": False, "receptacle": False, "pickupable": True},
    "GarbageCan": {"openable": False, "receptacle": True, "pickupable": False},
    "Egg": {"openable": False, "receptacle": False, "pickupable": True},
    "CreditCard": {"openable": False, "receptacle": False, "pickupable": True},
    "WineBottle": {"openable": False, "receptacle": False, "pickupable": True},
    "Pot": {"openable": False, "receptacle": True, "pickupable": True},
    "Spatula": {"openable": False, "receptacle": False, "pickupable": True},
    "PaperTowelRoll": {"openable": False, "receptacle": False, "pickupable": True},
    "Cup": {"openable": False, "receptacle": True, "pickupable": True},
    "Fridge": {"openable": True, "receptacle": True, "pickupable": False},
    "CoffeeMachine": {"openable": False, "receptacle": True, "pickupable": False},
    "Bowl": {"openable": False, "receptacle": True, "pickupable": True},
    "SinkBasin": {"openable": False, "receptacle": True, "pickupable": False},
    "SaltShaker": {"openable": False, "receptacle": False, "pickupable": True},
    "PepperShaker": {"openable": False, "receptacle": False, "pickupable": True},
    "Lettuce": {"openable": False, "receptacle": False, "pickupable": True},
    "ButterKnife": {"openable": False, "receptacle": False, "pickupable": True},
    "Apple": {"openable": False, "receptacle": False, "pickupable": True},
    "DishSponge": {"openable": False, "receptacle": False, "pickupable": True},
    "Spoon": {"openable": False, "receptacle": False, "pickupable": True},
    "LightSwitch": {"openable": False, "receptacle": False, "pickupable": False},
    "Mug": {"openable": False, "receptacle": True, "pickupable": True},
    "ShelvingUnit": {"openable": False, "receptacle": True, "pickupable": False},
    "Statue": {"openable": False, "receptacle": False, "pickupable": True},
    "Stool": {"openable": False, "receptacle": True, "pickupable": False},
    "Faucet": {"openable": False, "receptacle": False, "pickupable": False},
    "Ladle": {"openable": False, "receptacle": False, "pickupable": True},
    "CellPhone": {"openable": False, "receptacle": False, "pickupable": True},
    "Chair": {"openable": False, "receptacle": True, "pickupable": False},
    "SideTable": {"openable": False, "receptacle": True, "pickupable": False},
    "DiningTable": {"openable": False, "receptacle": True, "pickupable": False},
    "Pen": {"openable": False, "receptacle": False, "pickupable": True},
    "SprayBottle": {"openable": False, "receptacle": False, "pickupable": True},
    "Curtains": {"openable": False, "receptacle": False, "pickupable": False},
    "Pencil": {"openable": False, "receptacle": False, "pickupable": True},
    "Blinds": {"openable": True, "receptacle": False, "pickupable": False},
    "GarbageBag": {"openable": False, "receptacle": False, "pickupable": False},
    "Safe": {"openable": True, "receptacle": True, "pickupable": False},
    "Painting": {"openable": False, "receptacle": False, "pickupable": False},
    "Box": {"openable": True, "receptacle": True, "pickupable": True},
    "Laptop": {"openable": True, "receptacle": False, "pickupable": True},
    "Television": {"openable": False, "receptacle": False, "pickupable": False},
    "TissueBox": {"openable": False, "receptacle": False, "pickupable": True},
    "KeyChain": {"openable": False, "receptacle": False, "pickupable": True},
    "FloorLamp": {"openable": False, "receptacle": False, "pickupable": False},
    "DeskLamp": {"openable": False, "receptacle": False, "pickupable": False},
    "Pillow": {"openable": False, "receptacle": False, "pickupable": True},
    "RemoteControl": {"openable": False, "receptacle": False, "pickupable": True},
    "Watch": {"openable": False, "receptacle": False, "pickupable": True},
    "Newspaper": {"openable": False, "receptacle": False, "pickupable": True},
    "ArmChair": {"openable": False, "receptacle": True, "pickupable": False},
    "CoffeeTable": {"openable": False, "receptacle": True, "pickupable": False},
    "TVStand": {"openable": False, "receptacle": True, "pickupable": False},
    "Sofa": {"openable": False, "receptacle": True, "pickupable": False},
    "WateringCan": {"openable": False, "receptacle": False, "pickupable": True},
    "Boots": {"openable": False, "receptacle": False, "pickupable": True},
    "Ottoman": {"openable": False, "receptacle": True, "pickupable": False},
    "Desk": {"openable": False, "receptacle": True, "pickupable": False},
    "Dresser": {"openable": False, "receptacle": True, "pickupable": False},
    "Mirror": {"openable": False, "receptacle": False, "pickupable": False},
    "DogBed": {"openable": False, "receptacle": True, "pickupable": False},
    "Candle": {"openable": False, "receptacle": False, "pickupable": True},
    "RoomDecor": {"openable": False, "receptacle": False, "pickupable": False},
    "Bed": {"openable": False, "receptacle": True, "pickupable": False},
    "BaseballBat": {"openable": False, "receptacle": False, "pickupable": True},
    "BasketBall": {"openable": False, "receptacle": False, "pickupable": True},
    "AlarmClock": {"openable": False, "receptacle": False, "pickupable": True},
    "CD": {"openable": False, "receptacle": False, "pickupable": True},
    "TennisRacket": {"openable": False, "receptacle": False, "pickupable": True},
    "TeddyBear": {"openable": False, "receptacle": False, "pickupable": True},
    "Poster": {"openable": False, "receptacle": False, "pickupable": False},
    "Cloth": {"openable": False, "receptacle": False, "pickupable": True},
    "Dumbbell": {"openable": False, "receptacle": False, "pickupable": True},
    "LaundryHamper": {"openable": True, "receptacle": True, "pickupable": False},
    "TableTopDecor": {"openable": False, "receptacle": False, "pickupable": True},
    "Desktop": {"openable": False, "receptacle": False, "pickupable": False},
    "Footstool": {"openable": False, "receptacle": True, "pickupable": True},
    "BathtubBasin": {"openable": False, "receptacle": True, "pickupable": False},
    "ShowerCurtain": {"openable": True, "receptacle": False, "pickupable": False},
    "ShowerHead": {"openable": False, "receptacle": False, "pickupable": False},
    "Bathtub": {"openable": False, "receptacle": True, "pickupable": False},
    "Towel": {"openable": False, "receptacle": False, "pickupable": True},
    "HandTowel": {"openable": False, "receptacle": False, "pickupable": True},
    "Plunger": {"openable": False, "receptacle": False, "pickupable": True},
    "TowelHolder": {"openable": False, "receptacle": True, "pickupable": False},
    "ToiletPaperHanger": {"openable": False, "receptacle": True, "pickupable": False},
    "SoapBar": {"openable": False, "receptacle": False, "pickupable": True},
    "ToiletPaper": {"openable": False, "receptacle": False, "pickupable": True},
    "HandTowelHolder": {"openable": False, "receptacle": True, "pickupable": False},
    "ScrubBrush": {"openable": False, "receptacle": False, "pickupable": True},
    "Toilet": {"openable": True, "receptacle": True, "pickupable": False},
    "ShowerGlass": {"openable": False, "receptacle": False, "pickupable": False},
    "ShowerDoor": {"openable": True, "receptacle": False, "pickupable": False},
    "AluminumFoil": {"openable": False, "receptacle": False, "pickupable": True},
    "VacuumCleaner": {"openable": False, "receptacle": False, "pickupable": False}
}
# fmt: on

PICKUPABLE_OBJECTS = list(
    sorted(
        [
            object_type
            for object_type, properties in OBJECT_TYPES_WITH_PROPERTIES.items()
            if properties["pickupable"]
        ]
    )
)

OPENABLE_OBJECTS = list(
    sorted(
        [
            object_type
            for object_type, properties in OBJECT_TYPES_WITH_PROPERTIES.items()
            if properties["openable"] and not properties["pickupable"]
        ]
    )
)

RECEPTACLE_OBJECTS = list(
    sorted(
        [
            object_type
            for object_type, properties in OBJECT_TYPES_WITH_PROPERTIES.items()
            if properties["receptacle"] and not properties["pickupable"]
        ]
    )
)

NOT_PROPER_RECEPTACLES = ["Toilet", "LaundryHamper"]

PICKUPABLE_OBJECTS_TO_EXISTING_SCENES = {
    'AlarmClock': ['FloorPlan301', 'FloorPlan302', 'FloorPlan303', 'FloorPlan304', 'FloorPlan305', 'FloorPlan306', 'FloorPlan307', 'FloorPlan308', 'FloorPlan309', 'FloorPlan310', 'FloorPlan311', 'FloorPlan312', 'FloorPlan313', 'FloorPlan314', 'FloorPlan315', 'FloorPlan316', 'FloorPlan317', 'FloorPlan318', 'FloorPlan319', 'FloorPlan320', 'FloorPlan321', 'FloorPlan322', 'FloorPlan323', 'FloorPlan324', 'FloorPlan325', 'FloorPlan326', 'FloorPlan327', 'FloorPlan328', 'FloorPlan329', 'FloorPlan330'],
    'AluminumFoil': ['FloorPlan21'],
    'Apple': ['FloorPlan1', 'FloorPlan2', 'FloorPlan3', 'FloorPlan4', 'FloorPlan5', 'FloorPlan6', 'FloorPlan7', 'FloorPlan8', 'FloorPlan9', 'FloorPlan10', 'FloorPlan11', 'FloorPlan12', 'FloorPlan13', 'FloorPlan14', 'FloorPlan15', 'FloorPlan16', 'FloorPlan17', 'FloorPlan18', 'FloorPlan19', 'FloorPlan20', 'FloorPlan21', 'FloorPlan22', 'FloorPlan23', 'FloorPlan24', 'FloorPlan25', 'FloorPlan26', 'FloorPlan27', 'FloorPlan28', 'FloorPlan29', 'FloorPlan30'],
    'BaseballBat': ['FloorPlan301', 'FloorPlan303', 'FloorPlan305', 'FloorPlan308', 'FloorPlan310', 'FloorPlan313', 'FloorPlan315', 'FloorPlan316', 'FloorPlan322', 'FloorPlan324', 'FloorPlan326', 'FloorPlan327', 'FloorPlan329'],
    'BasketBall': ['FloorPlan301', 'FloorPlan304', 'FloorPlan305', 'FloorPlan308', 'FloorPlan310', 'FloorPlan312', 'FloorPlan314', 'FloorPlan319', 'FloorPlan320', 'FloorPlan326', 'FloorPlan327'],
    'Book': ['FloorPlan1', 'FloorPlan7', 'FloorPlan201', 'FloorPlan202', 'FloorPlan203', 'FloorPlan209', 'FloorPlan213', 'FloorPlan224', 'FloorPlan229', 'FloorPlan301', 'FloorPlan302', 'FloorPlan303', 'FloorPlan304', 'FloorPlan305', 'FloorPlan306', 'FloorPlan307', 'FloorPlan308', 'FloorPlan309', 'FloorPlan310', 'FloorPlan311', 'FloorPlan312', 'FloorPlan313', 'FloorPlan314', 'FloorPlan315', 'FloorPlan316', 'FloorPlan317', 'FloorPlan318', 'FloorPlan319', 'FloorPlan320', 'FloorPlan321', 'FloorPlan322', 'FloorPlan323', 'FloorPlan324', 'FloorPlan325', 'FloorPlan326', 'FloorPlan327', 'FloorPlan328', 'FloorPlan329', 'FloorPlan330'],
    'Boots': ['FloorPlan203', 'FloorPlan212', 'FloorPlan220', 'FloorPlan229', 'FloorPlan230', 'FloorPlan301', 'FloorPlan327', 'FloorPlan330'],
    'Bottle': ['FloorPlan1', 'FloorPlan8', 'FloorPlan10', 'FloorPlan16', 'FloorPlan17', 'FloorPlan23', 'FloorPlan30'],
    'Bowl': ['FloorPlan1', 'FloorPlan2', 'FloorPlan3', 'FloorPlan4', 'FloorPlan5', 'FloorPlan6', 'FloorPlan7', 'FloorPlan8', 'FloorPlan9', 'FloorPlan10', 'FloorPlan11', 'FloorPlan12', 'FloorPlan13', 'FloorPlan14', 'FloorPlan15', 'FloorPlan16', 'FloorPlan17', 'FloorPlan18', 'FloorPlan19', 'FloorPlan20', 'FloorPlan21', 'FloorPlan22', 'FloorPlan23', 'FloorPlan24', 'FloorPlan25', 'FloorPlan26', 'FloorPlan27', 'FloorPlan28', 'FloorPlan29', 'FloorPlan30', 'FloorPlan201', 'FloorPlan203', 'FloorPlan206', 'FloorPlan214', 'FloorPlan215', 'FloorPlan301', 'FloorPlan302', 'FloorPlan304', 'FloorPlan305', 'FloorPlan307', 'FloorPlan308', 'FloorPlan311', 'FloorPlan316', 'FloorPlan317', 'FloorPlan323', 'FloorPlan326', 'FloorPlan327', 'FloorPlan330'],
    'Box': ['FloorPlan201', 'FloorPlan202', 'FloorPlan203', 'FloorPlan204', 'FloorPlan205', 'FloorPlan206', 'FloorPlan207', 'FloorPlan208', 'FloorPlan209', 'FloorPlan210', 'FloorPlan211', 'FloorPlan212', 'FloorPlan213', 'FloorPlan214', 'FloorPlan215', 'FloorPlan216', 'FloorPlan217', 'FloorPlan218', 'FloorPlan219', 'FloorPlan220', 'FloorPlan221', 'FloorPlan222', 'FloorPlan223', 'FloorPlan224', 'FloorPlan225', 'FloorPlan226', 'FloorPlan227', 'FloorPlan228', 'FloorPlan229', 'FloorPlan230', 'FloorPlan301', 'FloorPlan303', 'FloorPlan304', 'FloorPlan305', 'FloorPlan310', 'FloorPlan313', 'FloorPlan314', 'FloorPlan316', 'FloorPlan317', 'FloorPlan318', 'FloorPlan319', 'FloorPlan329'],
    'Bread': ['FloorPlan1', 'FloorPlan2', 'FloorPlan3', 'FloorPlan4', 'FloorPlan5', 'FloorPlan6', 'FloorPlan7', 'FloorPlan8', 'FloorPlan9', 'FloorPlan10', 'FloorPlan11', 'FloorPlan12', 'FloorPlan13', 'FloorPlan14', 'FloorPlan15', 'FloorPlan16', 'FloorPlan17', 'FloorPlan18', 'FloorPlan19', 'FloorPlan20', 'FloorPlan21', 'FloorPlan22', 'FloorPlan23', 'FloorPlan24', 'FloorPlan25', 'FloorPlan26', 'FloorPlan27', 'FloorPlan28', 'FloorPlan29', 'FloorPlan30'],
    'ButterKnife': ['FloorPlan1', 'FloorPlan2', 'FloorPlan3', 'FloorPlan4', 'FloorPlan5', 'FloorPlan6', 'FloorPlan7', 'FloorPlan8', 'FloorPlan9', 'FloorPlan10', 'FloorPlan11', 'FloorPlan12', 'FloorPlan13', 'FloorPlan14', 'FloorPlan15', 'FloorPlan16', 'FloorPlan17', 'FloorPlan18', 'FloorPlan19', 'FloorPlan20', 'FloorPlan21', 'FloorPlan22', 'FloorPlan23', 'FloorPlan24', 'FloorPlan25', 'FloorPlan26', 'FloorPlan27', 'FloorPlan28', 'FloorPlan29', 'FloorPlan30'],
    'CD': ['FloorPlan301', 'FloorPlan302', 'FloorPlan303', 'FloorPlan304', 'FloorPlan305', 'FloorPlan306', 'FloorPlan307', 'FloorPlan308', 'FloorPlan309', 'FloorPlan310', 'FloorPlan311', 'FloorPlan312', 'FloorPlan313', 'FloorPlan314', 'FloorPlan315', 'FloorPlan316', 'FloorPlan317', 'FloorPlan318', 'FloorPlan319', 'FloorPlan320', 'FloorPlan321', 'FloorPlan322', 'FloorPlan323', 'FloorPlan324', 'FloorPlan325', 'FloorPlan326', 'FloorPlan327', 'FloorPlan328', 'FloorPlan329', 'FloorPlan330'],
    'Candle': ['FloorPlan220', 'FloorPlan223', 'FloorPlan230', 'FloorPlan401', 'FloorPlan402', 'FloorPlan403', 'FloorPlan404', 'FloorPlan405', 'FloorPlan406', 'FloorPlan407', 'FloorPlan408', 'FloorPlan409', 'FloorPlan410', 'FloorPlan411', 'FloorPlan412', 'FloorPlan413', 'FloorPlan414', 'FloorPlan415', 'FloorPlan416', 'FloorPlan417', 'FloorPlan418', 'FloorPlan419', 'FloorPlan420', 'FloorPlan421', 'FloorPlan422', 'FloorPlan423', 'FloorPlan424', 'FloorPlan425', 'FloorPlan426', 'FloorPlan427', 'FloorPlan428', 'FloorPlan429', 'FloorPlan430'],
    'CellPhone': ['FloorPlan2', 'FloorPlan10', 'FloorPlan17', 'FloorPlan30', 'FloorPlan203', 'FloorPlan204', 'FloorPlan211', 'FloorPlan218', 'FloorPlan219', 'FloorPlan224', 'FloorPlan229', 'FloorPlan301', 'FloorPlan302', 'FloorPlan303', 'FloorPlan304', 'FloorPlan305', 'FloorPlan306', 'FloorPlan307', 'FloorPlan308', 'FloorPlan309', 'FloorPlan310', 'FloorPlan311', 'FloorPlan312', 'FloorPlan313', 'FloorPlan314', 'FloorPlan315', 'FloorPlan316', 'FloorPlan317', 'FloorPlan318', 'FloorPlan319', 'FloorPlan320', 'FloorPlan321', 'FloorPlan322', 'FloorPlan323', 'FloorPlan324', 'FloorPlan325', 'FloorPlan326', 'FloorPlan327', 'FloorPlan328', 'FloorPlan329', 'FloorPlan330'],
    'Cloth': ['FloorPlan303', 'FloorPlan307', 'FloorPlan317', 'FloorPlan322', 'FloorPlan326', 'FloorPlan401', 'FloorPlan402', 'FloorPlan403', 'FloorPlan404', 'FloorPlan405', 'FloorPlan406', 'FloorPlan407', 'FloorPlan408', 'FloorPlan409', 'FloorPlan410', 'FloorPlan411', 'FloorPlan412', 'FloorPlan413', 'FloorPlan414', 'FloorPlan415', 'FloorPlan416', 'FloorPlan417', 'FloorPlan418', 'FloorPlan419', 'FloorPlan420', 'FloorPlan421', 'FloorPlan422', 'FloorPlan423', 'FloorPlan424', 'FloorPlan425', 'FloorPlan426', 'FloorPlan427', 'FloorPlan428', 'FloorPlan429', 'FloorPlan430'],
    'CreditCard': ['FloorPlan1', 'FloorPlan10', 'FloorPlan22', 'FloorPlan201', 'FloorPlan202', 'FloorPlan203', 'FloorPlan204', 'FloorPlan205', 'FloorPlan206', 'FloorPlan207', 'FloorPlan208', 'FloorPlan209', 'FloorPlan210', 'FloorPlan211', 'FloorPlan212', 'FloorPlan213', 'FloorPlan214', 'FloorPlan215', 'FloorPlan216', 'FloorPlan217', 'FloorPlan218', 'FloorPlan219', 'FloorPlan220', 'FloorPlan221', 'FloorPlan222', 'FloorPlan223', 'FloorPlan224', 'FloorPlan225', 'FloorPlan226', 'FloorPlan227', 'FloorPlan228', 'FloorPlan229', 'FloorPlan230', 'FloorPlan301', 'FloorPlan302', 'FloorPlan303', 'FloorPlan304', 'FloorPlan305', 'FloorPlan306', 'FloorPlan307', 'FloorPlan308', 'FloorPlan309', 'FloorPlan310', 'FloorPlan311', 'FloorPlan312', 'FloorPlan313', 'FloorPlan314', 'FloorPlan315', 'FloorPlan316', 'FloorPlan317', 'FloorPlan318', 'FloorPlan319', 'FloorPlan320', 'FloorPlan321', 'FloorPlan322', 'FloorPlan323', 'FloorPlan324', 'FloorPlan325', 'FloorPlan326', 'FloorPlan327', 'FloorPlan328', 'FloorPlan329', 'FloorPlan330'],
    'Cup': ['FloorPlan1', 'FloorPlan2', 'FloorPlan3', 'FloorPlan4', 'FloorPlan5', 'FloorPlan6', 'FloorPlan7', 'FloorPlan8', 'FloorPlan9', 'FloorPlan10', 'FloorPlan11', 'FloorPlan12', 'FloorPlan13', 'FloorPlan14', 'FloorPlan15', 'FloorPlan16', 'FloorPlan17', 'FloorPlan18', 'FloorPlan19', 'FloorPlan20', 'FloorPlan21', 'FloorPlan22', 'FloorPlan23', 'FloorPlan24', 'FloorPlan25', 'FloorPlan26', 'FloorPlan27', 'FloorPlan28', 'FloorPlan29', 'FloorPlan30'],
    'DishSponge': ['FloorPlan1', 'FloorPlan2', 'FloorPlan3', 'FloorPlan4', 'FloorPlan5', 'FloorPlan6', 'FloorPlan7', 'FloorPlan8', 'FloorPlan9', 'FloorPlan10', 'FloorPlan11', 'FloorPlan12', 'FloorPlan13', 'FloorPlan14', 'FloorPlan15', 'FloorPlan16', 'FloorPlan17', 'FloorPlan18', 'FloorPlan19', 'FloorPlan20', 'FloorPlan21', 'FloorPlan22', 'FloorPlan23', 'FloorPlan24', 'FloorPlan25', 'FloorPlan26', 'FloorPlan27', 'FloorPlan28', 'FloorPlan29', 'FloorPlan30', 'FloorPlan401', 'FloorPlan403', 'FloorPlan414', 'FloorPlan421', 'FloorPlan427', 'FloorPlan430'],
    'Dumbbell': ['FloorPlan303', 'FloorPlan312', 'FloorPlan320', 'FloorPlan328'],
    'Egg': ['FloorPlan1', 'FloorPlan2', 'FloorPlan3', 'FloorPlan4', 'FloorPlan5', 'FloorPlan6', 'FloorPlan7', 'FloorPlan8', 'FloorPlan9', 'FloorPlan10', 'FloorPlan11', 'FloorPlan12', 'FloorPlan13', 'FloorPlan14', 'FloorPlan15', 'FloorPlan16', 'FloorPlan17', 'FloorPlan18', 'FloorPlan19', 'FloorPlan20', 'FloorPlan21', 'FloorPlan22', 'FloorPlan23', 'FloorPlan24', 'FloorPlan25', 'FloorPlan26', 'FloorPlan27', 'FloorPlan28', 'FloorPlan29', 'FloorPlan30'],
    'Footstool': ['FloorPlan320', 'FloorPlan430'],
    'Fork': ['FloorPlan1', 'FloorPlan2', 'FloorPlan3', 'FloorPlan4', 'FloorPlan5', 'FloorPlan6', 'FloorPlan7', 'FloorPlan8', 'FloorPlan9', 'FloorPlan10', 'FloorPlan11', 'FloorPlan12', 'FloorPlan13', 'FloorPlan14', 'FloorPlan15', 'FloorPlan16', 'FloorPlan17', 'FloorPlan18', 'FloorPlan19', 'FloorPlan20', 'FloorPlan21', 'FloorPlan22', 'FloorPlan23', 'FloorPlan24', 'FloorPlan25', 'FloorPlan26', 'FloorPlan27', 'FloorPlan28', 'FloorPlan29', 'FloorPlan30'],
    'HandTowel': ['FloorPlan401', 'FloorPlan402', 'FloorPlan403', 'FloorPlan404', 'FloorPlan405', 'FloorPlan406', 'FloorPlan407', 'FloorPlan408', 'FloorPlan409', 'FloorPlan410', 'FloorPlan411', 'FloorPlan412', 'FloorPlan413', 'FloorPlan414', 'FloorPlan415', 'FloorPlan416', 'FloorPlan417', 'FloorPlan418', 'FloorPlan419', 'FloorPlan420', 'FloorPlan421', 'FloorPlan422', 'FloorPlan423', 'FloorPlan424', 'FloorPlan425', 'FloorPlan426', 'FloorPlan427', 'FloorPlan428', 'FloorPlan429', 'FloorPlan430'],
    'Kettle': ['FloorPlan1', 'FloorPlan2', 'FloorPlan3', 'FloorPlan5', 'FloorPlan6', 'FloorPlan7', 'FloorPlan8', 'FloorPlan9', 'FloorPlan11', 'FloorPlan15', 'FloorPlan17', 'FloorPlan18', 'FloorPlan22', 'FloorPlan29', 'FloorPlan30'],
    'KeyChain': ['FloorPlan201', 'FloorPlan202', 'FloorPlan203', 'FloorPlan204', 'FloorPlan205', 'FloorPlan206', 'FloorPlan207', 'FloorPlan208', 'FloorPlan209', 'FloorPlan210', 'FloorPlan211', 'FloorPlan212', 'FloorPlan213', 'FloorPlan214', 'FloorPlan215', 'FloorPlan216', 'FloorPlan217', 'FloorPlan218', 'FloorPlan219', 'FloorPlan220', 'FloorPlan221', 'FloorPlan222', 'FloorPlan223', 'FloorPlan224', 'FloorPlan225', 'FloorPlan226', 'FloorPlan227', 'FloorPlan228', 'FloorPlan229', 'FloorPlan230', 'FloorPlan301', 'FloorPlan302', 'FloorPlan303', 'FloorPlan304', 'FloorPlan305', 'FloorPlan306', 'FloorPlan307', 'FloorPlan308', 'FloorPlan309', 'FloorPlan310', 'FloorPlan311', 'FloorPlan312', 'FloorPlan313', 'FloorPlan314', 'FloorPlan315', 'FloorPlan316', 'FloorPlan317', 'FloorPlan318', 'FloorPlan319', 'FloorPlan320', 'FloorPlan321', 'FloorPlan322', 'FloorPlan323', 'FloorPlan324', 'FloorPlan325', 'FloorPlan326', 'FloorPlan327', 'FloorPlan328', 'FloorPlan329', 'FloorPlan330'],
    'Knife': ['FloorPlan1', 'FloorPlan2', 'FloorPlan3', 'FloorPlan4', 'FloorPlan5', 'FloorPlan6', 'FloorPlan7', 'FloorPlan8', 'FloorPlan9', 'FloorPlan10', 'FloorPlan11', 'FloorPlan12', 'FloorPlan13', 'FloorPlan14', 'FloorPlan15', 'FloorPlan16', 'FloorPlan17', 'FloorPlan18', 'FloorPlan19', 'FloorPlan20', 'FloorPlan21', 'FloorPlan22', 'FloorPlan23', 'FloorPlan24', 'FloorPlan25', 'FloorPlan26', 'FloorPlan27', 'FloorPlan28', 'FloorPlan29', 'FloorPlan30'],
    'Ladle': ['FloorPlan2', 'FloorPlan4', 'FloorPlan5', 'FloorPlan8', 'FloorPlan14', 'FloorPlan16', 'FloorPlan20', 'FloorPlan25', 'FloorPlan27', 'FloorPlan30'],
    'Laptop': ['FloorPlan201', 'FloorPlan202', 'FloorPlan203', 'FloorPlan204', 'FloorPlan205', 'FloorPlan206', 'FloorPlan207', 'FloorPlan208', 'FloorPlan209', 'FloorPlan210', 'FloorPlan211', 'FloorPlan212', 'FloorPlan213', 'FloorPlan214', 'FloorPlan215', 'FloorPlan216', 'FloorPlan217', 'FloorPlan218', 'FloorPlan219', 'FloorPlan220', 'FloorPlan221', 'FloorPlan222', 'FloorPlan223', 'FloorPlan224', 'FloorPlan225', 'FloorPlan226', 'FloorPlan227', 'FloorPlan228', 'FloorPlan229', 'FloorPlan230', 'FloorPlan301', 'FloorPlan302', 'FloorPlan303', 'FloorPlan304', 'FloorPlan305', 'FloorPlan306', 'FloorPlan307', 'FloorPlan308', 'FloorPlan309', 'FloorPlan310', 'FloorPlan311', 'FloorPlan312', 'FloorPlan313', 'FloorPlan314', 'FloorPlan315', 'FloorPlan316', 'FloorPlan317', 'FloorPlan318', 'FloorPlan319', 'FloorPlan320', 'FloorPlan321', 'FloorPlan322', 'FloorPlan323', 'FloorPlan324', 'FloorPlan325', 'FloorPlan326', 'FloorPlan327', 'FloorPlan328', 'FloorPlan329', 'FloorPlan330'],
    'Lettuce': ['FloorPlan1', 'FloorPlan2', 'FloorPlan3', 'FloorPlan4', 'FloorPlan5', 'FloorPlan6', 'FloorPlan7', 'FloorPlan8', 'FloorPlan9', 'FloorPlan10', 'FloorPlan11', 'FloorPlan12', 'FloorPlan13', 'FloorPlan14', 'FloorPlan15', 'FloorPlan16', 'FloorPlan17', 'FloorPlan18', 'FloorPlan19', 'FloorPlan20', 'FloorPlan21', 'FloorPlan22', 'FloorPlan23', 'FloorPlan24', 'FloorPlan25', 'FloorPlan26', 'FloorPlan27', 'FloorPlan28', 'FloorPlan29', 'FloorPlan30'],
    'Mug': ['FloorPlan1', 'FloorPlan2', 'FloorPlan3', 'FloorPlan4', 'FloorPlan5', 'FloorPlan6', 'FloorPlan7', 'FloorPlan8', 'FloorPlan9', 'FloorPlan10', 'FloorPlan11', 'FloorPlan12', 'FloorPlan13', 'FloorPlan14', 'FloorPlan15', 'FloorPlan16', 'FloorPlan17', 'FloorPlan18', 'FloorPlan19', 'FloorPlan20', 'FloorPlan21', 'FloorPlan22', 'FloorPlan23', 'FloorPlan24', 'FloorPlan25', 'FloorPlan26', 'FloorPlan27', 'FloorPlan28', 'FloorPlan29', 'FloorPlan30', 'FloorPlan301', 'FloorPlan303', 'FloorPlan304', 'FloorPlan305', 'FloorPlan308', 'FloorPlan309', 'FloorPlan311', 'FloorPlan313', 'FloorPlan315', 'FloorPlan318', 'FloorPlan322', 'FloorPlan323', 'FloorPlan324', 'FloorPlan326', 'FloorPlan328', 'FloorPlan329'],
    'Newspaper': ['FloorPlan201', 'FloorPlan203', 'FloorPlan209', 'FloorPlan210', 'FloorPlan211', 'FloorPlan212', 'FloorPlan214', 'FloorPlan215', 'FloorPlan216', 'FloorPlan218', 'FloorPlan219', 'FloorPlan222', 'FloorPlan224', 'FloorPlan225', 'FloorPlan226', 'FloorPlan227', 'FloorPlan228', 'FloorPlan230'],
    'Pan': ['FloorPlan1', 'FloorPlan2', 'FloorPlan3', 'FloorPlan4', 'FloorPlan5', 'FloorPlan6', 'FloorPlan7', 'FloorPlan8', 'FloorPlan9', 'FloorPlan10', 'FloorPlan11', 'FloorPlan12', 'FloorPlan13', 'FloorPlan14', 'FloorPlan15', 'FloorPlan16', 'FloorPlan17', 'FloorPlan18', 'FloorPlan19', 'FloorPlan20', 'FloorPlan21', 'FloorPlan22', 'FloorPlan23', 'FloorPlan24', 'FloorPlan25', 'FloorPlan26', 'FloorPlan27', 'FloorPlan28', 'FloorPlan29', 'FloorPlan30'],
    'PaperTowelRoll': ['FloorPlan1', 'FloorPlan3', 'FloorPlan5', 'FloorPlan6', 'FloorPlan10', 'FloorPlan11', 'FloorPlan13', 'FloorPlan16', 'FloorPlan18', 'FloorPlan24', 'FloorPlan401', 'FloorPlan403', 'FloorPlan414', 'FloorPlan427'],
    'Pen': ['FloorPlan8', 'FloorPlan23', 'FloorPlan201', 'FloorPlan209', 'FloorPlan212', 'FloorPlan221', 'FloorPlan229', 'FloorPlan301', 'FloorPlan302', 'FloorPlan303', 'FloorPlan304', 'FloorPlan305', 'FloorPlan306', 'FloorPlan307', 'FloorPlan308', 'FloorPlan309', 'FloorPlan310', 'FloorPlan311', 'FloorPlan312', 'FloorPlan313', 'FloorPlan314', 'FloorPlan315', 'FloorPlan316', 'FloorPlan317', 'FloorPlan318', 'FloorPlan319', 'FloorPlan320', 'FloorPlan321', 'FloorPlan322', 'FloorPlan323', 'FloorPlan324', 'FloorPlan325', 'FloorPlan326', 'FloorPlan327', 'FloorPlan328', 'FloorPlan329', 'FloorPlan330'],
    'Pencil': ['FloorPlan13', 'FloorPlan23', 'FloorPlan201', 'FloorPlan203', 'FloorPlan212', 'FloorPlan229', 'FloorPlan301', 'FloorPlan302', 'FloorPlan303', 'FloorPlan304', 'FloorPlan305', 'FloorPlan306', 'FloorPlan307', 'FloorPlan308', 'FloorPlan309', 'FloorPlan310', 'FloorPlan311', 'FloorPlan312', 'FloorPlan313', 'FloorPlan314', 'FloorPlan315', 'FloorPlan316', 'FloorPlan317', 'FloorPlan318', 'FloorPlan319', 'FloorPlan320', 'FloorPlan321', 'FloorPlan322', 'FloorPlan323', 'FloorPlan324', 'FloorPlan325', 'FloorPlan326', 'FloorPlan327', 'FloorPlan328', 'FloorPlan329', 'FloorPlan330'],
    'PepperShaker': ['FloorPlan1', 'FloorPlan2', 'FloorPlan3', 'FloorPlan4', 'FloorPlan5', 'FloorPlan6', 'FloorPlan7', 'FloorPlan8', 'FloorPlan9', 'FloorPlan10', 'FloorPlan11', 'FloorPlan12', 'FloorPlan13', 'FloorPlan14', 'FloorPlan15', 'FloorPlan16', 'FloorPlan17', 'FloorPlan18', 'FloorPlan19', 'FloorPlan20', 'FloorPlan21', 'FloorPlan22', 'FloorPlan23', 'FloorPlan24', 'FloorPlan25', 'FloorPlan26', 'FloorPlan27', 'FloorPlan28', 'FloorPlan29', 'FloorPlan30'],
    'Pillow': ['FloorPlan201', 'FloorPlan202', 'FloorPlan203', 'FloorPlan204', 'FloorPlan205', 'FloorPlan206', 'FloorPlan207', 'FloorPlan208', 'FloorPlan209', 'FloorPlan210', 'FloorPlan211', 'FloorPlan212', 'FloorPlan213', 'FloorPlan214', 'FloorPlan215', 'FloorPlan216', 'FloorPlan217', 'FloorPlan218', 'FloorPlan219', 'FloorPlan220', 'FloorPlan221', 'FloorPlan222', 'FloorPlan223', 'FloorPlan224', 'FloorPlan225', 'FloorPlan226', 'FloorPlan227', 'FloorPlan228', 'FloorPlan229', 'FloorPlan230', 'FloorPlan301', 'FloorPlan302', 'FloorPlan303', 'FloorPlan304', 'FloorPlan305', 'FloorPlan306', 'FloorPlan307', 'FloorPlan308', 'FloorPlan309', 'FloorPlan310', 'FloorPlan311', 'FloorPlan312', 'FloorPlan313', 'FloorPlan314', 'FloorPlan315', 'FloorPlan316', 'FloorPlan317', 'FloorPlan318', 'FloorPlan319', 'FloorPlan320', 'FloorPlan321', 'FloorPlan322', 'FloorPlan323', 'FloorPlan324', 'FloorPlan325', 'FloorPlan326', 'FloorPlan327', 'FloorPlan328', 'FloorPlan329', 'FloorPlan330'],
    'Plate': ['FloorPlan1', 'FloorPlan2', 'FloorPlan3', 'FloorPlan4', 'FloorPlan5', 'FloorPlan6', 'FloorPlan7', 'FloorPlan8', 'FloorPlan9', 'FloorPlan10', 'FloorPlan11', 'FloorPlan12', 'FloorPlan13', 'FloorPlan14', 'FloorPlan15', 'FloorPlan16', 'FloorPlan17', 'FloorPlan18', 'FloorPlan19', 'FloorPlan20', 'FloorPlan21', 'FloorPlan22', 'FloorPlan23', 'FloorPlan24', 'FloorPlan25', 'FloorPlan26', 'FloorPlan27', 'FloorPlan28', 'FloorPlan29', 'FloorPlan30', 'FloorPlan201', 'FloorPlan203', 'FloorPlan211', 'FloorPlan214', 'FloorPlan218', 'FloorPlan221', 'FloorPlan223', 'FloorPlan227', 'FloorPlan228', 'FloorPlan230'],
    'Plunger': ['FloorPlan401', 'FloorPlan402', 'FloorPlan403', 'FloorPlan404', 'FloorPlan405', 'FloorPlan406', 'FloorPlan407', 'FloorPlan408', 'FloorPlan409', 'FloorPlan410', 'FloorPlan411', 'FloorPlan412', 'FloorPlan413', 'FloorPlan414', 'FloorPlan415', 'FloorPlan416', 'FloorPlan417', 'FloorPlan418', 'FloorPlan419', 'FloorPlan420', 'FloorPlan421', 'FloorPlan422', 'FloorPlan423', 'FloorPlan424', 'FloorPlan425', 'FloorPlan426', 'FloorPlan427', 'FloorPlan428', 'FloorPlan429', 'FloorPlan430'],
    'Pot': ['FloorPlan1', 'FloorPlan2', 'FloorPlan3', 'FloorPlan4', 'FloorPlan5', 'FloorPlan6', 'FloorPlan7', 'FloorPlan8', 'FloorPlan9', 'FloorPlan10', 'FloorPlan11', 'FloorPlan12', 'FloorPlan13', 'FloorPlan14', 'FloorPlan15', 'FloorPlan16', 'FloorPlan17', 'FloorPlan18', 'FloorPlan19', 'FloorPlan20', 'FloorPlan21', 'FloorPlan22', 'FloorPlan23', 'FloorPlan24', 'FloorPlan25', 'FloorPlan26', 'FloorPlan27', 'FloorPlan28', 'FloorPlan29', 'FloorPlan30'],
    'Potato': ['FloorPlan1', 'FloorPlan2', 'FloorPlan3', 'FloorPlan4', 'FloorPlan5', 'FloorPlan6', 'FloorPlan7', 'FloorPlan8', 'FloorPlan9', 'FloorPlan10', 'FloorPlan11', 'FloorPlan12', 'FloorPlan13', 'FloorPlan14', 'FloorPlan15', 'FloorPlan16', 'FloorPlan17', 'FloorPlan18', 'FloorPlan19', 'FloorPlan20', 'FloorPlan21', 'FloorPlan22', 'FloorPlan23', 'FloorPlan24', 'FloorPlan25', 'FloorPlan26', 'FloorPlan27', 'FloorPlan28', 'FloorPlan29', 'FloorPlan30'],
    'RemoteControl': ['FloorPlan201', 'FloorPlan202', 'FloorPlan203', 'FloorPlan204', 'FloorPlan205', 'FloorPlan206', 'FloorPlan207', 'FloorPlan208', 'FloorPlan209', 'FloorPlan210', 'FloorPlan211', 'FloorPlan212', 'FloorPlan213', 'FloorPlan214', 'FloorPlan215', 'FloorPlan216', 'FloorPlan217', 'FloorPlan218', 'FloorPlan219', 'FloorPlan220', 'FloorPlan221', 'FloorPlan222', 'FloorPlan223', 'FloorPlan224', 'FloorPlan225', 'FloorPlan226', 'FloorPlan227', 'FloorPlan228', 'FloorPlan229', 'FloorPlan230', 'FloorPlan307', 'FloorPlan309', 'FloorPlan311'],
    'SaltShaker': ['FloorPlan1', 'FloorPlan2', 'FloorPlan3', 'FloorPlan4', 'FloorPlan5', 'FloorPlan6', 'FloorPlan7', 'FloorPlan8', 'FloorPlan9', 'FloorPlan10', 'FloorPlan11', 'FloorPlan12', 'FloorPlan13', 'FloorPlan14', 'FloorPlan15', 'FloorPlan16', 'FloorPlan17', 'FloorPlan18', 'FloorPlan19', 'FloorPlan20', 'FloorPlan21', 'FloorPlan22', 'FloorPlan23', 'FloorPlan24', 'FloorPlan25', 'FloorPlan26', 'FloorPlan27', 'FloorPlan28', 'FloorPlan29', 'FloorPlan30'],
    'ScrubBrush': ['FloorPlan401', 'FloorPlan402', 'FloorPlan403', 'FloorPlan404', 'FloorPlan405', 'FloorPlan406', 'FloorPlan407', 'FloorPlan408', 'FloorPlan409', 'FloorPlan410', 'FloorPlan411', 'FloorPlan412', 'FloorPlan413', 'FloorPlan414', 'FloorPlan415', 'FloorPlan416', 'FloorPlan417', 'FloorPlan418', 'FloorPlan419', 'FloorPlan420', 'FloorPlan421', 'FloorPlan422', 'FloorPlan423', 'FloorPlan424', 'FloorPlan425', 'FloorPlan426', 'FloorPlan427', 'FloorPlan428', 'FloorPlan429', 'FloorPlan430'],
    'SoapBar': ['FloorPlan401', 'FloorPlan402', 'FloorPlan403', 'FloorPlan404', 'FloorPlan405', 'FloorPlan406', 'FloorPlan407', 'FloorPlan408', 'FloorPlan409', 'FloorPlan410', 'FloorPlan411', 'FloorPlan412', 'FloorPlan413', 'FloorPlan414', 'FloorPlan415', 'FloorPlan416', 'FloorPlan417', 'FloorPlan418', 'FloorPlan419', 'FloorPlan420', 'FloorPlan421', 'FloorPlan422', 'FloorPlan423', 'FloorPlan424', 'FloorPlan425', 'FloorPlan426', 'FloorPlan427', 'FloorPlan428', 'FloorPlan429', 'FloorPlan430'],
    'SoapBottle': ['FloorPlan1', 'FloorPlan2', 'FloorPlan3', 'FloorPlan4', 'FloorPlan5', 'FloorPlan6', 'FloorPlan7', 'FloorPlan8', 'FloorPlan9', 'FloorPlan10', 'FloorPlan11', 'FloorPlan12', 'FloorPlan13', 'FloorPlan14', 'FloorPlan15', 'FloorPlan16', 'FloorPlan17', 'FloorPlan18', 'FloorPlan19', 'FloorPlan20', 'FloorPlan21', 'FloorPlan22', 'FloorPlan23', 'FloorPlan24', 'FloorPlan25', 'FloorPlan26', 'FloorPlan27', 'FloorPlan28', 'FloorPlan29', 'FloorPlan30', 'FloorPlan401', 'FloorPlan402', 'FloorPlan403', 'FloorPlan404', 'FloorPlan405', 'FloorPlan406', 'FloorPlan407', 'FloorPlan408', 'FloorPlan409', 'FloorPlan410', 'FloorPlan411', 'FloorPlan412', 'FloorPlan413', 'FloorPlan414', 'FloorPlan415', 'FloorPlan416', 'FloorPlan417', 'FloorPlan418', 'FloorPlan419', 'FloorPlan420', 'FloorPlan421', 'FloorPlan422', 'FloorPlan423', 'FloorPlan424', 'FloorPlan425', 'FloorPlan426', 'FloorPlan427', 'FloorPlan428', 'FloorPlan429', 'FloorPlan430'],
    'Spatula': ['FloorPlan1', 'FloorPlan2', 'FloorPlan3', 'FloorPlan4', 'FloorPlan5', 'FloorPlan6', 'FloorPlan7', 'FloorPlan8', 'FloorPlan9', 'FloorPlan10', 'FloorPlan11', 'FloorPlan12', 'FloorPlan13', 'FloorPlan14', 'FloorPlan15', 'FloorPlan16', 'FloorPlan17', 'FloorPlan18', 'FloorPlan19', 'FloorPlan20', 'FloorPlan21', 'FloorPlan22', 'FloorPlan23', 'FloorPlan24', 'FloorPlan25', 'FloorPlan26', 'FloorPlan27', 'FloorPlan28', 'FloorPlan29', 'FloorPlan30'],
    'Spoon': ['FloorPlan1', 'FloorPlan2', 'FloorPlan3', 'FloorPlan4', 'FloorPlan5', 'FloorPlan6', 'FloorPlan7', 'FloorPlan8', 'FloorPlan9', 'FloorPlan10', 'FloorPlan11', 'FloorPlan12', 'FloorPlan13', 'FloorPlan14', 'FloorPlan15', 'FloorPlan16', 'FloorPlan17', 'FloorPlan18', 'FloorPlan19', 'FloorPlan20', 'FloorPlan21', 'FloorPlan22', 'FloorPlan23', 'FloorPlan24', 'FloorPlan25', 'FloorPlan26', 'FloorPlan27', 'FloorPlan28', 'FloorPlan29', 'FloorPlan30'],
    'SprayBottle': ['FloorPlan8', 'FloorPlan17', 'FloorPlan401', 'FloorPlan402', 'FloorPlan403', 'FloorPlan404', 'FloorPlan405', 'FloorPlan406', 'FloorPlan407', 'FloorPlan408', 'FloorPlan409', 'FloorPlan410', 'FloorPlan411', 'FloorPlan412', 'FloorPlan413', 'FloorPlan414', 'FloorPlan415', 'FloorPlan416', 'FloorPlan417', 'FloorPlan418', 'FloorPlan419', 'FloorPlan420', 'FloorPlan421', 'FloorPlan422', 'FloorPlan423', 'FloorPlan424', 'FloorPlan425', 'FloorPlan426', 'FloorPlan427', 'FloorPlan428', 'FloorPlan429', 'FloorPlan430'],
    'Statue': ['FloorPlan1', 'FloorPlan5', 'FloorPlan7', 'FloorPlan10', 'FloorPlan201', 'FloorPlan202', 'FloorPlan204', 'FloorPlan205', 'FloorPlan206', 'FloorPlan207', 'FloorPlan208', 'FloorPlan209', 'FloorPlan210', 'FloorPlan211', 'FloorPlan212', 'FloorPlan213', 'FloorPlan214', 'FloorPlan215', 'FloorPlan216', 'FloorPlan217', 'FloorPlan218', 'FloorPlan219', 'FloorPlan220', 'FloorPlan221', 'FloorPlan222', 'FloorPlan223', 'FloorPlan224', 'FloorPlan225', 'FloorPlan226', 'FloorPlan227', 'FloorPlan228', 'FloorPlan229', 'FloorPlan230', 'FloorPlan301', 'FloorPlan304', 'FloorPlan319', 'FloorPlan330'],
    'TableTopDecor': ['FloorPlan315'],
    'TeddyBear': ['FloorPlan302', 'FloorPlan306', 'FloorPlan309', 'FloorPlan313', 'FloorPlan315', 'FloorPlan317', 'FloorPlan320', 'FloorPlan323', 'FloorPlan326', 'FloorPlan328'],
    'TennisRacket': ['FloorPlan302', 'FloorPlan303', 'FloorPlan307', 'FloorPlan310', 'FloorPlan313', 'FloorPlan318', 'FloorPlan320', 'FloorPlan324', 'FloorPlan326', 'FloorPlan328'],
    'TissueBox': ['FloorPlan201', 'FloorPlan203', 'FloorPlan212', 'FloorPlan216', 'FloorPlan219', 'FloorPlan220', 'FloorPlan225', 'FloorPlan229', 'FloorPlan230', 'FloorPlan301', 'FloorPlan313', 'FloorPlan321', 'FloorPlan328', 'FloorPlan402', 'FloorPlan404', 'FloorPlan419', 'FloorPlan422', 'FloorPlan425', 'FloorPlan426', 'FloorPlan427', 'FloorPlan429', 'FloorPlan430'],
    'ToiletPaper': ['FloorPlan401', 'FloorPlan402', 'FloorPlan403', 'FloorPlan404', 'FloorPlan405', 'FloorPlan406', 'FloorPlan407', 'FloorPlan408', 'FloorPlan409', 'FloorPlan410', 'FloorPlan411', 'FloorPlan412', 'FloorPlan413', 'FloorPlan414', 'FloorPlan415', 'FloorPlan416', 'FloorPlan417', 'FloorPlan418', 'FloorPlan419', 'FloorPlan420', 'FloorPlan421', 'FloorPlan422', 'FloorPlan423', 'FloorPlan424', 'FloorPlan425', 'FloorPlan426', 'FloorPlan427', 'FloorPlan428', 'FloorPlan429', 'FloorPlan430'],
    'Tomato': ['FloorPlan1', 'FloorPlan2', 'FloorPlan3', 'FloorPlan4', 'FloorPlan5', 'FloorPlan6', 'FloorPlan7', 'FloorPlan8', 'FloorPlan9', 'FloorPlan10', 'FloorPlan11', 'FloorPlan12', 'FloorPlan13', 'FloorPlan14', 'FloorPlan15', 'FloorPlan16', 'FloorPlan17', 'FloorPlan18', 'FloorPlan19', 'FloorPlan20', 'FloorPlan21', 'FloorPlan22', 'FloorPlan23', 'FloorPlan24', 'FloorPlan25', 'FloorPlan26', 'FloorPlan27', 'FloorPlan28', 'FloorPlan29', 'FloorPlan30'],
    'Towel': ['FloorPlan401', 'FloorPlan402', 'FloorPlan403', 'FloorPlan404', 'FloorPlan405', 'FloorPlan406', 'FloorPlan407', 'FloorPlan408', 'FloorPlan409', 'FloorPlan410', 'FloorPlan411', 'FloorPlan412', 'FloorPlan413', 'FloorPlan414', 'FloorPlan415', 'FloorPlan416', 'FloorPlan417', 'FloorPlan418', 'FloorPlan419', 'FloorPlan420', 'FloorPlan421', 'FloorPlan422', 'FloorPlan423', 'FloorPlan424', 'FloorPlan425', 'FloorPlan426', 'FloorPlan427', 'FloorPlan428', 'FloorPlan429', 'FloorPlan430'],
    'Vase': ['FloorPlan1', 'FloorPlan5', 'FloorPlan7', 'FloorPlan10', 'FloorPlan18', 'FloorPlan20', 'FloorPlan201', 'FloorPlan203', 'FloorPlan204', 'FloorPlan206', 'FloorPlan207', 'FloorPlan208', 'FloorPlan209', 'FloorPlan211', 'FloorPlan214', 'FloorPlan215', 'FloorPlan216', 'FloorPlan219', 'FloorPlan221', 'FloorPlan223', 'FloorPlan224', 'FloorPlan225', 'FloorPlan227', 'FloorPlan228', 'FloorPlan303', 'FloorPlan325', 'FloorPlan330'],
    'Watch': ['FloorPlan201', 'FloorPlan202', 'FloorPlan203', 'FloorPlan205', 'FloorPlan207', 'FloorPlan209', 'FloorPlan210', 'FloorPlan213', 'FloorPlan215', 'FloorPlan217', 'FloorPlan219', 'FloorPlan222', 'FloorPlan225', 'FloorPlan226', 'FloorPlan228', 'FloorPlan230', 'FloorPlan301', 'FloorPlan326'],
    'WateringCan': ['FloorPlan203', 'FloorPlan204', 'FloorPlan205', 'FloorPlan209', 'FloorPlan210', 'FloorPlan212', 'FloorPlan214', 'FloorPlan215', 'FloorPlan217', 'FloorPlan218', 'FloorPlan223', 'FloorPlan224', 'FloorPlan229'],
    'WineBottle': ['FloorPlan1', 'FloorPlan3', 'FloorPlan7', 'FloorPlan13', 'FloorPlan15', 'FloorPlan16', 'FloorPlan17', 'FloorPlan20', 'FloorPlan23', 'FloorPlan27'],
}

OPENABLE_OBJECTS_TO_EXISTING_SCENES = {
    'Blinds': ['FloorPlan15', 'FloorPlan18', 'FloorPlan21', 'FloorPlan23', 'FloorPlan28', 'FloorPlan226', 'FloorPlan301', 'FloorPlan302', 'FloorPlan303', 'FloorPlan304', 'FloorPlan305', 'FloorPlan306', 'FloorPlan307', 'FloorPlan308', 'FloorPlan309', 'FloorPlan310', 'FloorPlan311', 'FloorPlan313', 'FloorPlan314', 'FloorPlan315', 'FloorPlan316', 'FloorPlan317', 'FloorPlan318', 'FloorPlan319', 'FloorPlan321', 'FloorPlan323', 'FloorPlan324', 'FloorPlan325', 'FloorPlan326', 'FloorPlan327', 'FloorPlan328', 'FloorPlan329', 'FloorPlan330'],
    'Cabinet': ['FloorPlan1', 'FloorPlan2', 'FloorPlan3', 'FloorPlan4', 'FloorPlan5', 'FloorPlan6', 'FloorPlan7', 'FloorPlan8', 'FloorPlan9', 'FloorPlan10', 'FloorPlan11', 'FloorPlan12', 'FloorPlan13', 'FloorPlan14', 'FloorPlan15', 'FloorPlan16', 'FloorPlan17', 'FloorPlan18', 'FloorPlan19', 'FloorPlan20', 'FloorPlan21', 'FloorPlan22', 'FloorPlan23', 'FloorPlan24', 'FloorPlan25', 'FloorPlan26', 'FloorPlan27', 'FloorPlan28', 'FloorPlan29', 'FloorPlan30', 'FloorPlan206', 'FloorPlan207', 'FloorPlan217', 'FloorPlan219', 'FloorPlan224', 'FloorPlan227', 'FloorPlan309', 'FloorPlan310', 'FloorPlan318', 'FloorPlan322', 'FloorPlan325', 'FloorPlan402', 'FloorPlan403', 'FloorPlan405', 'FloorPlan406', 'FloorPlan407', 'FloorPlan408', 'FloorPlan412', 'FloorPlan413', 'FloorPlan414', 'FloorPlan417', 'FloorPlan418', 'FloorPlan422', 'FloorPlan424', 'FloorPlan425', 'FloorPlan428'],
    'Drawer': ['FloorPlan1', 'FloorPlan2', 'FloorPlan3', 'FloorPlan4', 'FloorPlan5', 'FloorPlan6', 'FloorPlan7', 'FloorPlan8', 'FloorPlan9', 'FloorPlan10', 'FloorPlan11', 'FloorPlan12', 'FloorPlan13', 'FloorPlan15', 'FloorPlan16', 'FloorPlan17', 'FloorPlan18', 'FloorPlan19', 'FloorPlan20', 'FloorPlan21', 'FloorPlan22', 'FloorPlan23', 'FloorPlan24', 'FloorPlan25', 'FloorPlan26', 'FloorPlan27', 'FloorPlan28', 'FloorPlan30', 'FloorPlan201', 'FloorPlan203', 'FloorPlan204', 'FloorPlan205', 'FloorPlan208', 'FloorPlan209', 'FloorPlan210', 'FloorPlan211', 'FloorPlan212', 'FloorPlan213', 'FloorPlan216', 'FloorPlan217', 'FloorPlan218', 'FloorPlan219', 'FloorPlan220', 'FloorPlan221', 'FloorPlan222', 'FloorPlan224', 'FloorPlan225', 'FloorPlan226', 'FloorPlan227', 'FloorPlan228', 'FloorPlan229', 'FloorPlan301', 'FloorPlan302', 'FloorPlan303', 'FloorPlan304', 'FloorPlan305', 'FloorPlan306', 'FloorPlan307', 'FloorPlan308', 'FloorPlan309', 'FloorPlan310', 'FloorPlan311', 'FloorPlan312', 'FloorPlan313', 'FloorPlan314', 'FloorPlan315', 'FloorPlan316', 'FloorPlan317', 'FloorPlan318', 'FloorPlan319', 'FloorPlan321', 'FloorPlan322', 'FloorPlan323', 'FloorPlan324', 'FloorPlan325', 'FloorPlan326', 'FloorPlan327', 'FloorPlan328', 'FloorPlan329', 'FloorPlan330', 'FloorPlan409', 'FloorPlan410', 'FloorPlan411', 'FloorPlan413', 'FloorPlan414', 'FloorPlan415', 'FloorPlan416', 'FloorPlan419', 'FloorPlan420', 'FloorPlan421', 'FloorPlan422', 'FloorPlan423', 'FloorPlan426', 'FloorPlan427', 'FloorPlan430'],
    'Fridge': ['FloorPlan1', 'FloorPlan2', 'FloorPlan3', 'FloorPlan4', 'FloorPlan5', 'FloorPlan6', 'FloorPlan7', 'FloorPlan8', 'FloorPlan9', 'FloorPlan10', 'FloorPlan11', 'FloorPlan12', 'FloorPlan13', 'FloorPlan14', 'FloorPlan15', 'FloorPlan16', 'FloorPlan17', 'FloorPlan18', 'FloorPlan19', 'FloorPlan20', 'FloorPlan21', 'FloorPlan22', 'FloorPlan23', 'FloorPlan24', 'FloorPlan25', 'FloorPlan26', 'FloorPlan27', 'FloorPlan28', 'FloorPlan29', 'FloorPlan30'],
    'LaundryHamper': ['FloorPlan304', 'FloorPlan305', 'FloorPlan306', 'FloorPlan307', 'FloorPlan308', 'FloorPlan309', 'FloorPlan311', 'FloorPlan317'],
    'Microwave': ['FloorPlan1', 'FloorPlan2', 'FloorPlan3', 'FloorPlan4', 'FloorPlan5', 'FloorPlan6', 'FloorPlan7', 'FloorPlan8', 'FloorPlan9', 'FloorPlan10', 'FloorPlan11', 'FloorPlan12', 'FloorPlan13', 'FloorPlan14', 'FloorPlan15', 'FloorPlan16', 'FloorPlan17', 'FloorPlan18', 'FloorPlan19', 'FloorPlan20', 'FloorPlan21', 'FloorPlan22', 'FloorPlan23', 'FloorPlan24', 'FloorPlan25', 'FloorPlan26', 'FloorPlan27', 'FloorPlan28', 'FloorPlan29', 'FloorPlan30'],
    'Safe': ['FloorPlan18', 'FloorPlan204', 'FloorPlan219', 'FloorPlan302', 'FloorPlan308', 'FloorPlan309', 'FloorPlan317', 'FloorPlan323'],
    'ShowerCurtain': ['FloorPlan401', 'FloorPlan404', 'FloorPlan405', 'FloorPlan407', 'FloorPlan408', 'FloorPlan415', 'FloorPlan419', 'FloorPlan420', 'FloorPlan422', 'FloorPlan423', 'FloorPlan426', 'FloorPlan427', 'FloorPlan429'],
    'ShowerDoor': ['FloorPlan402', 'FloorPlan403', 'FloorPlan407', 'FloorPlan409', 'FloorPlan410', 'FloorPlan411', 'FloorPlan412', 'FloorPlan413', 'FloorPlan416', 'FloorPlan417', 'FloorPlan418', 'FloorPlan421', 'FloorPlan423', 'FloorPlan424', 'FloorPlan425', 'FloorPlan428', 'FloorPlan430'],
    'Toilet': ['FloorPlan401', 'FloorPlan402', 'FloorPlan403', 'FloorPlan404', 'FloorPlan405', 'FloorPlan406', 'FloorPlan407', 'FloorPlan408', 'FloorPlan409', 'FloorPlan410', 'FloorPlan411', 'FloorPlan412', 'FloorPlan413', 'FloorPlan414', 'FloorPlan415', 'FloorPlan416', 'FloorPlan417', 'FloorPlan418', 'FloorPlan419', 'FloorPlan420', 'FloorPlan421', 'FloorPlan422', 'FloorPlan423', 'FloorPlan424', 'FloorPlan425', 'FloorPlan426', 'FloorPlan427', 'FloorPlan428', 'FloorPlan429', 'FloorPlan430'],
}

RECEPTACLE_OBJECTS_TO_EXISTING_SCENES = {
    'ArmChair': ['FloorPlan201', 'FloorPlan202', 'FloorPlan203', 'FloorPlan204', 'FloorPlan205', 'FloorPlan206', 'FloorPlan207', 'FloorPlan208', 'FloorPlan209', 'FloorPlan210', 'FloorPlan211', 'FloorPlan212', 'FloorPlan213', 'FloorPlan214', 'FloorPlan216', 'FloorPlan217', 'FloorPlan219', 'FloorPlan220', 'FloorPlan221', 'FloorPlan222', 'FloorPlan223', 'FloorPlan224', 'FloorPlan225', 'FloorPlan226', 'FloorPlan227', 'FloorPlan229', 'FloorPlan230', 'FloorPlan309', 'FloorPlan311', 'FloorPlan318', 'FloorPlan321', 'FloorPlan322', 'FloorPlan330'],
    'Bathtub': ['FloorPlan401', 'FloorPlan402', 'FloorPlan403', 'FloorPlan404', 'FloorPlan405', 'FloorPlan406', 'FloorPlan407', 'FloorPlan408', 'FloorPlan410', 'FloorPlan411', 'FloorPlan413', 'FloorPlan414', 'FloorPlan415', 'FloorPlan419', 'FloorPlan420', 'FloorPlan422', 'FloorPlan423', 'FloorPlan426', 'FloorPlan427', 'FloorPlan429', 'FloorPlan430'],
    'BathtubBasin': ['FloorPlan401', 'FloorPlan402', 'FloorPlan403', 'FloorPlan404', 'FloorPlan405', 'FloorPlan406', 'FloorPlan407', 'FloorPlan408', 'FloorPlan410', 'FloorPlan411', 'FloorPlan413', 'FloorPlan414', 'FloorPlan415', 'FloorPlan419', 'FloorPlan422', 'FloorPlan423', 'FloorPlan426', 'FloorPlan427', 'FloorPlan429'],
    'Bed': ['FloorPlan301', 'FloorPlan302', 'FloorPlan303', 'FloorPlan304', 'FloorPlan305', 'FloorPlan306', 'FloorPlan307', 'FloorPlan308', 'FloorPlan309', 'FloorPlan310', 'FloorPlan311', 'FloorPlan312', 'FloorPlan313', 'FloorPlan314', 'FloorPlan315', 'FloorPlan316', 'FloorPlan317', 'FloorPlan318', 'FloorPlan319', 'FloorPlan320', 'FloorPlan321', 'FloorPlan322', 'FloorPlan323', 'FloorPlan324', 'FloorPlan325', 'FloorPlan326', 'FloorPlan327', 'FloorPlan328', 'FloorPlan329', 'FloorPlan330'],
    'Cabinet': ['FloorPlan1', 'FloorPlan2', 'FloorPlan3', 'FloorPlan4', 'FloorPlan5', 'FloorPlan6', 'FloorPlan7', 'FloorPlan8', 'FloorPlan9', 'FloorPlan10', 'FloorPlan11', 'FloorPlan12', 'FloorPlan13', 'FloorPlan14', 'FloorPlan15', 'FloorPlan16', 'FloorPlan17', 'FloorPlan18', 'FloorPlan19', 'FloorPlan20', 'FloorPlan21', 'FloorPlan22', 'FloorPlan23', 'FloorPlan24', 'FloorPlan25', 'FloorPlan26', 'FloorPlan27', 'FloorPlan28', 'FloorPlan29', 'FloorPlan30', 'FloorPlan206', 'FloorPlan207', 'FloorPlan217', 'FloorPlan219', 'FloorPlan224', 'FloorPlan227', 'FloorPlan309', 'FloorPlan310', 'FloorPlan318', 'FloorPlan322', 'FloorPlan325', 'FloorPlan402', 'FloorPlan403', 'FloorPlan405', 'FloorPlan406', 'FloorPlan407', 'FloorPlan408', 'FloorPlan412', 'FloorPlan413', 'FloorPlan414', 'FloorPlan417', 'FloorPlan418', 'FloorPlan422', 'FloorPlan424', 'FloorPlan425', 'FloorPlan428'],
    'Chair': ['FloorPlan2', 'FloorPlan7', 'FloorPlan9', 'FloorPlan10', 'FloorPlan14', 'FloorPlan16', 'FloorPlan18', 'FloorPlan19', 'FloorPlan21', 'FloorPlan23', 'FloorPlan27', 'FloorPlan29', 'FloorPlan201', 'FloorPlan203', 'FloorPlan204', 'FloorPlan205', 'FloorPlan208', 'FloorPlan210', 'FloorPlan213', 'FloorPlan215', 'FloorPlan216', 'FloorPlan217', 'FloorPlan218', 'FloorPlan220', 'FloorPlan221', 'FloorPlan223', 'FloorPlan224', 'FloorPlan225', 'FloorPlan227', 'FloorPlan228', 'FloorPlan229', 'FloorPlan230', 'FloorPlan301', 'FloorPlan302', 'FloorPlan303', 'FloorPlan304', 'FloorPlan305', 'FloorPlan306', 'FloorPlan307', 'FloorPlan308', 'FloorPlan309', 'FloorPlan310', 'FloorPlan311', 'FloorPlan312', 'FloorPlan313', 'FloorPlan314', 'FloorPlan316', 'FloorPlan318', 'FloorPlan320', 'FloorPlan321', 'FloorPlan323', 'FloorPlan326', 'FloorPlan327', 'FloorPlan328'],
    'CoffeeMachine': ['FloorPlan1', 'FloorPlan2', 'FloorPlan3', 'FloorPlan4', 'FloorPlan5', 'FloorPlan6', 'FloorPlan7', 'FloorPlan8', 'FloorPlan9', 'FloorPlan10', 'FloorPlan11', 'FloorPlan12', 'FloorPlan13', 'FloorPlan14', 'FloorPlan15', 'FloorPlan16', 'FloorPlan17', 'FloorPlan18', 'FloorPlan19', 'FloorPlan20', 'FloorPlan21', 'FloorPlan22', 'FloorPlan23', 'FloorPlan24', 'FloorPlan25', 'FloorPlan26', 'FloorPlan27', 'FloorPlan28', 'FloorPlan29', 'FloorPlan30'],
    'CoffeeTable': ['FloorPlan201', 'FloorPlan202', 'FloorPlan203', 'FloorPlan204', 'FloorPlan206', 'FloorPlan207', 'FloorPlan209', 'FloorPlan211', 'FloorPlan212', 'FloorPlan213', 'FloorPlan214', 'FloorPlan215', 'FloorPlan217', 'FloorPlan218', 'FloorPlan220', 'FloorPlan221', 'FloorPlan222', 'FloorPlan223', 'FloorPlan225', 'FloorPlan226', 'FloorPlan227', 'FloorPlan228', 'FloorPlan229', 'FloorPlan230', 'FloorPlan315'],
    'CounterTop': ['FloorPlan1', 'FloorPlan2', 'FloorPlan3', 'FloorPlan4', 'FloorPlan5', 'FloorPlan6', 'FloorPlan7', 'FloorPlan8', 'FloorPlan9', 'FloorPlan10', 'FloorPlan11', 'FloorPlan12', 'FloorPlan13', 'FloorPlan14', 'FloorPlan15', 'FloorPlan16', 'FloorPlan17', 'FloorPlan18', 'FloorPlan19', 'FloorPlan20', 'FloorPlan21', 'FloorPlan22', 'FloorPlan23', 'FloorPlan24', 'FloorPlan25', 'FloorPlan26', 'FloorPlan27', 'FloorPlan28', 'FloorPlan29', 'FloorPlan30', 'FloorPlan311', 'FloorPlan402', 'FloorPlan403', 'FloorPlan405', 'FloorPlan406', 'FloorPlan407', 'FloorPlan408', 'FloorPlan409', 'FloorPlan410', 'FloorPlan411', 'FloorPlan412', 'FloorPlan413', 'FloorPlan414', 'FloorPlan415', 'FloorPlan416', 'FloorPlan417', 'FloorPlan418', 'FloorPlan421', 'FloorPlan422', 'FloorPlan423', 'FloorPlan424', 'FloorPlan425', 'FloorPlan426', 'FloorPlan427', 'FloorPlan428', 'FloorPlan430'],
    'Desk': ['FloorPlan204', 'FloorPlan229', 'FloorPlan301', 'FloorPlan302', 'FloorPlan303', 'FloorPlan304', 'FloorPlan305', 'FloorPlan306', 'FloorPlan307', 'FloorPlan308', 'FloorPlan309', 'FloorPlan310', 'FloorPlan311', 'FloorPlan312', 'FloorPlan313', 'FloorPlan314', 'FloorPlan315', 'FloorPlan316', 'FloorPlan318', 'FloorPlan320', 'FloorPlan321', 'FloorPlan323', 'FloorPlan326', 'FloorPlan327', 'FloorPlan328', 'FloorPlan329'],
    'DiningTable': ['FloorPlan4', 'FloorPlan7', 'FloorPlan9', 'FloorPlan11', 'FloorPlan15', 'FloorPlan16', 'FloorPlan17', 'FloorPlan18', 'FloorPlan19', 'FloorPlan20', 'FloorPlan21', 'FloorPlan23', 'FloorPlan24', 'FloorPlan26', 'FloorPlan27', 'FloorPlan28', 'FloorPlan201', 'FloorPlan203', 'FloorPlan204', 'FloorPlan205', 'FloorPlan208', 'FloorPlan211', 'FloorPlan216', 'FloorPlan218', 'FloorPlan220', 'FloorPlan221', 'FloorPlan223', 'FloorPlan227', 'FloorPlan228', 'FloorPlan230'],
    'DogBed': ['FloorPlan210', 'FloorPlan224', 'FloorPlan301'],
    'Drawer': ['FloorPlan1', 'FloorPlan2', 'FloorPlan3', 'FloorPlan4', 'FloorPlan5', 'FloorPlan6', 'FloorPlan7', 'FloorPlan8', 'FloorPlan9', 'FloorPlan10', 'FloorPlan11', 'FloorPlan12', 'FloorPlan13', 'FloorPlan15', 'FloorPlan16', 'FloorPlan17', 'FloorPlan18', 'FloorPlan19', 'FloorPlan20', 'FloorPlan21', 'FloorPlan22', 'FloorPlan23', 'FloorPlan24', 'FloorPlan25', 'FloorPlan26', 'FloorPlan27', 'FloorPlan28', 'FloorPlan30', 'FloorPlan201', 'FloorPlan203', 'FloorPlan204', 'FloorPlan205', 'FloorPlan208', 'FloorPlan209', 'FloorPlan210', 'FloorPlan211', 'FloorPlan212', 'FloorPlan213', 'FloorPlan216', 'FloorPlan217', 'FloorPlan218', 'FloorPlan219', 'FloorPlan220', 'FloorPlan221', 'FloorPlan222', 'FloorPlan224', 'FloorPlan225', 'FloorPlan226', 'FloorPlan227', 'FloorPlan228', 'FloorPlan229', 'FloorPlan301', 'FloorPlan302', 'FloorPlan303', 'FloorPlan304', 'FloorPlan305', 'FloorPlan306', 'FloorPlan307', 'FloorPlan308', 'FloorPlan309', 'FloorPlan310', 'FloorPlan311', 'FloorPlan312', 'FloorPlan313', 'FloorPlan314', 'FloorPlan315', 'FloorPlan316', 'FloorPlan317', 'FloorPlan318', 'FloorPlan319', 'FloorPlan321', 'FloorPlan322', 'FloorPlan323', 'FloorPlan324', 'FloorPlan325', 'FloorPlan326', 'FloorPlan327', 'FloorPlan328', 'FloorPlan329', 'FloorPlan330', 'FloorPlan409', 'FloorPlan410', 'FloorPlan411', 'FloorPlan413', 'FloorPlan414', 'FloorPlan415', 'FloorPlan416', 'FloorPlan419', 'FloorPlan420', 'FloorPlan421', 'FloorPlan422', 'FloorPlan423', 'FloorPlan426', 'FloorPlan427', 'FloorPlan430'],
    'Dresser': ['FloorPlan205', 'FloorPlan209', 'FloorPlan210', 'FloorPlan213', 'FloorPlan217', 'FloorPlan218', 'FloorPlan219', 'FloorPlan220', 'FloorPlan224', 'FloorPlan229', 'FloorPlan301', 'FloorPlan311', 'FloorPlan315', 'FloorPlan317', 'FloorPlan318', 'FloorPlan319', 'FloorPlan322', 'FloorPlan323', 'FloorPlan324', 'FloorPlan325', 'FloorPlan330', 'FloorPlan413', 'FloorPlan415'],
    'Floor': ['FloorPlan1', 'FloorPlan2', 'FloorPlan3', 'FloorPlan4', 'FloorPlan5', 'FloorPlan6', 'FloorPlan7', 'FloorPlan8', 'FloorPlan9', 'FloorPlan10', 'FloorPlan11', 'FloorPlan12', 'FloorPlan13', 'FloorPlan14', 'FloorPlan15', 'FloorPlan16', 'FloorPlan17', 'FloorPlan18', 'FloorPlan19', 'FloorPlan20', 'FloorPlan21', 'FloorPlan22', 'FloorPlan23', 'FloorPlan24', 'FloorPlan25', 'FloorPlan26', 'FloorPlan27', 'FloorPlan28', 'FloorPlan29', 'FloorPlan30', 'FloorPlan201', 'FloorPlan202', 'FloorPlan203', 'FloorPlan204', 'FloorPlan205', 'FloorPlan206', 'FloorPlan207', 'FloorPlan208', 'FloorPlan209', 'FloorPlan210', 'FloorPlan211', 'FloorPlan212', 'FloorPlan213', 'FloorPlan214', 'FloorPlan215', 'FloorPlan216', 'FloorPlan217', 'FloorPlan218', 'FloorPlan219', 'FloorPlan220', 'FloorPlan221', 'FloorPlan222', 'FloorPlan223', 'FloorPlan224', 'FloorPlan225', 'FloorPlan226', 'FloorPlan227', 'FloorPlan228', 'FloorPlan229', 'FloorPlan230', 'FloorPlan301', 'FloorPlan302', 'FloorPlan303', 'FloorPlan304', 'FloorPlan305', 'FloorPlan306', 'FloorPlan307', 'FloorPlan308', 'FloorPlan309', 'FloorPlan310', 'FloorPlan311', 'FloorPlan312', 'FloorPlan313', 'FloorPlan314', 'FloorPlan315', 'FloorPlan316', 'FloorPlan317', 'FloorPlan318', 'FloorPlan319', 'FloorPlan320', 'FloorPlan321', 'FloorPlan322', 'FloorPlan323', 'FloorPlan324', 'FloorPlan325', 'FloorPlan326', 'FloorPlan327', 'FloorPlan328', 'FloorPlan329', 'FloorPlan330', 'FloorPlan401', 'FloorPlan402', 'FloorPlan403', 'FloorPlan404', 'FloorPlan405', 'FloorPlan406', 'FloorPlan407', 'FloorPlan408', 'FloorPlan409', 'FloorPlan410', 'FloorPlan411', 'FloorPlan412', 'FloorPlan413', 'FloorPlan414', 'FloorPlan415', 'FloorPlan416', 'FloorPlan417', 'FloorPlan418', 'FloorPlan419', 'FloorPlan420', 'FloorPlan421', 'FloorPlan422', 'FloorPlan423', 'FloorPlan424', 'FloorPlan425', 'FloorPlan426', 'FloorPlan427', 'FloorPlan428', 'FloorPlan429', 'FloorPlan430'],
    'Fridge': ['FloorPlan1', 'FloorPlan2', 'FloorPlan3', 'FloorPlan4', 'FloorPlan5', 'FloorPlan6', 'FloorPlan7', 'FloorPlan8', 'FloorPlan9', 'FloorPlan10', 'FloorPlan11', 'FloorPlan12', 'FloorPlan13', 'FloorPlan14', 'FloorPlan15', 'FloorPlan16', 'FloorPlan17', 'FloorPlan18', 'FloorPlan19', 'FloorPlan20', 'FloorPlan21', 'FloorPlan22', 'FloorPlan23', 'FloorPlan24', 'FloorPlan25', 'FloorPlan26', 'FloorPlan27', 'FloorPlan28', 'FloorPlan29', 'FloorPlan30'],
    'GarbageCan': ['FloorPlan1', 'FloorPlan2', 'FloorPlan3', 'FloorPlan4', 'FloorPlan5', 'FloorPlan6', 'FloorPlan7', 'FloorPlan8', 'FloorPlan9', 'FloorPlan10', 'FloorPlan11', 'FloorPlan12', 'FloorPlan13', 'FloorPlan14', 'FloorPlan15', 'FloorPlan16', 'FloorPlan17', 'FloorPlan18', 'FloorPlan19', 'FloorPlan20', 'FloorPlan21', 'FloorPlan22', 'FloorPlan23', 'FloorPlan24', 'FloorPlan25', 'FloorPlan26', 'FloorPlan27', 'FloorPlan28', 'FloorPlan29', 'FloorPlan30', 'FloorPlan201', 'FloorPlan202', 'FloorPlan203', 'FloorPlan204', 'FloorPlan205', 'FloorPlan206', 'FloorPlan207', 'FloorPlan208', 'FloorPlan209', 'FloorPlan210', 'FloorPlan211', 'FloorPlan212', 'FloorPlan213', 'FloorPlan214', 'FloorPlan215', 'FloorPlan216', 'FloorPlan217', 'FloorPlan218', 'FloorPlan219', 'FloorPlan220', 'FloorPlan221', 'FloorPlan222', 'FloorPlan223', 'FloorPlan224', 'FloorPlan225', 'FloorPlan226', 'FloorPlan227', 'FloorPlan228', 'FloorPlan229', 'FloorPlan230', 'FloorPlan301', 'FloorPlan302', 'FloorPlan303', 'FloorPlan304', 'FloorPlan305', 'FloorPlan306', 'FloorPlan307', 'FloorPlan308', 'FloorPlan309', 'FloorPlan310', 'FloorPlan311', 'FloorPlan312', 'FloorPlan313', 'FloorPlan314', 'FloorPlan315', 'FloorPlan316', 'FloorPlan317', 'FloorPlan318', 'FloorPlan319', 'FloorPlan320', 'FloorPlan321', 'FloorPlan322', 'FloorPlan323', 'FloorPlan324', 'FloorPlan325', 'FloorPlan326', 'FloorPlan327', 'FloorPlan328', 'FloorPlan329', 'FloorPlan330', 'FloorPlan401', 'FloorPlan402', 'FloorPlan403', 'FloorPlan404', 'FloorPlan405', 'FloorPlan406', 'FloorPlan407', 'FloorPlan408', 'FloorPlan409', 'FloorPlan410', 'FloorPlan411', 'FloorPlan412', 'FloorPlan413', 'FloorPlan414', 'FloorPlan415', 'FloorPlan416', 'FloorPlan417', 'FloorPlan418', 'FloorPlan419', 'FloorPlan420', 'FloorPlan421', 'FloorPlan422', 'FloorPlan423', 'FloorPlan424', 'FloorPlan425', 'FloorPlan426', 'FloorPlan427', 'FloorPlan428', 'FloorPlan429', 'FloorPlan430'],
    'HandTowelHolder': ['FloorPlan401', 'FloorPlan402', 'FloorPlan403', 'FloorPlan404', 'FloorPlan405', 'FloorPlan406', 'FloorPlan407', 'FloorPlan408', 'FloorPlan409', 'FloorPlan410', 'FloorPlan411', 'FloorPlan412', 'FloorPlan413', 'FloorPlan414', 'FloorPlan415', 'FloorPlan416', 'FloorPlan417', 'FloorPlan418', 'FloorPlan419', 'FloorPlan420', 'FloorPlan421', 'FloorPlan422', 'FloorPlan423', 'FloorPlan424', 'FloorPlan425', 'FloorPlan426', 'FloorPlan427', 'FloorPlan428', 'FloorPlan429', 'FloorPlan430'],
    'LaundryHamper': ['FloorPlan304', 'FloorPlan305', 'FloorPlan306', 'FloorPlan307', 'FloorPlan308', 'FloorPlan309', 'FloorPlan311', 'FloorPlan317'],
    'Microwave': ['FloorPlan1', 'FloorPlan2', 'FloorPlan3', 'FloorPlan4', 'FloorPlan5', 'FloorPlan6', 'FloorPlan7', 'FloorPlan8', 'FloorPlan9', 'FloorPlan10', 'FloorPlan11', 'FloorPlan12', 'FloorPlan13', 'FloorPlan14', 'FloorPlan15', 'FloorPlan16', 'FloorPlan17', 'FloorPlan18', 'FloorPlan19', 'FloorPlan20', 'FloorPlan21', 'FloorPlan22', 'FloorPlan23', 'FloorPlan24', 'FloorPlan25', 'FloorPlan26', 'FloorPlan27', 'FloorPlan28', 'FloorPlan29', 'FloorPlan30'],
    'Ottoman': ['FloorPlan203', 'FloorPlan205', 'FloorPlan208', 'FloorPlan210'],
    'Safe': ['FloorPlan18', 'FloorPlan204', 'FloorPlan219', 'FloorPlan302', 'FloorPlan308', 'FloorPlan309', 'FloorPlan317', 'FloorPlan323'],
    'Shelf': ['FloorPlan1', 'FloorPlan5', 'FloorPlan7', 'FloorPlan10', 'FloorPlan17', 'FloorPlan20', 'FloorPlan21', 'FloorPlan23', 'FloorPlan28', 'FloorPlan201', 'FloorPlan202', 'FloorPlan204', 'FloorPlan206', 'FloorPlan207', 'FloorPlan208', 'FloorPlan209', 'FloorPlan211', 'FloorPlan212', 'FloorPlan213', 'FloorPlan214', 'FloorPlan215', 'FloorPlan216', 'FloorPlan219', 'FloorPlan225', 'FloorPlan227', 'FloorPlan301', 'FloorPlan302', 'FloorPlan303', 'FloorPlan304', 'FloorPlan305', 'FloorPlan306', 'FloorPlan307', 'FloorPlan308', 'FloorPlan310', 'FloorPlan312', 'FloorPlan313', 'FloorPlan316', 'FloorPlan318', 'FloorPlan319', 'FloorPlan320', 'FloorPlan322', 'FloorPlan324', 'FloorPlan325', 'FloorPlan326', 'FloorPlan327', 'FloorPlan328', 'FloorPlan329', 'FloorPlan330', 'FloorPlan401', 'FloorPlan402', 'FloorPlan403', 'FloorPlan404', 'FloorPlan415', 'FloorPlan422', 'FloorPlan430'],
    'ShelvingUnit': ['FloorPlan1', 'FloorPlan5', 'FloorPlan7', 'FloorPlan10', 'FloorPlan20', 'FloorPlan206', 'FloorPlan207', 'FloorPlan211', 'FloorPlan215', 'FloorPlan219', 'FloorPlan303', 'FloorPlan307', 'FloorPlan313', 'FloorPlan319', 'FloorPlan324', 'FloorPlan325', 'FloorPlan330'],
    'SideTable': ['FloorPlan3', 'FloorPlan21', 'FloorPlan28', 'FloorPlan201', 'FloorPlan202', 'FloorPlan203', 'FloorPlan205', 'FloorPlan207', 'FloorPlan208', 'FloorPlan209', 'FloorPlan210', 'FloorPlan211', 'FloorPlan212', 'FloorPlan213', 'FloorPlan214', 'FloorPlan215', 'FloorPlan216', 'FloorPlan217', 'FloorPlan218', 'FloorPlan219', 'FloorPlan220', 'FloorPlan221', 'FloorPlan222', 'FloorPlan223', 'FloorPlan224', 'FloorPlan225', 'FloorPlan226', 'FloorPlan228', 'FloorPlan229', 'FloorPlan230', 'FloorPlan302', 'FloorPlan303', 'FloorPlan305', 'FloorPlan306', 'FloorPlan307', 'FloorPlan309', 'FloorPlan310', 'FloorPlan311', 'FloorPlan312', 'FloorPlan313', 'FloorPlan314', 'FloorPlan316', 'FloorPlan317', 'FloorPlan320', 'FloorPlan321', 'FloorPlan322', 'FloorPlan323', 'FloorPlan325', 'FloorPlan326', 'FloorPlan327', 'FloorPlan328', 'FloorPlan329', 'FloorPlan330', 'FloorPlan401', 'FloorPlan419', 'FloorPlan420', 'FloorPlan429', 'FloorPlan430'],
    'Sink': ['FloorPlan1', 'FloorPlan2', 'FloorPlan3', 'FloorPlan4', 'FloorPlan5', 'FloorPlan6', 'FloorPlan7', 'FloorPlan8', 'FloorPlan9', 'FloorPlan10', 'FloorPlan11', 'FloorPlan12', 'FloorPlan13', 'FloorPlan14', 'FloorPlan15', 'FloorPlan16', 'FloorPlan17', 'FloorPlan18', 'FloorPlan19', 'FloorPlan20', 'FloorPlan21', 'FloorPlan22', 'FloorPlan23', 'FloorPlan24', 'FloorPlan25', 'FloorPlan26', 'FloorPlan27', 'FloorPlan28', 'FloorPlan29', 'FloorPlan30', 'FloorPlan401', 'FloorPlan402', 'FloorPlan403', 'FloorPlan404', 'FloorPlan405', 'FloorPlan406', 'FloorPlan407', 'FloorPlan408', 'FloorPlan409', 'FloorPlan410', 'FloorPlan411', 'FloorPlan412', 'FloorPlan413', 'FloorPlan414', 'FloorPlan415', 'FloorPlan416', 'FloorPlan417', 'FloorPlan418', 'FloorPlan419', 'FloorPlan420', 'FloorPlan421', 'FloorPlan422', 'FloorPlan423', 'FloorPlan424', 'FloorPlan425', 'FloorPlan426', 'FloorPlan427', 'FloorPlan428', 'FloorPlan429', 'FloorPlan430'],
    'SinkBasin': ['FloorPlan1', 'FloorPlan2', 'FloorPlan3', 'FloorPlan4', 'FloorPlan5', 'FloorPlan6', 'FloorPlan7', 'FloorPlan8', 'FloorPlan9', 'FloorPlan10', 'FloorPlan11', 'FloorPlan12', 'FloorPlan13', 'FloorPlan14', 'FloorPlan15', 'FloorPlan16', 'FloorPlan17', 'FloorPlan18', 'FloorPlan19', 'FloorPlan20', 'FloorPlan21', 'FloorPlan22', 'FloorPlan23', 'FloorPlan24', 'FloorPlan25', 'FloorPlan26', 'FloorPlan27', 'FloorPlan28', 'FloorPlan29', 'FloorPlan30', 'FloorPlan401', 'FloorPlan402', 'FloorPlan403', 'FloorPlan404', 'FloorPlan405', 'FloorPlan406', 'FloorPlan407', 'FloorPlan408', 'FloorPlan409', 'FloorPlan410', 'FloorPlan411', 'FloorPlan412', 'FloorPlan413', 'FloorPlan414', 'FloorPlan415', 'FloorPlan416', 'FloorPlan417', 'FloorPlan418', 'FloorPlan419', 'FloorPlan420', 'FloorPlan421', 'FloorPlan422', 'FloorPlan423', 'FloorPlan424', 'FloorPlan425', 'FloorPlan426', 'FloorPlan427', 'FloorPlan428', 'FloorPlan429', 'FloorPlan430'],
    'Sofa': ['FloorPlan201', 'FloorPlan202', 'FloorPlan203', 'FloorPlan204', 'FloorPlan205', 'FloorPlan206', 'FloorPlan207', 'FloorPlan208', 'FloorPlan209', 'FloorPlan210', 'FloorPlan211', 'FloorPlan212', 'FloorPlan213', 'FloorPlan214', 'FloorPlan215', 'FloorPlan216', 'FloorPlan217', 'FloorPlan218', 'FloorPlan219', 'FloorPlan220', 'FloorPlan221', 'FloorPlan222', 'FloorPlan223', 'FloorPlan224', 'FloorPlan225', 'FloorPlan226', 'FloorPlan227', 'FloorPlan228', 'FloorPlan229', 'FloorPlan230', 'FloorPlan323'],
    'Stool': ['FloorPlan1', 'FloorPlan3', 'FloorPlan5', 'FloorPlan6', 'FloorPlan7', 'FloorPlan8', 'FloorPlan11', 'FloorPlan12', 'FloorPlan15', 'FloorPlan17', 'FloorPlan20', 'FloorPlan22', 'FloorPlan24', 'FloorPlan25', 'FloorPlan221', 'FloorPlan224', 'FloorPlan319'],
    'StoveBurner': ['FloorPlan1', 'FloorPlan2', 'FloorPlan3', 'FloorPlan4', 'FloorPlan5', 'FloorPlan6', 'FloorPlan7', 'FloorPlan8', 'FloorPlan9', 'FloorPlan10', 'FloorPlan11', 'FloorPlan12', 'FloorPlan13', 'FloorPlan14', 'FloorPlan15', 'FloorPlan16', 'FloorPlan17', 'FloorPlan18', 'FloorPlan19', 'FloorPlan20', 'FloorPlan21', 'FloorPlan22', 'FloorPlan23', 'FloorPlan24', 'FloorPlan25', 'FloorPlan26', 'FloorPlan27', 'FloorPlan28', 'FloorPlan29', 'FloorPlan30'],
    'TVStand': ['FloorPlan201', 'FloorPlan202', 'FloorPlan204', 'FloorPlan206', 'FloorPlan207', 'FloorPlan208', 'FloorPlan211', 'FloorPlan212', 'FloorPlan220', 'FloorPlan222', 'FloorPlan223', 'FloorPlan225'],
    'Toaster': ['FloorPlan1', 'FloorPlan2', 'FloorPlan3', 'FloorPlan4', 'FloorPlan5', 'FloorPlan6', 'FloorPlan7', 'FloorPlan8', 'FloorPlan9', 'FloorPlan10', 'FloorPlan11', 'FloorPlan12', 'FloorPlan13', 'FloorPlan14', 'FloorPlan15', 'FloorPlan16', 'FloorPlan17', 'FloorPlan18', 'FloorPlan19', 'FloorPlan20', 'FloorPlan21', 'FloorPlan22', 'FloorPlan23', 'FloorPlan24', 'FloorPlan25', 'FloorPlan26', 'FloorPlan27', 'FloorPlan28', 'FloorPlan29', 'FloorPlan30'],
    'Toilet': ['FloorPlan401', 'FloorPlan402', 'FloorPlan403', 'FloorPlan404', 'FloorPlan405', 'FloorPlan406', 'FloorPlan407', 'FloorPlan408', 'FloorPlan409', 'FloorPlan410', 'FloorPlan411', 'FloorPlan412', 'FloorPlan413', 'FloorPlan414', 'FloorPlan415', 'FloorPlan416', 'FloorPlan417', 'FloorPlan418', 'FloorPlan419', 'FloorPlan420', 'FloorPlan421', 'FloorPlan422', 'FloorPlan423', 'FloorPlan424', 'FloorPlan425', 'FloorPlan426', 'FloorPlan427', 'FloorPlan428', 'FloorPlan429', 'FloorPlan430'],
    'ToiletPaperHanger': ['FloorPlan401', 'FloorPlan402', 'FloorPlan403', 'FloorPlan404', 'FloorPlan405', 'FloorPlan406', 'FloorPlan407', 'FloorPlan408', 'FloorPlan409', 'FloorPlan410', 'FloorPlan411', 'FloorPlan412', 'FloorPlan413', 'FloorPlan414', 'FloorPlan415', 'FloorPlan416', 'FloorPlan417', 'FloorPlan418', 'FloorPlan419', 'FloorPlan420', 'FloorPlan421', 'FloorPlan422', 'FloorPlan423', 'FloorPlan424', 'FloorPlan425', 'FloorPlan426', 'FloorPlan427', 'FloorPlan428', 'FloorPlan429', 'FloorPlan430'],
    'TowelHolder': ['FloorPlan401', 'FloorPlan402', 'FloorPlan403', 'FloorPlan404', 'FloorPlan405', 'FloorPlan406', 'FloorPlan407', 'FloorPlan408', 'FloorPlan409', 'FloorPlan410', 'FloorPlan411', 'FloorPlan412', 'FloorPlan413', 'FloorPlan414', 'FloorPlan415', 'FloorPlan416', 'FloorPlan417', 'FloorPlan418', 'FloorPlan419', 'FloorPlan420', 'FloorPlan421', 'FloorPlan422', 'FloorPlan423', 'FloorPlan424', 'FloorPlan425', 'FloorPlan426', 'FloorPlan427', 'FloorPlan428', 'FloorPlan429', 'FloorPlan430'],
}