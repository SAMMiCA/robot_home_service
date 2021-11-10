import os
from pathlib import Path

VISIBILITY_DISTANCE = 1.0
MAX_HAND_METERS = 0.5
FOV = 90

REQUIRED_THOR_VERSION = "2.7.2"
STARTER_REARRANGE_DATA_DIR = os.path.join(
    os.path.abspath(os.path.dirname(Path(__file__))), "../data/rearrange"
)
STARTER_HOME_SERVICE_DATA_DIR = os.path.join(
    os.path.abspath(os.path.dirname(Path(__file__))), "../data/home_service"
)
STARTER_HOME_SERVICE_SIMPLE_PICK_AND_PLACE_DATA_DIR = os.path.join(
    os.path.abspath(os.path.dirname(Path(__file__))), "../data/home_service/simple_pick_and_place"
)

THOR_COMMIT_ID = "f46d5ec42b65fdae9d9a48db2b4fb6d25afbd1fe"
STEP_SIZE = 0.25
ROTATION_ANGLE = 90
HORIZON_ANGLE = 30
SMOOTHING_FACTOR = 1

KITCHENS = [f"FloorPlan{i}" for i in range(1, 31)]
LIVING_ROOMS = [f"FloorPlan{200 + i}" for i in range(1, 31)]
BEDROOMS = [f"FloorPlan{300 + i}" for i in range(1, 31)]
BATHROOMS = [f"FloorPlan{400 + i}" for i in range(1, 31)]

SCENE_TYPE_TO_SCENES = {
    "Kitchen": KITCHENS,
    "LivingRoom": LIVING_ROOMS,
    "Bedroom": BEDROOMS,
    "Bathroom": BATHROOMS,
}

SCENE_TO_SCENE_TYPE = {
    scene: scene_type
    for scene_type, scenes in SCENE_TYPE_TO_SCENES.items()
    for scene in scenes
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
    "AppleSliced": ["Pot", "Pan", "Bowl", "Microwave", "Fridge", "Plate", "Sink", "SinkBasin", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "Desk", "CounterTop", "GarbageCan", "Dresser"], 
    "BaseballBat": ["Bed", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "Desk", "CounterTop"], 
    "BasketBall": ["Sofa", "ArmChair", "Dresser", "Desk", "Bed", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop"], 
    "Book": ["Sofa", "ArmChair", "Box", "Ottoman", "Dresser", "Desk", "Bed", "Cabinet", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Shelf", "Drawer"], 
    "Bottle": ["Fridge", "Box", "Dresser", "Desk", "Sink", "SinkBasin", "Cabinet", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Shelf", "GarbageCan"], 
    "Bowl": ["Microwave", "Fridge", "Dresser", "Desk", "Sink", "SinkBasin", "Cabinet", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Shelf"], 
    "Box": ["Sofa", "ArmChair", "Dresser", "Desk", "Cabinet", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Shelf", "Ottoman"], 
    "Bread": ["Microwave", "Fridge", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "Desk", "CounterTop", "GarbageCan", "Plate"], 
    "BreadSliced": ["Microwave", "Fridge", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "Desk", "CounterTop", "GarbageCan", "Toaster", "Plate"], 
    "ButterKnife": ["Pot", "Pan", "Bowl", "Mug", "Plate", "Cup", "Sink", "SinkBasin", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "Desk", "CounterTop", "Drawer"], 
    "Candle": ["Box", "Dresser", "Desk", "Toilet", "Cart", "Bathtub", "Cabinet", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Shelf", "Drawer"], 
    "CD": ["Box", "Ottoman", "Dresser", "Desk", "Cabinet", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Shelf", "Drawer", "GarbageCan", "Safe", "Sofa", "ArmChair"], 
    "CellPhone": ["Sofa", "ArmChair", "Box", "Ottoman", "Dresser", "Desk", "Bed", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Shelf", "Drawer", "Safe"], 
    "Cloth": ["Sofa", "ArmChair", "Box", "Ottoman", "Dresser", "LaundryHamper", "Desk", "Toilet", "Cart", "BathtubBasin", "Bathtub", "Sink", "SinkBasin", "Cabinet", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Shelf", "Drawer", "GarbageCan"], 
    "CreditCard": ["Sofa", "ArmChair", "Box", "Ottoman", "Dresser", "Desk", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Shelf", "Drawer", "Shelf"], 
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
    "LettuceSliced": ["Pot", "Pan", "Bowl", "Fridge", "Plate", "Sink", "SinkBasin", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "GarbageCan"], 
    "Mug": ["SinkBasin", "Cabinet", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Shelf"], 
    "Newspaper": ["Sofa", "ArmChair", "Ottoman", "Dresser", "Desk", "Bed", "Toilet", "Cabinet", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Shelf", "Drawer", "GarbageCan"], 
    "Pan": ["DiningTable", "CounterTop", "TVStand", "CoffeeTable", "SideTable", "Sink", "SinkBasin", "Cabinet", "StoveBurner", "Fridge"], 
    "Pen": ["Mug", "Box", "Dresser", "Desk", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Shelf", "Drawer", "GarbageCan"], 
    "Pencil": ["Mug", "Box", "Dresser", "Desk", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Shelf", "Drawer", "GarbageCan"], 
    "PepperShaker": ["DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Drawer", "Cabinet", "Shelf"], 
    "Pillow": ["Sofa", "ArmChair", "Ottoman", "Bed"], 
    "Plate": ["Microwave", "Fridge", "Dresser", "Desk", "Sink", "Sink", "SinkBasin", "Cabinet", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Shelf"], 
    "Plunger": ["Cart", "Cabinet"], 
    "Pot": ["StoveBurner", "Fridge", "Sink", "SinkBasin", "Cabinet", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Shelf"], 
    "Potato": ["Pot", "Pan", "Bowl", "Microwave", "Fridge", "Plate", "Sink", "SinkBasin", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "GarbageCan"], 
    "PotatoSliced": ["Pot", "Pan", "Bowl", "Microwave", "Fridge", "Plate", "Sink", "SinkBasin", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "GarbageCan"], 
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
    "TomatoSliced": ["DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Sink", "SinkBasin", "Pot", "Bowl", "Fridge", "GarbageCan", "Plate"], 
    "Towel": ["TowelHolder"], 
    "Vase": ["Box", "Dresser", "Desk", "Cart", "Cabinet", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Shelf", "Safe"], 
    "Watch": ["Box", "Dresser", "Desk", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Shelf", "Drawer", "Safe"], 
    "WateringCan": ["Dresser", "Desk", "Cabinet", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Shelf", "Drawer"], 
    "WineBottle": ["Fridge", "Dresser", "Desk", "Cabinet", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Shelf", "GarbageCan"]
}

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