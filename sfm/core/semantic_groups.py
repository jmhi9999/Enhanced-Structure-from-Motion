"""
Coarse semantic grouping for ADE20K 150 classes
Groups fine-grained classes into broader categories for robust semantic matching
"""

# ADE20K 150 classes grouped for indoor residential environments
# Based on actual SegFormer model class IDs, optimized for home interiors

COARSE_SEMANTIC_GROUPS = {
    # Fixed architectural elements - walls, ceilings, floors
    'walls_structure': [
        0,   # wall
        1,   # building  
        3,   # floor
        5,   # ceiling
        14,  # door
        53,  # stairs
        59,  # stairway
        95,  # bannister
        121, # step
    ],
    
    # Large furniture - sofas, beds, major pieces
    'large_furniture': [
        7,   # bed
        23,  # sofa
        30,  # armchair
        31,  # seat
        49,  # fireplace
        69,  # bench
        117, # cradle
    ],
    
    # Storage furniture - cabinets, shelves, wardrobes
    'storage_furniture': [
        10,  # cabinet
        24,  # shelf
        35,  # wardrobe
        44,  # chest of drawers
        55,  # case
        62,  # bookcase
        99,  # buffet
        112, # basket
    ],
    
    # Tables and work surfaces
    'tables_surfaces': [
        15,  # table
        33,  # desk
        64,  # coffee table
        70,  # countertop
        73,  # kitchen island
        77,  # bar
    ],
    
    # Seating - chairs, stools
    'seating': [
        19,  # chair
        75,  # swivel chair
        97,  # ottoman
        110, # stool
    ],
    
    # Kitchen appliances and fixtures
    'kitchen_appliances': [
        50,  # refrigerator
        47,  # sink
        65,  # toilet
        37,  # bathtub
        71,  # stove
        118, # oven
        124, # microwave
        125, # pot
        129, # dishwasher
        107, # washer
        146, # radiator
    ],
    
    # Lighting fixtures
    'lighting': [
        36,  # lamp
        82,  # light
        85,  # chandelier
        134, # sconce
    ],
    
    # Windows and transparent surfaces
    'windows_glass': [
        8,   # windowpane
        27,  # mirror
        147, # glass
    ],
    
    # Electronics and screens
    'electronics': [
        74,  # computer
        78,  # arcade machine
        89,  # television receiver
        130, # screen
        141, # crt screen
        143, # monitor
        148, # clock
    ],
    
    # Soft furnishings and textiles
    'soft_furnishings': [
        18,  # curtain
        28,  # rug
        39,  # cushion
        57,  # pillow
        81,  # towel
        131, # blanket
    ],
    
    # Decorative objects and art
    'decorative_objects': [
        22,  # painting
        92,  # apparel
        100, # poster
        132, # sculpture
        135, # vase
        142, # plate
        149, # flag
    ],
    
    # Books and media
    'books_media': [
        67,  # book
        144, # bulletin board
    ],
    
    # Kitchen items and containers
    'kitchen_items': [
        98,  # bottle
        115, # bag
        120, # food
        137, # tray
        138, # ashcan
    ],
    
    # Plants and natural elements (indoor)
    'indoor_plants': [
        17,  # plant
        66,  # flower
    ],
    
    # People and living beings
    'living_beings': [
        12,  # person
        126, # animal
    ],
    
    # Bathroom fixtures
    'bathroom_fixtures': [
        145, # shower
    ],
    
    # Doors and screens
    'doors_screens': [
        58,  # screen door
        63,  # blind
    ],
    
    # Base and structural supports
    'supports_bases': [
        40,  # base
        41,  # box
        42,  # column
    ],
    
    # Outdoor elements (visible through windows/doors)
    'outdoor_elements': [
        4,   # tree
        6,   # road
        9,   # grass
        11,  # sidewalk
        13,  # earth
        16,  # mountain
        21,  # water
        25,  # house
        26,  # sea
        29,  # field
        32,  # fence
        34,  # rock
        46,  # sand
        48,  # skyscraper
        52,  # path
        54,  # runway
        60,  # river
        61,  # bridge
        68,  # hill
        72,  # palm
        79,  # hovel
        84,  # tower
        86,  # awning
        87,  # streetlight
        88,  # booth
        93,  # pole
        94,  # land
        96,  # escalator
        103, # ship
        104, # fountain
        109, # swimming pool
        111, # barrel
        113, # waterfall
        114, # tent
        116, # minibike
        122, # tank
        127, # bicycle
        128, # lake
        136, # traffic light
        140, # pier
    ],
    
    # Sky (visible through windows)
    'sky': [
        2,   # sky
    ],
    
    # Vehicles (less common indoors)
    'vehicles': [
        20,  # car
        76,  # boat
        80,  # bus
        83,  # truck
        90,  # airplane
        102, # van
    ],
    
    # Entertainment and games
    'entertainment': [
        56,  # pool table
        108, # plaything
        119, # ball
    ],
    
    # Commercial/industrial items
    'commercial_items': [
        43,  # signboard
        45,  # counter
        51,  # grandstand
        101, # stage
        105, # conveyer belt
        106, # canopy
        123, # trade name
        139, # fan
    ]
}

# Reverse mapping: class_id -> coarse_group_name  
CLASS_TO_COARSE_GROUP = {}
for group_name, class_ids in COARSE_SEMANTIC_GROUPS.items():
    for class_id in class_ids:
        CLASS_TO_COARSE_GROUP[class_id] = group_name

# Fill in any missing classes as 'objects'
for i in range(150):
    if i not in CLASS_TO_COARSE_GROUP:
        CLASS_TO_COARSE_GROUP[i] = 'objects'

def get_coarse_semantic_label(class_id: int) -> str:
    """
    Convert fine-grained ADE20K class ID to coarse semantic group
    
    Args:
        class_id: ADE20K class ID (0-149)
        
    Returns:
        str: Coarse semantic group name
    """
    return CLASS_TO_COARSE_GROUP.get(class_id, 'objects')

def are_semantically_compatible(class_id1: int, class_id2: int) -> bool:
    """
    Check if two ADE20K classes belong to the same coarse semantic group
    
    Args:
        class_id1: First class ID
        class_id2: Second class ID
        
    Returns:
        bool: True if classes are semantically compatible
    """
    return get_coarse_semantic_label(class_id1) == get_coarse_semantic_label(class_id2)

def get_semantic_group_info():
    """Get information about semantic groups"""
    info = {}
    for group_name, class_ids in COARSE_SEMANTIC_GROUPS.items():
        info[group_name] = {
            'count': len(class_ids),
            'classes': class_ids
        }
    return info

if __name__ == "__main__":
    # Test the groupings
    print("Coarse Semantic Groups:")
    print(f"Total groups: {len(COARSE_SEMANTIC_GROUPS)}")
    
    for group_name, class_ids in COARSE_SEMANTIC_GROUPS.items():
        print(f"\n{group_name.upper()}: {len(class_ids)} classes")
        print(f"  Classes: {class_ids[:10]}{'...' if len(class_ids) > 10 else ''}")
    
    print(f"\nTotal classes mapped: {len(CLASS_TO_COARSE_GROUP)}")
    print(f"Classes in 'objects': {sum(1 for v in CLASS_TO_COARSE_GROUP.values() if v == 'objects')}")
    
    # Test examples
    print(f"\nTest examples:")
    print(f"chair (19) -> {get_coarse_semantic_label(19)}")
    print(f"armchair (31) -> {get_coarse_semantic_label(31)}")  
    print(f"Compatible: {are_semantically_compatible(19, 31)}")
    
    print(f"car (20) -> {get_coarse_semantic_label(20)}")
    print(f"chair (19) -> {get_coarse_semantic_label(19)}")
    print(f"Compatible: {are_semantically_compatible(20, 19)}")