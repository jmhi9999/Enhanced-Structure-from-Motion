"""
Coarse semantic grouping for ADE20K 150 classes
Groups fine-grained classes into broader categories for robust semantic matching
"""

# ADE20K 150 classes grouped into coarse semantic categories
# This reduces overly strict filtering while maintaining semantic consistency

COARSE_SEMANTIC_GROUPS = {
    # Architectural structures - permanent building elements
    'structure': [
        0,   # wall  
        1,   # building
        3,   # floor
        5,   # ceiling
        6,   # road
        11,  # sidewalk
        13,  # earth
        14,  # door
        25,  # house
        49,  # skyscraper
        52,  # path
        53,  # stairs
        55,  # runway
        58,  # stairway
        63,  # bridge
        84,  # hovel
        89,  # bannister
        104, # awning
        111, # booth
        139, # step
        140, # pier
    ],
    
    # Furniture and interior objects
    'furniture': [
        7,   # bed
        10,  # cabinet
        15,  # table
        19,  # chair
        23,  # sofa
        24,  # shelf
        31,  # armchair
        32,  # seat
        35,  # desk
        36,  # wardrobe
        37,  # lamp
        38,  # bathtub
        43,  # chest of drawers
        48,  # refrigerator
        50,  # fireplace
        56,  # case
        57,  # pool table
        64,  # bookcase
        66,  # coffee table
        67,  # toilet
        70,  # countertop
        71,  # kitchen island
        73,  # swivel chair
        75,  # bar
        80,  # ottoman
        85,  # crib
        93,  # bookshelf
        120, # stool
        121, # wardrobe
        146, # radiator
    ],
    
    # Vehicles and transportation
    'vehicle': [
        20,  # car
        74,  # boat
        83,  # bus
        86,  # truck
        103, # airplane
        109, # van
        133, # streetlight (transportation infrastructure)
    ],
    
    # Natural elements - landscape and vegetation
    'nature': [
        4,   # tree
        9,   # grass
        16,  # mountain
        17,  # plant
        21,  # water
        26,  # sea
        29,  # field
        33,  # fence
        34,  # rock
        46,  # sand
        59,  # river
        68,  # flower
        69,  # hill
        72,  # palm
        94,  # fruit
        130, # pole
    ],
    
    # Sky and atmospheric elements  
    'sky': [
        2,   # sky
    ],
    
    # Living beings
    'living': [
        12,  # person
        91,  # animal
        107, # dog
        127, # horse
        132, # cow
    ],
    
    # Textiles and soft materials
    'textile': [
        18,  # curtain
        28,  # rug
        40,  # cushion
        58,  # pillow
        114, # blanket
        118, # towel
        124, # napkin
    ],
    
    # Windows and transparent surfaces
    'transparent': [
        8,   # windowpane
        27,  # mirror
        47,  # sink
        96,  # window
        147, # glass
    ],
    
    # Electronic and technical objects
    'electronics': [
        72,  # computer
        76,  # arcade machine
        141, # crt screen
        143, # monitor
        148, # clock
    ],
    
    # Decorative and art objects
    'decoration': [
        22,  # painting
        78,  # apparel
        88,  # chandelier
        98,  # sconce
        99,  # vase
        119, # sculpture
        122, # sculpture  
        142, # plate
        149, # flag
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