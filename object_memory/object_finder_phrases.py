def filter_caption(caption: list[str]) -> list[str]:
    filtered_caption = []
    for c in caption:
        if c.strip() in _words_to_ignore_in_caption:
            continue
        if _check_whether_in_sub_phrases(c.strip()):
            continue
        else:
            filtered_caption.append(c.strip())
    return filtered_caption

def _check_whether_in_sub_phrases(text: str) -> bool:
    for sub_phrase in _sub_phrases_to_ignore_in_caption:
        if sub_phrase in text:
            return True

    return False

def check_if_floor(texts: list[str]):
    words = [
        "floor",
        "ground",
        "earth"
    ]
    for word in words:
        if word in texts:
            return True
    return False

_words_to_ignore_in_caption = [
                    "living room", 
                    "ceiling", 
                    "room", 
                    "curtain", 
                    "den", 
                    "window", 
                    "floor", 
                    "wall", 
                    "red", 
                    "yellow", 
                    "white", 
                    "blue", 
                    "green", 
                    "brown",
                    "corridor",
                    "image",
                    "picture frame",
                    "mat",
                    "wood floor",
                    "shadow",
                    "hardwood",
                    "plywood",
                    "waiting room",
                    "lead to",
                    "belly",
                    "person",
                    "chest",
                    "black",
                    "accident",
                    "act",
                    "door",
                    "doorway",
                    "illustration",
                    "animal",
                    "mountain",
                    "table top", # since we don't want a flat object as an instance
                    "pen",
                    "pencil",
                    "corner",
                    "notepad",
                    "flower",
                    "man",
                    "pad",
                    "lead",
                    "ramp",
                    "plank",
                    "scale",
                    "beam",
                    "pink",
                    "tie",
                    "crack",
                    "mirror",
                    "square",
                    "rectangle",
                    "woman",
                    "tree",
                    "umbrella",
                    "hat",
                    "salon",
                    "beach",
                    "open",
                    "closet",
                    "blanket",
                    "circle",
                    "furniture",
                    "balustrade",
                    "cube",
                    "dress",
                    "ladder",
                    "briefcase",
                    "marble",
                    "pillar",
                    "dark",
                    "sea",
                    "cabinet",
                    "office"
]

_sub_phrases_to_ignore_in_caption = [
                    "room",
                    "floor",
                    "wall",
                    "frame",
                    "image",
                    "building",
                    "ceiling"
                    "lead",
                    "paint",
                    "shade",
                    "snow",
                    "rain",
                    "cloud",
                    "frost",
                    "fog",
                    "sky",
                    "carpet",
                    "view",
                    "scene",
                    "mat",
                    "window",
                    "vase",
                    "bureau",
                    "computer",
                    "cubicle",
                    "supply",
                    "sit",
                    "stall",
                    "fan",
                    "cabinet",
                    "job"
]