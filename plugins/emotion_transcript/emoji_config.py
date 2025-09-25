emoji_map = {
    "anger": ["😐", "😠", "😤", "😡", "🤬"],
    "disgust": ["😒"],
    "fear": ["😯", "😨", "😰", "😱"],
    "joy": ["🙂", "😊", "😄", "🤩"],
    "neutral": [""],
    "sarcasm": ["😏", "🤨", "😉"],
    "sadness": ["😔", "😞", "😭"],
    "surprise": ["😯", "😲", "😮"]
}

emoji_groups = {
    "🤬": [("anger", "disgust")],
    "😡": [("anger", "fear"), ("anger", "sadness")],
    "😤": [("anger", "joy"), ("sadness", "anger")],
    "😠": [("anger", "neutral"), ("anger", "surprise")],
    "😒": [("disgust", "sarcasm"),("neutral", "disgust"),("anger", "sarcasm"), ("disgust", "surprise"),("disgust", "fear"), ("disgust", "joy"), ("disgust", "neutral"), ("disgust", "anger"), ("disgust", "sadness")],
    "😨": [("neutral", "fear"),("fear", "anger"), ("fear", "neutral"), ("fear", "sadness")],
    "😰": [("fear", "disgust"), ("fear", "sarcasm")],
    "😅": [("fear", "joy"), ("joy", "disgust"), ("joy", "fear"), ("joy", "sadness")],
    "🙂": [("neutral", "joy")],
    "😊": [("joy", "neutral")],
    "😏": [("neutral", "sarcasm"), ("sarcasm", "sadness")],
    "😔": [("neutral", "sadness"), ("sadness", "joy"), ("sadness", "neutral")],
    "😓": [("sadness", "sarcasm")],
    "😞": [("sadness", "surprise")],
    "😅": [("sarcasm", "neutral"), ("sarcasm", "joy"), ("joy", "sarcasm")],
    "🤨": [("neutral", "anger"),("sarcasm", "anger"), ("sarcasm", "disgust")],
    "🤩": [("joy", "surprise"), ("surprise", "joy")],
    "😱": [("fear", "surprise"), ("surprise", "fear"), ("sarcasm", "fear")],
    "😮": [("neutral", "surprise"), ("surprise", "sarcasm"), ("surprise", "anger"), ("surprise", "disgust"), ("surprise", "neutral"), ("surprise", "sadness")]
}
