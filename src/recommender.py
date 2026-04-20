import csv
from typing import List, Dict, Tuple, Optional


def load_songs(csv_path: str) -> List[Dict]:
    """
    Loads songs from a CSV file and normalizes numeric fields.
    """
    songs: List[Dict] = []
    numeric_converters = {
        "id": int,
        "energy": float,
        "tempo_bpm": float,
        "valence": float,
        "danceability": float,
        "acousticness": float,
    }

    with open(csv_path, mode="r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            song = {
                key: (value.strip() if isinstance(value, str) else value)
                for key, value in dict(row).items()
            }
            for field, converter in numeric_converters.items():
                if field in song and song[field] != "":
                    song[field] = converter(song[field])
            songs.append(song)

    return songs


def _clamp01(value: float) -> float:
    """Clamp a value to the range [0.0, 1.0]."""
    return max(0.0, min(1.0, value))


def _score_song_dict(
    user_prefs: Dict,
    song: Dict,
    min_tempo: Optional[float],
    max_tempo: Optional[float],
) -> Tuple[float, List[str]]:
    """Score a song dict based on user preferences and return score with explanation reasons."""
    weights = {
        "genre": 1.0,
        "mood": 2.5,
        "energy": 3.0,
        "tempo_bpm": 1.5,
        "valence": 1.0,
        "danceability": 1.0,
        "acousticness": 0.5,
    }

    score = 0.0
    reasons: List[str] = []

    preferred_genre = user_prefs.get("genre")
    if preferred_genre is not None:
        if song.get("genre") == preferred_genre:
            score += weights["genre"]
            reasons.append(f"it matches the genre you asked for (+{weights['genre']:.1f})")
        else:
            reasons.append("the genre is a little different (+0.0)")

    preferred_mood = user_prefs.get("mood")
    if preferred_mood is not None:
        if song.get("mood") == preferred_mood:
            score += weights["mood"]
            reasons.append(f"the mood lines up with what you want (+{weights['mood']:.1f})")
        else:
            reasons.append("the mood is not an exact match (+0.0)")

    for feature in ["energy", "valence", "danceability", "acousticness"]:
        if feature in user_prefs and feature in song:
            similarity = _clamp01(1.0 - abs(float(song[feature]) - float(user_prefs[feature])))
            points = similarity * weights[feature]
            score += points
            if feature == "valence":
                reasons.append(
                    "the valence is close to what you seem to want to hear right now "
                    f"({similarity:.2f}, +{points:.2f})"
                )
            else:
                readable_feature = feature.replace("_", " ")
                reasons.append(f"its {readable_feature} is close to your target ({similarity:.2f}, +{points:.2f})")

    if "tempo_bpm" in user_prefs and "tempo_bpm" in song:
        if min_tempo is not None and max_tempo is not None and max_tempo > min_tempo:
            tempo_similarity = 1.0 - (
                abs(float(song["tempo_bpm"]) - float(user_prefs["tempo_bpm"]))
                / (max_tempo - min_tempo)
            )
            tempo_similarity = _clamp01(tempo_similarity)
        else:
            tempo_similarity = _clamp01(1.0 - abs(float(song["tempo_bpm"]) - float(user_prefs["tempo_bpm"])) / 200.0)
        tempo_points = tempo_similarity * weights["tempo_bpm"]
        score += tempo_points
        reasons.append(f"the tempo fits your request ({tempo_similarity:.2f}, +{tempo_points:.2f})")

    return score, reasons


def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, List[str]]]:
    """
    Scores songs based on user preferences and returns top-k recommendations.
    Explanations are grounded in the RAG corpus via build_grounded_explanation().
    """
    if "acousticness" not in user_prefs and "likes_acoustic" in user_prefs:
        user_prefs = dict(user_prefs)
        user_prefs["acousticness"] = 0.8 if bool(user_prefs["likes_acoustic"]) else 0.2

    tempos = [float(song["tempo_bpm"]) for song in songs if "tempo_bpm" in song and song["tempo_bpm"] != ""]
    min_tempo = min(tempos) if tempos else None
    max_tempo = max(tempos) if tempos else None

    scored = [
        (song, *_score_song_dict(user_prefs, song, min_tempo, max_tempo))
        for song in songs
    ]
    top_k = sorted(scored, key=lambda item: item[1], reverse=True)[:k]

    return top_k
