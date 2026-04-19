import csv
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))

    def _score_song(self, user: UserProfile, song: Song) -> Tuple[float, List[str]]:
        score = 0.0
        reasons: List[str] = []

        if song.genre == user.favorite_genre:
            score += 1.0
            reasons.append("it matches the genre you asked for (+1.0)")
        else:
            reasons.append("the genre is a little different (+0.0)")

        if song.mood == user.favorite_mood:
            score += 2.5
            reasons.append("the mood lines up with what you want (+2.5)")
        else:
            reasons.append("the mood is not an exact match (+0.0)")

        energy_similarity = self._clamp01(1.0 - abs(song.energy - user.target_energy))
        energy_points = 3.0 * energy_similarity
        score += energy_points
        reasons.append(f"it is close to your target energy ({energy_similarity:.2f}, +{energy_points:.2f})")

        acoustic_target = 0.8 if user.likes_acoustic else 0.2
        acoustic_similarity = self._clamp01(1.0 - abs(song.acousticness - acoustic_target))
        acoustic_points = 0.5 * acoustic_similarity
        score += acoustic_points
        reasons.append(f"the acoustic feel also fits ({acoustic_similarity:.2f}, +{acoustic_points:.2f})")

        return score, reasons

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        ranked = sorted(
            self.songs,
            key=lambda song: self._score_song(user, song)[0],
            reverse=True,
        )
        return ranked[:k]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        score, reasons = self._score_song(user, song)
        primary_reason = reasons[0] if reasons else "it is a reasonable match"
        extra_reason = f" Also, {reasons[1].rstrip('.')}." if len(reasons) > 1 else ""
        return f"This song scores {score:.2f} because {primary_reason}.{extra_reason}"

def load_songs(csv_path: str) -> List[Dict]:
    """
    Loads songs from a CSV file.
    Required by src/main.py
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
            # Normalize whitespace so CSV edits do not create accidental mismatches.
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
    Functional implementation of the recommendation logic.
    Required by src/main.py
    """
    # Backward-compatibility with profile style used by the OOP API.
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
