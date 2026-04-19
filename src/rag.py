import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .recommender import _score_song_dict


@dataclass
class RagDocument:
    id: str
    title: str
    text: str
    tags: List[str]


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def load_corpus(path: str = "data/rag_corpus.json") -> List[RagDocument]:
    corpus_path = Path(path)
    with corpus_path.open("r", encoding="utf-8") as file:
        raw_docs = json.load(file)

    docs: List[RagDocument] = []
    for item in raw_docs:
        docs.append(
            RagDocument(
                id=str(item["id"]),
                title=str(item["title"]),
                text=str(item["text"]),
                tags=[str(tag) for tag in item.get("tags", [])],
            )
        )
    return docs


def retrieve_docs(
    query: str,
    docs: Sequence[RagDocument],
    k: int = 3,
    required_tag_prefixes: Optional[Iterable[str]] = None,
) -> List[RagDocument]:
    query_tokens = set(_tokenize(query))
    required = list(required_tag_prefixes or [])

    scored: List[Tuple[float, str, RagDocument]] = []
    for doc in docs:
        doc_tokens = set(_tokenize(doc.title + " " + doc.text + " " + " ".join(doc.tags)))
        overlap = len(query_tokens.intersection(doc_tokens))

        prefix_bonus = 0.0
        for prefix in required:
            if any(tag.startswith(prefix) for tag in doc.tags):
                prefix_bonus += 0.5

        score = float(overlap) + prefix_bonus
        scored.append((score, doc.id, doc))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [doc for _, _, doc in scored[:k]]


def infer_profile_from_text(user_text: str, songs: Sequence[Dict]) -> Tuple[Dict, List[RagDocument]]:
    docs = load_corpus()
    retrieved = retrieve_docs(user_text, docs, k=3)

    text = user_text.lower()
    prefs: Dict[str, float | str] = {}

    genre_keywords = [
        "pop",
        "lofi",
        "rock",
        "ambient",
        "jazz",
        "synthwave",
        "metal",
        "reggae",
        "classical",
        "hip hop",
        "country",
        "edm",
        "folk",
        "r&b",
        "indie pop",
    ]
    mood_keywords = [
        "happy",
        "chill",
        "intense",
        "relaxed",
        "moody",
        "focused",
        "romantic",
        "nostalgic",
        "euphoric",
        "reflective",
        "confident",
        "aggressive",
        "carefree",
        "serene",
    ]

    for genre in sorted(genre_keywords, key=len, reverse=True):
        if genre in text:
            prefs["genre"] = genre
            break

    for mood in mood_keywords:
        if mood in text:
            prefs["mood"] = mood
            break

    if any(word in text for word in ["workout", "gym", "run", "hype", "upbeat"]):
        prefs.setdefault("energy", 0.85)
        prefs.setdefault("danceability", 0.82)
    elif any(word in text for word in ["focus", "study", "coding"]):
        prefs.setdefault("energy", 0.45)
        prefs.setdefault("tempo_bpm", 85.0)
    elif any(word in text for word in ["sleep", "calm", "wind down", "recover"]):
        prefs.setdefault("energy", 0.25)
        prefs.setdefault("acousticness", 0.82)
    else:
        prefs.setdefault("energy", 0.60)

    if any(word in text for word in ["acoustic", "unplugged", "warm", "intimate"]):
        prefs["acousticness"] = 0.85
    elif "electronic" in text:
        prefs["acousticness"] = 0.20

    if any(word in text for word in ["happy", "euphoric", "upbeat"]):
        prefs.setdefault("valence", 0.80)
    elif any(word in text for word in ["moody", "reflective", "sad"]):
        prefs.setdefault("valence", 0.42)

    if "tempo" not in prefs and any(word in text for word in ["fast", "high bpm", "quick"]):
        prefs["tempo_bpm"] = 130.0
    elif "tempo" not in prefs and any(word in text for word in ["slow", "low bpm"]):
        prefs["tempo_bpm"] = 80.0

    # Fallbacks grounded in available catalog values.
    if "genre" not in prefs:
        prefs["genre"] = str(songs[0]["genre"]) if songs else "pop"
    if "mood" not in prefs:
        prefs["mood"] = str(songs[0]["mood"]) if songs else "happy"

    return prefs, retrieved


def detect_cluster_warning(recommendations: Sequence[Tuple[Dict, float, List[str]]]) -> Optional[str]:
    if not recommendations:
        return None

    genres = [str(song.get("genre", "")) for song, _, _ in recommendations]
    top_genre = max(set(genres), key=genres.count)
    ratio = genres.count(top_genre) / len(genres)
    energies = [float(song.get("energy", 0.0)) for song, _, _ in recommendations if song.get("energy") is not None]
    energy_range = max(energies) - min(energies) if energies else 1.0

    if ratio >= 0.8 or energy_range <= 0.20:
        if ratio >= 0.8:
            reason = f"clustered around '{top_genre}'"
        else:
            reason = f"clustered in a narrow energy band ({energy_range:.2f})"
        return (
            f"Transparency: recommendations are {reason}. "
            "Consider enabling discovery mode for more diversity."
        )
    return None


def select_bridge_recommendation(
    user_prefs: Dict,
    songs: Sequence[Dict],
    recommendations: Sequence[Tuple[Dict, float, List[str]]],
) -> Optional[Tuple[Dict, float, List[str]]]:
    if not recommendations:
        return None

    top_genres = [str(song.get("genre", "")) for song, _, _ in recommendations]
    dominant_genre = max(set(top_genres), key=top_genres.count)
    recommended_ids = {song.get("id") for song, _, _ in recommendations}

    tempos = [float(song["tempo_bpm"]) for song in songs if "tempo_bpm" in song and song["tempo_bpm"] != ""]
    min_tempo = min(tempos) if tempos else None
    max_tempo = max(tempos) if tempos else None

    bridge_candidates: List[Tuple[float, Dict, List[str]]] = []
    for song in songs:
        if song.get("id") in recommended_ids:
            continue

        score, reasons = _score_song_dict(user_prefs, song, min_tempo, max_tempo)
        if song.get("genre") == dominant_genre:
            score -= 0.75
            reasons = list(reasons) + ["cluster penalty (-0.75)"]
        else:
            reasons = list(reasons) + [f"bridge from {dominant_genre} to {song.get('genre', 'unknown')}"]

        bridge_candidates.append((score, song, reasons))

    if not bridge_candidates:
        return None

    bridge_candidates.sort(key=lambda item: (item[0], str(item[1].get("id", ""))), reverse=True)
    best_score, best_song, best_reasons = bridge_candidates[0]
    if best_score <= 0:
        return None

    return best_song, best_score, best_reasons


def build_grounded_explanation(
    song: Dict,
    score: float,
    reasons: Sequence[str],
    retrieved_docs: Sequence[RagDocument],
) -> str:
    base = f"Score {score:.2f}. " + "; ".join(reasons[:3])
    if not retrieved_docs:
        return base

    titles = ", ".join(doc.title for doc in retrieved_docs[:2])
    return base + f" | Grounded by: {titles}."
