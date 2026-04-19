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


def _normalize_for_matching(text: str) -> str:
    normalized = text.lower().replace("'", "")
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


_TOKEN_SYNONYMS = {
    "gym": "workout",
    "run": "workout",
    "running": "workout",
    "exercise": "workout",
    "rapping": "hip hop",
    "rapper": "hip hop",
    "rappers": "hip hop",
    "rap": "hip hop",
    "upbeat": "energetic",
    "hype": "energetic",
    "bouncy": "energetic",
    "pumped": "energetic",
    "study": "focus",
    "coding": "focus",
    "code": "focus",
    "flow": "confident",
    "drive": "driving",
    "roadtrip": "driving",
    "calm": "chill",
    "serene": "chill",
    "chillout": "chill",
    "moody": "reflective",
    "sad": "reflective",
    "wistful": "reflective",
    "hiphop": "hip hop",
    "hip-hop": "hip hop",
    "rap": "hip hop",
    "r&b": "r&b",
    "rnb": "r&b",
    "reggaeton": "reggaeton",
    "spanish": "reggaeton",
    "latin": "reggaeton",
    "club": "party",
    "turnup": "party",
    "turn-up": "party",
    "dancefloor": "party",
    "feelgood": "feel-good",
    "feel-good": "feel-good",
    "hyperpop": "edm",
}


_GENRE_ALIASES = {
    "edm": ["party", "club", "hyperpop", "hyper pop", "rave", "electronic", "electronic music", "dance music", 
            "club music", "house", "techno", "trance", "dubstep", "future bass", "festival edm"],
    "pop": ["pop music", "mainstream pop", "chart pop", "top 100", "radio", "popular", "catchy"],
    "indie pop": ["independent", "indie", "indiepop", "alt pop", "alternative pop", 
                  "bedroom pop", "dream pop", "bedroom", "twee"],
    "hip hop": ["hip-hop", "hiphop", "rap", "rap music", "rap songs", "rapping", "rapper", 
                "rappers", "trap", "drill", "boom bap"],
    "r&b": ["rnb", "rhythm and blues", "neo soul", "neosoul", "soul", "contemporary r&b"],
    "reggaeton": ["latin reggaeton", "latin trap", "dembow", "urbano"],
    "lofi": ["lo fi", "lo-fi", "chillhop", "study beats", "beats to study", "study music"],
    "ambient": ["ambient music", "soundscape", "drone", "meditation music", "background ambience"],
    "rock": ["rock music", "alternative rock", "alt rock", "indie rock", "hard rock", "classic rock", "garage rock", "punk rock"],
    "metal": ["heavy metal", "death metal", "black metal", "thrash metal", "metalcore"],
    "jazz": ["jazz music", "smooth jazz", "bebop", "swing", "blues jazz"],
    "folk": ["folk music", "indie folk", "americana", "singer songwriter", "bluegrass"],
    "country": ["yeehaw", "country music", "country pop", "americana country", "outlaw country"],
    "classical": ["classical music", "orchestral", "symphonic", "piano classical", "chamber music", "instrumental classical"],
    "reggae": ["reggae music", "roots reggae", "dub", "ska", "dancehall"],
    "synthwave": ["synth", "retrowave", "outrun", "vaporwave", "neon synth"],
    "psychedelic rock": ["weird", "trippy", "psychedelic", "psych rock", "space rock", "prog rock"],
}


_GENRE_ALIAS_LOOKUP = {
    _normalize_for_matching(alias): canonical
    for canonical, aliases in _GENRE_ALIASES.items()
    for alias in aliases + [canonical]
}


_GENRE_MATCH_ORDER = sorted(_GENRE_ALIAS_LOOKUP, key=len, reverse=True)


_GENRE_PRESETS = {
    "edm": {
        "mood": "euphoric",
        "energy": 0.92,
        "danceability": 0.90,
        "valence": 0.78,
        "acousticness": 0.06,
        "tempo_bpm": 128.0,
    },
    "pop": {
        "mood": "happy",
        "energy": 0.82,
        "danceability": 0.82,
        "valence": 0.80,
        "acousticness": 0.18,
        "tempo_bpm": 116.0,
    },
    "indie pop": {
        "mood": "dreamy",
        "energy": 0.68,
        "danceability": 0.74,
        "valence": 0.68,
        "acousticness": 0.28,
        "tempo_bpm": 112.0,
    },
    "hip hop": {
        "mood": "confident",
        "energy": 0.80,
        "danceability": 0.86,
        "valence": 0.62,
        "acousticness": 0.12,
        "tempo_bpm": 96.0,
    },
    "r&b": {
        "mood": "romantic",
        "energy": 0.58,
        "danceability": 0.74,
        "valence": 0.70,
        "acousticness": 0.42,
        "tempo_bpm": 94.0,
    },
    "reggaeton": {
        "mood": "party",
        "energy": 0.86,
        "danceability": 0.88,
        "valence": 0.76,
        "acousticness": 0.08,
        "tempo_bpm": 128.0,
    },
    "lofi": {
        "mood": "chill",
        "energy": 0.34,
        "danceability": 0.58,
        "valence": 0.58,
        "acousticness": 0.84,
        "tempo_bpm": 80.0,
    },
    "ambient": {
        "mood": "calm",
        "energy": 0.20,
        "danceability": 0.30,
        "valence": 0.46,
        "acousticness": 0.95,
        "tempo_bpm": 60.0,
    },
    "rock": {
        "mood": "intense",
        "energy": 0.88,
        "danceability": 0.60,
        "valence": 0.48,
        "acousticness": 0.12,
        "tempo_bpm": 144.0,
    },
    "metal": {
        "mood": "aggressive",
        "energy": 0.96,
        "danceability": 0.44,
        "valence": 0.34,
        "acousticness": 0.04,
        "tempo_bpm": 160.0,
    },
    "jazz": {
        "mood": "relaxed",
        "energy": 0.40,
        "danceability": 0.56,
        "valence": 0.66,
        "acousticness": 0.86,
        "tempo_bpm": 92.0,
    },
    "folk": {
        "mood": "nostalgic",
        "energy": 0.44,
        "danceability": 0.54,
        "valence": 0.62,
        "acousticness": 0.84,
        "tempo_bpm": 88.0,
    },
    "country": {
        "mood": "reflective",
        "energy": 0.52,
        "danceability": 0.64,
        "valence": 0.60,
        "acousticness": 0.68,
        "tempo_bpm": 104.0,
    },
    "classical": {
        "mood": "serene",
        "energy": 0.18,
        "danceability": 0.24,
        "valence": 0.58,
        "acousticness": 0.96,
        "tempo_bpm": 66.0,
    },
    "reggae": {
        "mood": "carefree",
        "energy": 0.58,
        "danceability": 0.72,
        "valence": 0.76,
        "acousticness": 0.52,
        "tempo_bpm": 92.0,
    },
    "synthwave": {
        "mood": "nostalgic",
        "energy": 0.74,
        "danceability": 0.74,
        "valence": 0.54,
        "acousticness": 0.18,
        "tempo_bpm": 112.0,
    },
    "psychedelic rock": {
        "mood": "groovy",
        "energy": 0.82,
        "danceability": 0.70,
        "valence": 0.66,
        "acousticness": 0.24,
        "tempo_bpm": 120.0,
    },
}


def _normalize_tokens(tokens: Sequence[str]) -> List[str]:
    return [_TOKEN_SYNONYMS.get(token, token) for token in tokens]


def _canonical_genre_from_text(text: str) -> Optional[str]:
    normalized = _normalize_for_matching(text)
    for alias in _GENRE_MATCH_ORDER:
        if alias in normalized:
            return _GENRE_ALIAS_LOOKUP[alias]
    return None


def _build_exploration_profile(songs: Sequence[Dict]) -> Dict[str, float]:
    numeric_fields = ["energy", "danceability", "valence", "acousticness", "tempo_bpm"]
    profile: Dict[str, float] = {}

    for field in numeric_fields:
        values = [float(song[field]) for song in songs if field in song and song[field] != ""]
        if values:
            profile[field] = sum(values) / len(values)

    profile.setdefault("energy", 0.55)
    profile.setdefault("danceability", 0.55)
    profile.setdefault("valence", 0.55)
    profile.setdefault("acousticness", 0.45)
    profile.setdefault("tempo_bpm", 100.0)
    profile["mode"] = "discovery"
    return profile


def _term_frequency(tokens: Sequence[str]) -> Dict[str, float]:
    if not tokens:
        return {}

    counts: Dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1

    token_count = float(len(tokens))
    return {token: count / token_count for token, count in counts.items()}


def _cosine_similarity(left: Dict[str, float], right: Dict[str, float]) -> float:
    if not left or not right:
        return 0.0

    dot = sum(value * right.get(token, 0.0) for token, value in left.items())
    left_norm = sum(value * value for value in left.values()) ** 0.5
    right_norm = sum(value * value for value in right.values()) ** 0.5
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0

    return dot / (left_norm * right_norm)


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
    query_raw_tokens = _tokenize(query)
    query_tokens = set(query_raw_tokens)
    query_norm_tokens = _normalize_tokens(query_raw_tokens)
    query_tf = _term_frequency(query_norm_tokens)
    required = list(required_tag_prefixes or [])

    scored: List[Tuple[float, str, RagDocument]] = []
    for doc in docs:
        raw_doc_tokens = _tokenize(doc.title + " " + doc.text + " " + " ".join(doc.tags))
        doc_tokens = set(raw_doc_tokens)
        doc_norm_tokens = _normalize_tokens(raw_doc_tokens)
        doc_tf = _term_frequency(doc_norm_tokens)

        keyword_overlap = len(query_tokens.intersection(doc_tokens))
        keyword_score = keyword_overlap / max(1, len(query_tokens))

        semantic_score = _cosine_similarity(query_tf, doc_tf)

        tag_text = " ".join(doc.tags).lower()
        tag_hits = sum(1 for token in set(query_norm_tokens) if token in tag_text)
        tag_score = min(1.0, tag_hits / 3.0)

        prefix_bonus = 0.0
        for prefix in required:
            if any(tag.startswith(prefix) for tag in doc.tags):
                prefix_bonus += 0.2

        score = (0.55 * semantic_score) + (0.35 * keyword_score) + (0.10 * tag_score) + prefix_bonus
        scored.append((score, doc.id, doc))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [doc for _, _, doc in scored[:k]]


def infer_profile_from_text(user_text: str, songs: Sequence[Dict]) -> Tuple[Dict, List[RagDocument]]:
    docs = load_corpus()
    retrieved = retrieve_docs(user_text, docs, k=3)

    text = _normalize_for_matching(user_text)
    prefs: Dict[str, float | str] = {}
    mood_keywords = [
        "feel-good",
        "heartbroken",
        "melancholic",
        "melancholy",
        "introspective",
        "haunting",
        "dreamy",
        "quirky",
        "dramatic",
        "calm",
        "epic",
        "hypnotic",
        "emotional",
        "playful",
        "groovy",
        "summer",
        "awkward",
        "party",
        "soulful",
        "dark",
        "heroic",
        "energetic",
        "empowered",
        "sensual",
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
        "bouncy",
        "uplifting",
    ]

    detected_genre = _canonical_genre_from_text(text)
    if detected_genre:
        prefs["genre"] = detected_genre

    genre_preset = _GENRE_PRESETS.get(str(prefs.get("genre", "")))
    if genre_preset:
        for key, value in genre_preset.items():
            prefs.setdefault(key, value)

    for mood in mood_keywords:
        if mood in text:
            prefs["mood"] = mood
            break

    if any(word in text for word in ["workout", "gym", "run", "hype", "upbeat"]):
        prefs.setdefault("energy", 0.85)
        prefs.setdefault("danceability", 0.82)
    elif any(word in text for word in ["party", "club", "dancefloor", "reggaeton"]):
        prefs.setdefault("energy", 0.82)
        prefs.setdefault("danceability", 0.86)
        prefs.setdefault("valence", 0.76)
    elif any(word in text for word in ["rap", "rapping", "hip hop", "hiphop", "hip-hop", "trap", "drill"]):
        prefs.setdefault("energy", 0.78)
        prefs.setdefault("danceability", 0.84)
        prefs.setdefault("valence", 0.62)
    elif any(word in text for word in ["focus", "study", "coding"]):
        prefs.setdefault("energy", 0.45)
        prefs.setdefault("tempo_bpm", 85.0)
    elif any(word in text for word in ["sleep", "calm", "wind down", "recover"]):
        prefs.setdefault("energy", 0.25)
        prefs.setdefault("acousticness", 0.82)

    if any(word in text for word in ["acoustic", "unplugged", "warm", "intimate"]):
        prefs["acousticness"] = 0.85
    elif "electronic" in text:
        prefs["acousticness"] = 0.20

    if prefs.get("genre") in {"edm", "reggaeton"}:
        prefs.setdefault("danceability", 0.90 if prefs["genre"] == "edm" else 0.88)
        prefs.setdefault("tempo_bpm", 128.0)
        prefs.setdefault("energy", 0.90 if prefs["genre"] == "edm" else 0.86)
        prefs.setdefault("valence", 0.78 if prefs["genre"] == "edm" else 0.76)
        prefs.setdefault("acousticness", 0.06 if prefs["genre"] == "edm" else 0.08)

    if any(word in text for word in ["happy", "euphoric", "upbeat", "party", "summer", "feel-good"]):
        prefs.setdefault("valence", 0.80)
    elif any(word in text for word in ["moody", "reflective", "sad", "heartbroken", "melancholic", "melancholy"]):
        prefs.setdefault("valence", 0.42)

    if "tempo" not in prefs and any(word in text for word in ["fast", "high bpm", "quick"]):
        prefs["tempo_bpm"] = 130.0
    elif "tempo" not in prefs and any(word in text for word in ["slow", "low bpm"]):
        prefs["tempo_bpm"] = 80.0

    discovery_requested = discovery_mode_requested(user_text)
    if discovery_requested:
        if "genre" not in prefs:
            prefs.update(_build_exploration_profile(songs))
        prefs["mode"] = "discovery"
    elif not prefs:
        prefs["needs_clarification"] = True

    return prefs, retrieved


def discovery_mode_requested(user_text: str) -> bool:
    text = _normalize_for_matching(user_text)
    return any(
        phrase in text
        for phrase in [
            "discovery mode",
            "surprise me",
            "explore",
            "something different",
            "something new",
            "take me somewhere new",
            "wildcard",
            "adventurous pick",
            "i dont know",
            "dont know",
            "do not know",
            "doesnt matter",
            "anything works",
            "open to anything"
        ]
    )


def retrieve_song_grounding_docs(
    user_text: str,
    song: Dict,
    docs: Sequence[RagDocument],
    k: int = 2,
) -> List[RagDocument]:
    query = f"{song.get('genre', '')} {song.get('mood', '')} {song.get('title', '')} {user_text}"
    required_tag_prefixes = []
    if song.get("genre"):
        required_tag_prefixes.append(f"genre:{song['genre']}")
    if song.get("mood"):
        required_tag_prefixes.append(f"mood:{song['mood']}")

    return retrieve_docs(query, docs, k=k, required_tag_prefixes=required_tag_prefixes)


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
    if reasons:
        main_reason = reasons[0].rstrip(".")
        base = f"This is a strong fit because {main_reason}."
        if len(reasons) > 1:
            base += f" Also, {reasons[1].rstrip('.')}."
    else:
        base = "This is a strong fit."

    if not retrieved_docs:
        return base + f" (Score {score:.2f}.)"

    titles = ", ".join(doc.title for doc in retrieved_docs[:2])
    return base + f" Grounded by: {titles}."
