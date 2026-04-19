from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

from .rag import build_grounded_explanation, infer_profile_from_text, retrieve_docs
from .recommender import load_songs, recommend_songs


PromptResult = Dict[str, object]


PROMPT_SET: Dict[str, str] = {
    "Workout Pop Session": "Give me upbeat pop for a workout. Keep it high energy.",
    "Late Night Focus": "I need chill lofi focus music for coding at night.",
    "Moody Drive": "I want moody songs for a night drive, not too acoustic.",
}


def top_titles(recommendations: Sequence[Tuple[Dict, float, List[str]]]) -> List[str]:
    return [str(song.get("title", "")) for song, _, _ in recommendations]


def top_genres(recommendations: Sequence[Tuple[Dict, float, List[str]]]) -> List[str]:
    return [str(song.get("genre", "")) for song, _, _ in recommendations]


def ranking_signature(recommendations: Sequence[Tuple[Dict, float, List[str]]]) -> List[Tuple[str, str]]:
    return [(str(song.get("title", "")), str(song.get("genre", ""))) for song, _, _ in recommendations]


def evaluate_baseline_stability(songs: Sequence[Dict], prompts: Dict[str, str] | None = None, k: int = 5) -> Dict[str, List[Tuple[str, str]]]:
    prompt_map = prompts or PROMPT_SET
    signatures: Dict[str, List[Tuple[str, str]]] = {}

    for name, user_text in prompt_map.items():
        user_prefs, _ = infer_profile_from_text(user_text, songs)
        recs = recommend_songs(user_prefs, list(songs), k=k)
        signatures[name] = ranking_signature(recs)

    return signatures


def evaluate_rag_explanations(songs: Sequence[Dict], prompts: Dict[str, str] | None = None, k: int = 5) -> Dict[str, List[str]]:
    prompt_map = prompts or PROMPT_SET
    explanations: Dict[str, List[str]] = {}

    for name, user_text in prompt_map.items():
        user_prefs, profile_docs = infer_profile_from_text(user_text, songs)
        recs = recommend_songs(user_prefs, list(songs), k=k)
        prompt_explanations: List[str] = []

        for song, score, reasons in recs:
            query = f"{song['genre']} {song['mood']} {user_text}"
            docs = retrieve_docs(query, profile_docs, k=2)
            prompt_explanations.append(build_grounded_explanation(song, score, reasons, docs))

        explanations[name] = prompt_explanations

    return explanations


def compare_baseline_vs_rag(songs: Sequence[Dict], prompts: Dict[str, str] | None = None, k: int = 5) -> Dict[str, PromptResult]:
    prompt_map = prompts or PROMPT_SET
    report: Dict[str, PromptResult] = {}

    for name, user_text in prompt_map.items():
        user_prefs, profile_docs = infer_profile_from_text(user_text, songs)
        baseline_recs = recommend_songs(user_prefs, list(songs), k=k)
        baseline_titles = top_titles(baseline_recs)

        rag_explanations: List[str] = []
        for song, score, reasons in baseline_recs:
            query = f"{song['genre']} {song['mood']} {user_text}"
            docs = retrieve_docs(query, profile_docs, k=2)
            rag_explanations.append(build_grounded_explanation(song, score, reasons, docs))

        report[name] = {
            "user_text": user_text,
            "profile": user_prefs,
            "baseline_titles": baseline_titles,
            "rag_explanations": rag_explanations,
        }

    return report
