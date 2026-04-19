"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

from .rag import (
    build_grounded_explanation,
    detect_cluster_warning,
    infer_profile_from_text,
    retrieve_docs,
)
from .recommender import load_songs, recommend_songs


def main() -> None:
    songs = load_songs("data/songs.csv")

    user_prompts = {
        "Workout Pop Session": "Give me upbeat pop for a workout. Keep it high energy.",
        "Late Night Focus": "I need chill lofi focus music for coding at night.",
        "Moody Drive": "I want moody songs for a night drive, not too acoustic.",
    }

    for prompt_name, user_text in user_prompts.items():
        user_prefs, profile_docs = infer_profile_from_text(user_text, songs)

        print(f"\n{'='*50}")
        print(f"Prompt: {prompt_name}")
        print(f"User text: {user_text}")
        print(f"Inferred profile: {user_prefs}")
        print(f"{'='*50}\n")

        recommendations = recommend_songs(user_prefs, songs, k=5)
        cluster_warning = detect_cluster_warning(recommendations)

        print("Top recommendations:\n")
        for song, score, reasons in recommendations:
            query = f"{song['genre']} {song['mood']} {user_text}"
            docs = retrieve_docs(query, profile_docs, k=2)
            explanation = build_grounded_explanation(song, score, reasons, docs)

            print(f"  {song['title']} - Score: {score:.2f}")
            print(f"  Because: {explanation}")
            print()

        if cluster_warning:
            print(cluster_warning)


if __name__ == "__main__":
    main()
