# 🎵 Song Recommender

## Original Project Summary

The original project was a music recommender simulation that ranks songs based on a user's taste profile. It scored songs using genre, mood, energy, tempo, valence, danceability, and acousticness. It used weighted similarity, then returned the top recommendations with score explanations so the ranking was easy to understand. There were tests for multiple user profiles, including conflicting and edge-case preferences, to check where the recommender worked well and where biases appeared.

---

## Song Recommender Ver. 1.0 Summary:
This A.I. system recommends songs from a local catalog using a weighted scoring model, then explains each pick with short evidence retrieved from a small music-context corpus JSON. It matters because users have the flexibility and freedom to just ask for any major mood or genre they want and it will fins a song for you quickly.

## Architecture Overview:
The system uses a hybrid pipeline that combines deterministic ranking with retrieval-grounded explanations:

1. User request enters through `main.py` (orchestrator), which calls `infer_profile_from_text` to convert natural language into structured preferences (genre/mood/energy intent).
2. The recommender reads `songs.csv` and runs `recommend_songs`, a weighted similarity ranker that produces Top-K candidates.
3. In parallel, the RAG path queries `rag_corpus.json` through `retrieve_song_grounding_docs` to fetch short, relevant context snippets for the user intent and candidate songs.
4. `build_grounded_explanation` merges ranking evidence + retrieved context so each recommendation includes a reason tied to both song features and corpus support.
5. Final recommendations are returned as ranked songs with grounded explanations.

The quality-control loop in the diagram is handled by `evaluation.py` and the pytest suite (`test_recommender.py`, `test_rag.py`, `test_evaluation.py`). Evaluation prompts and automated tests feed reviewer/developer adjustments to weights, retrieval tags, and parser heuristics, which are then re-run through the same pipeline.

## Setup Instructions:
1. Create and activate a virtual environment.
2. Install dependencies:
	`pip install -r requirements.txt`
3. Run the app:
	`python -m src.main`
4. Run tests:
	`pytest -q`

## Sample Interactions:
1. Input: "Give me upbeat pop for a workout."
	Output: Top picks are high-energy pop tracks with explanations that mention mood/energy fit and retrieved workout/pop context.

2. Input: "I need chill lofi focus music for coding at night."
	Output: Recommendations shift to lower-tempo, more acoustic/focus-friendly songs, with grounding from focus/lofi corpus docs.

3. Input: "Surprise me, discovery mode."
	Output: System returns normal top picks plus a bridge recommendation from a different genre to reduce cluster bias.

## Design Decisions:
I built it this way to be fast, reliable and maintainable. I kept the scoring logic from the original project but wanted users to enter naturally what theyw anted as their entries and have my system use RAG to get the best song for them. This meant getting rid of most of the OOP programming that was there. The retrieval works but might still not be as flexible as a full LLM. Tl;dr:
- Used deterministic weighted scoring for transparency and stable tests.
- Included alias/synonym mapping so natural language requests (e.g., "lo-fi", "rapping", "hyperpop") map to supported profiles.
- Trade-off: This design is explainable and fast, but less flexible than a fully learned recommendation model.

## Reliability and Evaluation:

To verify this system works reliably, I evaluated it with multiple checks:

- Automated tests: I used pytest suites for ranking behavior, profile inference, retrieval quality, explanation grounding, and baseline determinism.
- Determinism check: The same prompt set is run repeatedly to confirm ranking signatures remain stable across runs.
- Confidence proxy: Since this is a deterministic scorer, confidence is approximated by score separation (how far the top recommendation is from the next options) plus retrieval overlap quality in grounding docs.
- Error handling / guardrails: Ambiguous requests are flagged with a clarification path (`needs_clarification`) instead of forcing low-quality recommendations.
- Human evaluation: I manually reviewed recommendation + explanation pairs for sample prompts (workout pop, late-night focus, moody drive) to ensure the text rationale matched song features and retrieved context.

Testing snapshot:
- 20 out of 20 automated tests passed.
- Ranking outputs were deterministic across repeated runs for the same prompts.
- Reliability was strongest on supported genres/moods; performance dropped most on ambiguous or under-covered phrasing (for example, vague intent without clear genre/mood cues).


## Testing Summary:
- Tests validate ranking structure, deterministic evaluation outputs, profile inference from text, retrieval relevance, grounded explanations, and discovery-mode bridge behavior.
- Core pipeline works well for supported genres/moods and prompt styles covered by aliases and synonyms.
- The main limitation in the system's design is domain coverage: uncommon styles or ambiguous requests will trigger clarification instead of confident recommendations. For example: If someone put "thinking music" my sytems flags as unclear and prompts for clarity. Even though the system could very well interpret "thinking music" as music for "focus," something my system does recognize and acan retrieve data for.

## Reflection:
It taught me a lot about RAG implementation. The data you give or allow the system to access is all it can work with. I had to broaden my JSON to have more coverage multiple times. Agentic A.I. is really helpful for cleaning up redundancies in code which is another thing I had to deal with. I also understood by the end of this project that for systems like this problems are usually connected and even when they aren't, the fixes very easily can cause small or larger problems elesewhere.

## Screenshot Demo Walkthrough
**Input 1: Standard User Interaction:**
![alt text](<assets/Screenshot 2026-04-19 224122.png>)

**Input 2: Prompt for Clarification:**
![alt text](<assets/Screenshot 2026-04-19 224209.png>)

**Input 3: Discovery Mode:**
![alt text](<assets/Screenshot 2026-04-19 224250.png>)