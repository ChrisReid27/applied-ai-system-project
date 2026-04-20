# 🎧 Model Card

**Song Recommender Version 1.0**  

---

## 1. What are the limitations or biases in your system?

- **Coverage biases:** The recommender only knows what exists in `songs.csv` and `rag_corpus.json`. They're both local. If a genre, artist style, or mood framing is underrepresented there, results will be weaker or repetitive.
- **Heuristic biases:** Profile inference from text relies on aliases/synonyms I put in manually. This is good for common terms (for example, "workout" or "chill") but can underperform on niche phrases, slang, or multilingual requests.
- **No personalization:** The system does not learn from individual listening history over time, so recommendations are session-based rather than long-term personalized.
- **Popularity/proxy effects:** Features like energy, danceability, and valence can over-prioritize songs that numerically match the profile, even when lyrical themes or context might not fit user intent.
- **Small-domain reliability limit:** Because this is a deterministic and local RAG pipeline, it's explainable and stable, but less flexible than a large, continuously trained recommendation A.I.'s.

---

## 2. Could your AI be misused, and how would you prevent that? 

- **Inappropriate-context recs.** Users could request music for unsafe contexts (e.g., driving while drunk or intoxicated) and still receive energetic or distracting tracks. I would need to add some safety rules that can detect unsafe contexts and maybe return safer defaults or clarification responses.

---

## 3. What surprised you while testing your AI's reliability?

- **Deterministic stability was stronger than expected.** For repeated prompts, ranking behavior was very consistent, which made testing and debugging easier.
- **Input phrasing mattered more than expected.** Small wording changes ("focus" vs. "thinking music") could move the parser from a confident mapping to a clarification path.
- **Fixes had cross-effects.** Expanding alias/synonym rules often improved one test scenario while shifting the behavior for another, highlighting how parsing, ranking, and retrieval are very interconnected in a system like this.

---

## 4. Describe your collaboration with AI during this project. Identify one instance when the AI gave a helpful suggestion and one instance where its suggestion was flawed or incorrect. 

- **Where A.I. Helped:** It definitely helped me when it suggested making a full list of synonyms and general aliasing for words so that when users input something, the model wouldn't get stuck up on that.
- **Where it didn't help:** It was suggesting just printing the values for valence, energy, etc. into premade responses which wasn't the goal of the system. I needed it to respond back to what the user put in with natural language and actually use what the user put in as context.
- **Overall** I collaborated well with A.I. since it obviously made things go faster. I had a lot of different ways I thought of doing this project and it helped me narrow down to which ones were manageable and actually maintainable.

---

## Reflection: What this project says about me as an AI engineer.
This project made me understand so much more about how LLM work, specifically models using RAG. I think as and AI engineer I'm very comfortable with the concepts surrounding RAG. I also think I'm more careful now when coding. Working in codebases where there are som many moving parts feels more natural to me now, especially with the help of A.I. to stay organized and catch any discrepancies, which I had a lot of. I made a lot of small mistakes that A.I. helped me fix. So overall I think I've grown as an engineer who has now gotten more comfortable with some of the inner workings of these types of models and gotten better at using them too.