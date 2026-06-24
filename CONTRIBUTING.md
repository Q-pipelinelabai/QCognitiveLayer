# Contributing

QCognitiveLayer is a research prototype — contributions, experiments and critiques are all welcome.

## Ground rules
- **Keep the core dependency-free.** The engine (`engine/`) must run on the Python 3.9+ standard library.
  Optional capabilities (Zemberek lemmatization, future embeddings) go behind a graceful fallback.
- **Honesty over hype.** Document limitations next to features. If a recall trick only works under certain
  data, say so.
- **Determinism & transparency are features.** Prefer changes that keep recall explainable.

## Good first issues
- A pure-Python (or optional) **embedding / cosine² collapse** second-stage selector (closes the semantic gap).
- **POS-weighted focus** using the POS already stored during lemmatization.
- An **MCP tool** wrapper exposing `recall` / `remember`.
- A benchmark harness beyond `test_500` (multi-conversation, multilingual).

## Dev setup
```bash
python server.py        # http://localhost:8011
# optional: pip install zemberek-python   (Turkish lemmatization)
```

## Pull requests
- Describe the behaviour change and, where relevant, the effect on recall accuracy.
- Don't commit `data/` (checkpoints, chats, config with keys) — `.gitignore` already excludes it.

By contributing you agree your work is released under the project's [MIT License](LICENSE).
