# Architecture

The technical write-up of QCognitiveLayer: data structures, the mathematics of recall, and the
request/learning flow. For the *why*, see the [README](../README.md).

---

## 1. The izolojik memory (`engine/memory.py`)

Everything is one graph of **izobit nodes** (concepts) joined by **bonds** (co-occurrence), plus a flat list
of **öbeks** (sentences) that index into those nodes and roll up into **conversations**.

### Data structures

| structure | role |
|---|---|
| `words[]`, `idx{}` | id ⇄ concept string |
| `freq[]` | per-concept frequency |
| `adj{i:{j:w}}` | symmetric, distance-weighted **bond graph** |
| `degree{}` | total bond weight per node (hub detection) |
| `chunks[]` | **öbeks**: `{text, ids, ts, season, conv, role, k, pi}` (`ids` are *content* node ids) |
| `node2chunk{i:set}` | concept → öbeks it appears in |
| `conv2chunk{conv:[cid]}` | **conversation → its öbeks** (hierarchy / sibling context) |
| `_recent{season:[ids]}` | rolling conversation-context window (cross-sentence cooc) |
| `_sim_idx` | trigram inverted index (lazy) for fuzzy matching |

A concept node is **stored as a lemma** when Zemberek is available (`takımı/takım/takımlar → tak`), so
different inflections of the same idea are the same node. Without Zemberek it falls back to the surface form.

---

## 2. Writing — `remember(text, ts, season, role, conv_id)`

For each sentence in the message:

1. **Tokenize** → surface tokens (`[a-zçğıöşü0-9]+`, Turkish-correct lowercasing `İ→i`).
2. **Lemmatize** each token (optional Zemberek) → the node key.
3. **Intra-sentence bonds** — for every pair within a window `WIN = 5`, add weight `1/dist` to `adj`
   (closer words bind harder).
4. **Create the öbek** — store the *original* sentence text, its content-node ids, and a **π-address**
   `pi = 2π · frac(k/φ)` where `k` is the global creation order.
5. Index into `node2chunk` and `conv2chunk[conv]` (default `conv = season`).

Then, once per message:

6. **Conversation-level bonds** — link this message's concepts to the season's recent concepts
   (`_recent`, window `CONV_WIN = 40`, weight `CONV_W = 0.25`). This is what lets *"team"* and *"Fenerbahçe"*
   bond even though they sat in different sentences of the same chat.

> `freq`, `degree`, bonds all accumulate — the graph **learns** continuously and **strengthens** with use.

---

## 3. Reading — `recall(query, hops, k)`

### 3.1 Seeds (three safety nets)

```
query token → lemma →  exact in idx?  →  fuzzy match?  →  seed | dropped
```

`_fuzzy(word)` only fires on an exact miss. It uses a **trigram inverted index** to gather candidates, then
ranks each by **trigram Jaccard** *or* **Levenshtein edit-distance** (≤1 for short words, ≤2 for longer),
guarded by a length-difference cap so gibberish never matches.

### 3.2 Spreading activation (`cohf = 1`)

Pure-graph diffusion (no phase gating — the key improvement over the original substrate):

```
act = {seed: 1.0}
repeat hops times:
    act'[i] = SELF_KEEP · act[i] + SPREAD · ( Σ_j w_ij·act[j] / Σ_j w_ij )
    keep seeds pinned at 1.0 ; keep top ACTIVE_CAP nodes
normalize
```

`SELF_KEEP = 0.13`, `SPREAD = 0.84`, `ACTIVE_CAP = 4000`. Low `hops` (2–3) keeps it query-specific on dense
graphs; higher hops reach further but over-diffuse.

### 3.3 Saliency = IDF (izobit energy)

```
specificity(i) = log( (N_chunks + 1) / df_i )      df_i = #öbeks containing i
```

Rare concepts dominate; universal filler ("dostum", "tamam") is suppressed at query time — **without
touching the stored data**.

### 3.4 Candidate gathering (seed-anchored)

Candidates are öbeks containing a **seed** or a seed's **specific** neighbour (IDF-gated, so hub neighbours
are skipped). This keeps global hubs from flooding the candidate set.

### 3.5 Scoring

For each candidate öbek:

```
direct = Σ_{seed ∈ chunk}  specificity(seed)          # rare query words disambiguate
assoc  = Σ_{i ∈ chunk, i∉seed}  act[i]·specificity(i) # connectome bridge (multi-hop)
score  = (direct + 0.6·assoc) / len(chunk)^0.5 · recency
```

Debug/system-dump öbeks are filtered from results (the stored data stays raw; they are just never
surfaced). `direct` makes *"kızımın adı"* beat *"babamın adı"* because the rare seed `kız` outweighs the
shared `ad`.

### 3.6 Hierarchy — conversation focus, then content

```
for each conversation cv:
    cscore(cv) = best_member_score + 0.4 · Σ(other member scores)
rank conversations by cscore
for top conversations:
    emit its öbeks from conv2chunk, matched first then π/order siblings, up to k
```

So a query first selects the most relevant **conversation**, then returns that conversation's **content** —
including sibling sentences that didn't match the query directly (a months-old chat comes back *whole*).
With a single conversation this degrades gracefully to flat top-k recall.

---

## 4. Background learning & hybrid continuity (`server.py`)

```
recall(message)                         # BEFORE the answer (builds prompt)  — synchronous
stream LLM answer
append last-60 history + save_season     # synchronous, cheap → next turn continuity
_learn_bg(...)                           # BACKGROUND thread:
    MEM.remember(user) ; MEM.remember(answer) ; MEM.save()   # lemma + checkpoint
    autonomous.scan(...)                                     # 10-step association probe
```

- **Learning is parallel** — graph update, lemmatization and the full-pickle checkpoint never block the
  response. The next turn's recall waits on a lock only if learning is still running.
- **Continuity is hybrid** — the last `HISTORY_INJECT = 60` Q-A of the *current* chat go to the LLM
  verbatim; everything older is provided selectively by the recall above.

The system prompt + recalled memory block + last-60 history + new message form the final LLM prompt,
localized to `en` / `tr`.

---

## 5. Autonomous thought (`engine/autonomous.py`)

Query-triggered, **not** continuous: after each turn it seeds from the message, "wanders" the graph 10
steps (drifting to peripheral concepts), and if it finds an unusually strong, *non-obvious* öbek (above a
normalized threshold) it stores it as a `pending_aha` to fold into the **next** turn's prompt.

---

## 6. LLM router (`llm/router.py`)

- `chat_stream()` yields deltas: **Gemini** via `streamGenerateContent?alt=sse`, **Ollama** via
  `/api/chat` `stream:true`.
- `list_models()` fetches the **live** Gemini model list for the key (filtered to chat models, newest
  first), falling back to a curated list; Ollama models come from `/api/tags`. Uses `127.0.0.1` (not
  `localhost`) to avoid the Windows IPv6-empty-instance trap.

---

## 7. Tuning knobs

| constant | default | effect |
|---|---|---|
| `WIN` | 5 | intra-sentence bond window |
| `CONV_WIN` / `CONV_W` | 40 / 0.25 | conversation-level (cross-sentence) association |
| `SPREAD` / `SELF_KEEP` | 0.84 / 0.13 | diffusion vs. self-retention |
| `hops` | 3 | spreading depth (config, 2–7) |
| `assoc` weight | 0.6 | associative vs. direct-overlap balance |
| `HISTORY_INJECT` | 60 | verbatim recent turns sent to the LLM |
| recall `k` | 8 | öbeks injected as memory context |

---

## 8. Performance

Pure Python, no GPU. The 500-conversation benchmark trains in seconds and recalls in tens of milliseconds.
The checkpoint is a single pickle; it grows with memory and is saved in the background. For very large
memories, batching the save (roadmap) keeps the background step light.
