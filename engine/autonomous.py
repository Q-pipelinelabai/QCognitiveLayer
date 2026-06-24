# -*- coding: utf-8 -*-
"""
Cognitive Layer — OTONOM DÜŞÜNCE (query-triggered)
==================================================
8006 iterate()/_resonance_aha mantığının hafif numpy hâli.

DAVRANIŞ (kullanıcı tarifi):
  * SÜREKLİ DEĞİL — soru ile TETİKLENİR, arka planda 10 adım yürür, sonra DURUR.
  * Sorudan tohumlanıp grafta "gezinir" (drift): her adımda örüntüyü çevre-kavramla uzatır.
  * Doğrudan recall'da OLMAYAN ama örüntüye GÜÇLÜ rezonanslı bir öbek ararsa = "aha".
  * Eşik aşılırsa bulgu SONRAKİ tura cevaba entegre edilir; aşılmazsa sessiz (gürültü yok).

scan(memory, query, exclude_texts, steps=10, threshold=...) -> dict|None
"""
import math
from memory import _tok, _is_content

AHA_THRESHOLD = 0.55     # bulgunun "ciddi eşleşme" sayılması için min normalize skor
DRIFT_TOP = 8            # her adımda örüntüye katılabilecek çevre-kavram havuzu


def scan(memory, query, exclude_texts=None, steps=10, hops=3, threshold=AHA_THRESHOLD):
    """Sorudan gezinerek güçlü+beklenmedik bir öbek bul. Döner: {text, score, ...} ya da None."""
    exclude = set((t or "").lower() for t in (exclude_texts or []))
    qtok = [w for w in _tok(query) if _is_content(w)]
    seeds = [memory.idx[w] for w in qtok if w in memory.idx]
    if not seeds or not memory.chunks:
        return None

    pattern = list(seeds)                 # büyüyen düşünce örüntüsü (engram)
    seen_nodes = set(seeds)
    best = None
    for _ in range(steps):
        act = memory._spread(pattern, hops)
        # DRIFT: örüntüye, henüz alınmamış en aktif çevre-kavramı ekle (konudan hafif kay)
        cands = sorted(((i, act[i] * memory._specificity(i)) for i in act if i not in seen_nodes),
                       key=lambda kv: -kv[1])[:DRIFT_TOP]
        if cands:
            ni = cands[0][0]
            pattern.append(ni)
            seen_nodes.add(ni)
        # örüntüye en rezonanslı öbeği ara (doğrudan recall'da OLMAYANLARDAN)
        cand_chunks = set()
        for i in act:
            cand_chunks |= memory.node2chunk.get(i, set())
        for cid in cand_chunks:
            c = memory.chunks[cid]
            if c["text"].lower() in exclude:
                continue
            s = sum(act.get(i, 0.0) * memory._specificity(i) for i in c["ids"])
            if s <= 0:
                continue
            if best is None or s > best[0]:
                best = (s, cid)

    if best is None:
        return None
    # normalize: örüntünün kendi en-iyi-öz-skoruna göre (ölçek-bağımsız eşik)
    act0 = memory._spread(seeds, hops)
    self_top = max((sum(act0.get(i, 0.0) * memory._specificity(i) for i in memory.chunks[cid]["ids"])
                    for cid in range(len(memory.chunks))), default=1.0) or 1.0
    norm = best[0] / self_top
    if norm < threshold:
        return None
    c = memory.chunks[best[1]]
    return {"text": c["text"], "score": round(float(norm), 3),
            "ts": c["ts"], "season": c["season"], "role": c["role"]}
