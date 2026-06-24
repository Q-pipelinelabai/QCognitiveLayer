# -*- coding: utf-8 -*-
"""
Cognitive Layer — İZOLOJİK BELLEK MOTORU
========================================
8006 (qcognitive) algoritmasından türetilmiş, saf-numpy/paylaşılabilir geri-getirme çekirdeği.

İlke (İdeal Hayat Paradigması / izoloji):
  * kavram = düğüm (izobit); eş-geçiş = bağ (cooc); cümle = ÖBEK (π-adresli).
  * geri-getirme = YAYILIM (cohf=1, SAF bağ grafı — 8006'da kanıtlanan iyileştirme) + pi_focus
    (en ilgili öbekler = aktive kavramların KESİŞİMİ).
  * çıktı = ÜRETİM DEĞİL: ilgili öbeklerin BİREBİR metni → LLM yorumlar.

API:
  m = Memory()                      # boş ya da load_path ile
  m.remember(text, ts, season, role)   # konuşmayı cümle-cümle π-adresle ekle (graf öğrenir)
  m.recall(query, hops=4, k=8)      # ilgili öbekler + aktivasyon haritası
  m.save(path) / Memory.load(path)
"""
import re, math, pickle, time
from collections import defaultdict

GOLDEN = 1.6180339887498949
WIN = 5                      # cümle-içi eş-geçiş penceresi
CONV_WIN = 40                # KONUŞMA bağlamı penceresi (son N içerik kavramı) — cümleler/mesajlar arası çağrışım
CONV_W = 0.25                # konuşma-içi (cümleler-arası) cooc ağırlığı; cümle-içinden DÜŞÜK ('takım'<->'Fenerbahçe' köprüsü)
SPREAD = 0.84               # yayılım katsayısı (8006 ACT_SPREAD)
SELF_KEEP = 0.13            # düğümün kendi aktivasyonunu koruması (8006)
ACTIVE_CAP = 4000           # her hop'ta tutulan en aktif düğüm (hız)

# Türkçe stopword (SEED'den ve skordan elenir; öbek metninde KALIR)
STOP = set((
    "bir ve veya ya da de da ki mi mı mu mü bu şu o ile için çok daha en ne neyi nedir "
    "kim kime kimi kimin nasıl neden niçin niye nerede hangi kaç mıdır midir ben sen biz siz "
    "ben bana sana ona bize size onlara gibi kadar göre yani ise diye olarak ama fakat ancak "
    "her hep hiç değil evet hayır acaba şey var yok olan olur oldu the and of to in is a"
).split())

_word_re = re.compile(r"[a-zçğıöşü0-9]+")
_sent_re = re.compile(r"(?<=[.!?…])\s+")
# RETRIEVAL filtresi: debug/iç-durum dökümü öbekleri SONUÇTA gösterilmez (veri ham KALIR; sadece surface edilmez)
_DEBUG_CHUNK = re.compile(r"\[matrix:|\[DÜZELTME:|\[ODAK\]|\[conv:|\[memory:|→|chars_\d|flow_ws")
# Türkçe-doğru küçültme (İ->i, I->ı; Python'un default lower()'ı İ'yi 'i̇' yapıp böler)
_TR = str.maketrans({"İ": "i", "I": "ı", "Ş": "ş", "Ğ": "ğ", "Ü": "ü", "Ö": "ö", "Ç": "ç"})


def _tr_lower(s):
    return (s or "").translate(_TR).lower().replace("̇", "")


# ---------------------------------------------------------------- ZEMBEREK lemmatization (OPSİYONEL, graceful)
# Konuşurken kelimeler köke indirgenir: 'takımı/takım/takımlar'->'takım', 'tuttuğum'->'tut', 'ablamın'->'abla'.
# Böylece aylar arası farklı çekimli aynı kavram BAĞLANIR + morfoloji recall hatası biter. Zemberek yoksa
# kelime aynen kalır (ürün yine çalışır, morfoloji zayıf). POS de saklanır -> tür-ağırlıklı odak.
LEMMA_ON = True
_morph = None
_lem_cache = {}
_lem_failed = False
POS = {}     # lemma -> kaba tür ('isim'/'fiil'/'sifat'/'kisi'/'diger') ; tür-ağırlıklı odak için


def _ensure_morph():
    global _morph, _lem_failed
    if _morph is not None or _lem_failed:
        return _morph
    try:
        import logging
        logging.disable(logging.CRITICAL)
        from zemberek import TurkishMorphology
        _morph = TurkishMorphology.create_with_defaults()
    except Exception:
        _lem_failed = True
    return _morph


_POSMAP = {"Noun": "isim", "Verb": "fiil", "Adj": "sifat", "Adv": "zarf",
           "Pron": "zamir", "Num": "sayi", "Prop": "kisi"}


def _lemma(w):
    """Kelimenin kökü (lemma) + POS'u kaydet. Zemberek yoksa kelimeyi döndürür."""
    if not LEMMA_ON or _lem_failed:
        return w
    c = _lem_cache.get(w)
    if c is not None:
        return c
    lem = w
    m = _ensure_morph()
    if m is not None:
        try:
            s = None
            for x in m.analyze(w):
                s = x
                break
            if s is not None:
                di = getattr(s, "item", None)
                lm = getattr(di, "lemma", None) if di is not None else None
                if not lm:
                    try:
                        lm = s.get_stem()
                    except Exception:
                        lm = None
                if lm and lm not in ("UNK", "Unk", "?"):
                    lem = _tr_lower(lm)
                    if lem.endswith("mek") or lem.endswith("mak"):   # fiil mastarını kısalt
                        lem = lem[:-3]
                try:
                    morphs = [str(getattr(md, "name", md)) for md in s.get_morphemes()]
                    if morphs:
                        POS.setdefault(lem, _POSMAP.get(morphs[0].split(":")[0], "diger"))
                except Exception:
                    pass
        except Exception:
            pass
    _lem_cache[w] = lem
    return lem


def pi_address(k):
    """Altın-açı π-adres: çakışmasız kanonik sıra (kitabın 'rezonans imkânsız')."""
    return 2.0 * math.pi * ((k / GOLDEN) % 1.0)


def _tok(text):
    return _word_re.findall(_tr_lower(text))


def _is_content(w):
    return len(w) >= 2 and w not in STOP and not w.isdigit()


def _trigrams(w):
    """Kelimenin karakter-trigram kümesi (yanlış-yazım benzerliği için). '#' = kelime sınırı."""
    s = "#" + w + "#"
    return {s[i:i + 3] for i in range(len(s) - 2)} if len(s) >= 3 else {s}


def _levenshtein(a, b, cap=2):
    """Düzenleme mesafesi, cap'i aşınca erken çıkar (hız)."""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        mn = i
        for j, cb in enumerate(b, 1):
            v = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (0 if ca == cb else 1))
            cur.append(v)
            if v < mn:
                mn = v
        if mn > cap:
            return cap + 1
        prev = cur
    return prev[-1]


class Memory:
    def __init__(self):
        self.words = []                       # id -> kelime
        self.idx = {}                         # kelime -> id
        self.freq = []                        # id -> sıklık
        self.adj = defaultdict(lambda: defaultdict(float))   # i -> {j: ağırlık} (simetrik cooc)
        self.chunks = []                      # öbekler: {text, ids, ts, season, role, k, pi}
        self.node2chunk = defaultdict(set)    # düğüm -> içinde geçtiği öbek indeksleri
        self.conv2chunk = defaultdict(list)   # KONUŞMA -> öbek indeksleri (hiyerarşi: sohbet bütününe in)
        self.degree = defaultdict(float)      # toplam bağ ağırlığı (hub tespiti)
        self._recent = defaultdict(list)      # sezon -> son içerik kavramları (konuşma-düzeyi cooc penceresi)
        self._sim_idx = None                  # trigram ters indeks (yanlış-yazım fuzzy eşleşme); tembel kurulur
        self._sim_n = -1

    # ---------------------------------------------------------------- düğüm
    def _node(self, w):
        i = self.idx.get(w)
        if i is None:
            i = len(self.words)
            self.idx[w] = i
            self.words.append(w)
            self.freq.append(0.0)
        return i

    def _specificity(self, i):
        """İzobit/özgüllük = IDF: kaç öbekte geçiyor (df) az ise NADİR=önemli; çok ise hub=bastırılır.
        (veriyi temizlemeden 'dostum/[DÜZELTME]' gibi evrensel filler'ı sorgu-anında bastırır)."""
        df = len(self.node2chunk.get(i, ())) or 1
        return math.log((len(self.chunks) + 1.0) / df)

    # ---------------------------------------------------------------- YANLIŞ-YAZIM (fuzzy) emniyeti
    def _fuzzy(self, word, min_sim=0.5):
        """Tam eşleşmeyen sorgu kelimesi için EN BENZER mevcut kelimeyi bul (trigram Jaccard).
        'Fenrbahçe'->'fenerbahçe', 'Japnya'->'japonya'. Sadece tam-eşleşme YOKKEN çağrılır."""
        if len(word) < 3 or not self.words:
            return None
        if self._sim_idx is None or self._sim_n != len(self.words):    # tembel/sync ters indeks
            idx = defaultdict(list)
            for wid, w in enumerate(self.words):
                for g in _trigrams(w):
                    idx[g].append(wid)
            self._sim_idx = idx
            self._sim_n = len(self.words)
        wg = _trigrams(word)
        shared = defaultdict(int)
        for g in wg:
            for wid in self._sim_idx.get(g, ()):
                shared[wid] += 1
        best, best_score = None, 0.0
        for wid, sh in sorted(shared.items(), key=lambda kv: -kv[1])[:30]:   # en çok trigram paylaşan adaylar
            ow = self.words[wid]
            if abs(len(ow) - len(word)) > 3:
                continue
            jac = sh / (len(wg) + len(_trigrams(ow)) - sh)            # trigram Jaccard
            ed = _levenshtein(word, ow, cap=2)                       # + tek/çift harf typo yakala
            maxed = 1 if max(len(word), len(ow)) < 6 else 2
            score = max(jac, (0.97 - 0.12 * ed) if ed <= maxed else 0.0)
            if score > best_score:
                best_score, best = score, wid
        return best if best_score >= min_sim else None

    # ---------------------------------------------------------------- yazma
    def remember(self, text, ts=None, season="default", role="user", conv_id=None):
        """Metni cümle-cümle ÖBEK olarak ekle; graf (düğüm+bağ) öğrenir. conv_id = KONUŞMA grubu
        (hiyerarşi: öbek=cümle/π-adres, conv=sohbet). Döner: eklenen öbek sayısı."""
        if ts is None:
            ts = time.time()
        conv = conv_id or season
        added = 0
        msg_content = []                       # bu mesajın TÜM içerik kavramları (konuşma-düzeyi cooc için)
        for sent in _split_sentences(text):
            surf = _tok(sent)
            if not surf:
                continue
            toks = [_lemma(w) for w in surf]      # KÖKE indirge (morfoloji + aylar arası farklı çekim BAĞLANIR)
            ids = [self._node(w) for w in toks]
            for w_i, i in zip(toks, ids):
                self.freq[i] += 1.0
            # eş-geçiş (simetrik, mesafe-ağırlıklı) = CÜMLE-İÇİ bağ öğrenme
            for a in range(len(ids)):
                for b in range(a + 1, min(a + WIN, len(ids))):
                    i, j = ids[a], ids[b]
                    if i == j:
                        continue
                    w = 1.0 / (b - a)
                    self.adj[i][j] += w
                    self.adj[j][i] += w
                    self.degree[i] += w
                    self.degree[j] += w
            cid = len(self.chunks)
            content_ids = [i for w, i in zip(toks, ids) if _is_content(w)]
            self.chunks.append({
                "text": sent.strip(), "ids": content_ids, "ts": ts,
                "season": season, "conv": conv, "role": role, "k": cid, "pi": pi_address(cid),
            })
            for i in set(content_ids):
                self.node2chunk[i].add(cid)
            self.conv2chunk[conv].append(cid)
            msg_content.extend(content_ids)
            added += 1

        # KONUŞMA-DÜZEYİ cooc: bu mesajın kavramlarını sezonun SON kavramlarına bağla (cümleler/mesajlar arası
        # çağrışım). 'Fenerbahçe maçı' + 'tuttuğum takım' ayrı cümlelerde olsa da takım<->Fenerbahçe köprüsü kurulur.
        recent = self._recent[season]
        new = set(msg_content)
        for i in new:
            for j in recent:
                if i != j:
                    self.adj[i][j] += CONV_W
                    self.adj[j][i] += CONV_W
                    self.degree[i] += CONV_W
                    self.degree[j] += CONV_W
        recent.extend(msg_content)
        self._recent[season] = recent[-CONV_WIN:]
        return added

    # ---------------------------------------------------------------- yayılım (cohf=1)
    def _spread(self, seeds, hops):
        """SAF bağ grafı yayılımı (8006 cohf=1): act = SELF·act + SPREAD·(komşu ortalaması)."""
        act = {s: 1.0 for s in seeds}
        for _ in range(hops):
            nxt = defaultdict(float)
            for i, ai in act.items():
                nbr = self.adj.get(i)
                if not nbr:
                    nxt[i] += SELF_KEEP * ai
                    continue
                wsum = 0.0
                for j, w in nbr.items():
                    nxt[j] += SPREAD * w * ai
                    wsum += w
                nxt[i] += SELF_KEEP * ai
            # seed'leri sabit tut (çıpa) + en aktif ACTIVE_CAP düğümü koru (hız)
            for s in seeds:
                nxt[s] = max(nxt.get(s, 0.0), 1.0)
            if len(nxt) > ACTIVE_CAP:
                top = sorted(nxt.items(), key=lambda kv: -kv[1])[:ACTIVE_CAP]
                nxt = defaultdict(float, dict(top))
            act = nxt
        # normalize
        m = max(act.values()) if act else 1.0
        return {i: v / m for i, v in act.items()} if m > 0 else act

    # ---------------------------------------------------------------- okuma (recall)
    def recall(self, query, hops=4, k=8, season=None, recency_w=0.15):
        """Sorguya en ilgili öbekleri getir (pi_focus). Döner: dict(hits, hubs, chunks, map)."""
        qtok = [w for w in (_lemma(t) for t in _tok(query)) if _is_content(w)]   # sorgu da KÖKE indirgenir
        seeds = []
        for w in qtok:
            i = self.idx.get(w)
            if i is None:                          # tam eşleşme yok -> YANLIŞ-YAZIM emniyeti (en benzer kelime)
                i = self._fuzzy(w)
            if i is not None and i not in seeds:
                seeds.append(i)
        if not seeds or not self.chunks:
            return {"hits": [w for w in qtok if w in self.idx], "hubs": [], "chunks": [], "map": []}

        act = self._spread(seeds, hops)
        seedset = set(seeds)

        # 3+ HUB = en aktif İÇERİK kavramları (özgüllük-ağırlıklı); seed + sayı/stopword hariç
        hub_scores = sorted(
            ((self.words[i], act[i] * self._specificity(i)) for i in act
             if i not in seeds and i < len(self.words) and _is_content(self.words[i])),
            key=lambda kv: -kv[1])
        hubs = [w for w, s in hub_scores[:6] if s > 0]

        # pi_focus: her öbek skoru = içindeki aktive kavramların KESİŞİM ağırlığı (izobit×act)
        # ADAY ÖBEK = SEED-ÇAPALI: yalnız sorgu kelimelerini YA DA onların GÜÇLÜ doğrudan komşularını
        # içeren öbekler. Global hub flooding'i dışarıda bırakır -> çöp/matrix öbekleri sorguyla
        # gerçek kelime paylaşmadıkça aday olmaz (veriyi temizlemeye gerek kalmaz).
        focus = set(seeds)
        for s in seeds:
            # seed'in en güçlü AMA ÖZGÜL komşuları (evrensel filler/hub'ı atla -> çöp öbek aday olmaz)
            nbr = sorted(self.adj.get(s, {}).items(), key=lambda kv: -kv[1] * self._specificity(kv[0]))
            focus.update(j for j, _ in nbr[:20] if self._specificity(j) >= 4.0)
        cand = set()
        for i in focus:
            cand |= self.node2chunk.get(i, set())
            if len(cand) > 8000:
                break

        now = time.time()
        scored = []
        for cid in cand:
            c = self.chunks[cid]
            ids = c["ids"]
            if not ids:
                continue
            if _DEBUG_CHUNK.search(c["text"]):       # debug/iç-durum dökümü -> surface ETME (veri ham kalır)
                continue
            idset = set(ids)
            # DOĞRUDAN örtüşme = chunk'ta geçen sorgu SEED'lerinin IDF toplamı -> NADİR seed baskın
            # ("kızımın adı ne": her isimde 'adı' var ama 'kızımın' nadir -> doğru öbeği seçer)
            direct = sum(self._specificity(i) for i in seedset if i in idset)
            # ASSOCİATİF = yayılımdan gelen ÇAĞRIŞIMSAL kavramlar (ortak kelime olmasa da connectome köprüsü:
            # 'takım' sorusu -> öğrenilmiş takım<->Fenerbahçe kenarı -> Fenerbahçe öbeğini getirir)
            assoc = sum(act.get(i, 0.0) * self._specificity(i) for i in ids if i not in seedset)
            s = direct + 0.6 * assoc
            if s <= 0:
                continue
            s /= (1.0 + 0.15 * len(ids))          # ÇOK HAFİF uzunluk cezası (dilüsyonu hafif cezalandır)
            age_days = max(0.0, (now - c["ts"]) / 86400.0)
            s *= (1.0 + recency_w / (1.0 + age_days))   # recency hafif bonus
            scored.append((s, cid))
        scored.sort(key=lambda x: -x[0])

        # HİYERARŞİ: cümle skorlarını KONUŞMA'ya topla -> önce en ilgili SOHBETE odaklan, sonra İÇERİĞİNE in.
        # (π-adresli tek cümleler hatırlanır AMA sohbetin tamamı bağlam taşır -> aylar önceki sohbet bütünüyle gelir)
        conv_members = defaultdict(list)
        for s, cid in scored:
            cv = self.chunks[cid].get("conv") or self.chunks[cid]["season"]
            conv_members[cv].append((s, cid))
        conv_rank = []
        for cv, mem in conv_members.items():
            mem.sort(key=lambda x: -x[0])
            cscore = mem[0][0] + 0.4 * sum(s for s, _ in mem[1:])   # en güçlü hit + destekleyici bağlam
            conv_rank.append((cscore, cv))
        conv_rank.sort(key=lambda x: -x[0])

        score_of = {cid: s for s, cid in scored}
        chosen = []
        seen_text = set()
        for _, cv in conv_rank:
            # SOHBETİN İÇERİĞİNE İN: o sohbetin TÜM öbekleri (eşleşen + KARDEŞ cümleler), eşleşen önce + π/sıra
            cids = self.conv2chunk.get(cv) or [cid for _, cid in conv_members[cv]]
            cids = sorted(cids, key=lambda cid: (-score_of.get(cid, 0.0), self.chunks[cid]["k"]))
            take = 0
            for cid in cids:
                c = self.chunks[cid]
                if _DEBUG_CHUNK.search(c["text"]):
                    continue
                key = c["text"].lower()
                if key in seen_text:
                    continue
                seen_text.add(key)
                chosen.append({"text": c["text"], "score": round(float(score_of.get(cid, 0.0)), 3),
                               "ts": c["ts"], "season": c["season"], "conv": c.get("conv") or c["season"],
                               "role": c["role"], "k": c["k"]})
                take += 1
                if take >= k or len(chosen) >= k:    # en ilgili sohbet içeriğini doldurur; kalırsa sonraki sohbet
                    break
            if len(chosen) >= k:
                break
        return {"hits": [self.words[i] for i in seeds], "hubs": hubs,
                "chunks": chosen, "map": [[w, round(float(sc), 3)] for w, sc in hub_scores[:12]]}

    # ---------------------------------------------------------------- kalıcılık
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({
                "words": self.words, "idx": self.idx, "freq": self.freq,
                "adj": {i: dict(d) for i, d in self.adj.items()},
                "chunks": self.chunks,
                "node2chunk": {i: list(s) for i, s in self.node2chunk.items()},
                "conv2chunk": {k: v for k, v in self.conv2chunk.items()},
                "degree": dict(self.degree),
                "_recent": {k: v for k, v in self._recent.items()},
            }, f)

    @staticmethod
    def load(path):
        m = Memory()
        with open(path, "rb") as f:
            d = pickle.load(f)
        m.words = d["words"]; m.idx = d["idx"]; m.freq = d["freq"]; m.chunks = d["chunks"]
        for i, dd in d["adj"].items():
            m.adj[i] = defaultdict(float, dd)
        for i, s in d["node2chunk"].items():
            m.node2chunk[i] = set(s)
        m.conv2chunk = defaultdict(list, d.get("conv2chunk", {}))
        m.degree = defaultdict(float, d["degree"])
        m._recent = defaultdict(list, d.get("_recent", {}))
        return m

    def stats(self):
        return {"kelime": len(self.words), "öbek": len(self.chunks),
                "bağ": sum(len(d) for d in self.adj.values()) // 2}


def _split_sentences(text):
    text = " ".join((text or "").split())
    return [s for s in _sent_re.split(text) if s.strip()]
