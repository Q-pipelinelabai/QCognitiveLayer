# -*- coding: utf-8 -*-
"""
Cognitive Layer — SUNUCU (chat + sezonlar + ayarlar + prompt builder)
=====================================================================
İzolojik bellek katmanı bir LLM'in ÜSTÜNE oturur:
  girdi -> izoloji RECALL (geçmiş konuşmalardan ilgili öbekler) + son-60 Q-A history
        -> PROMPT BUILDER -> LLM (Gemini/Ollama) -> cevap -> belleğe yaz (graf öğrenir).
Otonom düşünce soru ile tetiklenir (10 adım, arka plan); bulursa SONRAKİ tura entegre.

Başlat:  python server.py    ->  http://localhost:8010
"""
import os, sys, json, time, threading, glob
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "engine"))
sys.path.insert(0, os.path.join(HERE, "llm"))
from memory import Memory          # noqa: E402
import autonomous                   # noqa: E402
import router                       # noqa: E402

PORT = 8011
DATA = os.path.join(HERE, "data")
CKPT = os.path.join(DATA, "checkpoints", "memory.pkl")
SEASON_DIR = os.path.join(DATA, "seasons")
CONFIG = os.path.join(DATA, "config.json")
HISTORY_INJECT = 60                 # prompt'a enjekte edilen son Q-A çifti sayısı

# ----------------------------------------------------------------- durum
MEM = Memory.load(CKPT) if os.path.exists(CKPT) else Memory()
_lock = threading.Lock()


def load_config():
    if os.path.exists(CONFIG):
        return json.load(open(CONFIG, encoding="utf-8"))
    return {"hops": 3, "model": "gemini:gemini-3.5-flash", "api_key": "", "lang": "en"}


def save_config(c):
    json.dump(c, open(CONFIG, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


def season_path(sid):
    safe = "".join(ch for ch in sid if ch.isalnum() or ch in "-_") or "default"
    return os.path.join(SEASON_DIR, f"{safe}.json")


def load_season(sid):
    p = season_path(sid)
    if os.path.exists(p):
        return json.load(open(p, encoding="utf-8"))
    return {"id": sid, "name": sid, "history": [], "pending_aha": None, "created": time.time()}


def save_season(s):
    json.dump(s, open(season_path(s["id"]), "w", encoding="utf-8"), ensure_ascii=False, indent=2)


def list_seasons():
    out = []
    for p in sorted(glob.glob(os.path.join(SEASON_DIR, "*.json"))):
        try:
            s = json.load(open(p, encoding="utf-8"))
            out.append({"id": s["id"], "name": s.get("name", s["id"]),
                        "turns": len(s.get("history", [])), "created": s.get("created", 0)})
        except Exception:
            pass
    return sorted(out, key=lambda x: -x["created"])


# ----------------------------------------------------------------- prompt builder
SYSTEM_BASE = {
    "en": (
        "You are an assistant connected to an izolojik LONG-TERM MEMORY layer. Below you may receive "
        "pieces of information associatively recalled from past conversations.\n"
        "RULES:\n"
        "1) Memory pieces are CONTEXT ONLY — if irrelevant to the question, IGNORE them; do not hallucinate.\n"
        "2) On conflict, trust the MOST RECENT (later date) information.\n"
        "3) Answer naturally, fluently and concisely; use memory when needed, don't parrot it.\n"
        "4) An '[assoc]' note is a possibly indirect recall; consider it if useful.\n"
    ),
    "tr": (
        "Sen, izolojik bir UZUN-DÖNEM HAFIZA katmanına bağlı bir asistansın. Aşağıda, geçmiş "
        "konuşmalardan çağrışımsal olarak hatırlanan bilgi parçaları verilebilir.\n"
        "KURALLAR:\n"
        "1) Hafıza parçaları YALNIZCA bağlamdır — soruyla ilgisizse YOK SAY, uydurma yapma.\n"
        "2) Çelişki olursa EN YENİ (tarihi geç) bilgiye güven.\n"
        "3) Doğal, akıcı ve kısa-öz cevap ver; hafızayı ezbere tekrar etme, gerektiğinde kullan.\n"
        "4) '[çağrışım]' notu, konuyla dolaylı bağlı bir hatırlama olabilir; faydalıysa değerlendir.\n"
    ),
}
_LBL = {
    "en": {"mem": "## MEMORY (relevant records from past chats):", "focus": "## FOCUS CONCEPTS: ",
           "assoc": "## [assoc] (strong relation caught last turn): "},
    "tr": {"mem": "## HAFIZA (geçmiş konuşmalardan ilgili kayıtlar):", "focus": "## ODAK KAVRAMLAR: ",
           "assoc": "## [çağrışım] (önceki turda yakalanan güçlü ilişki): "},
}


def build_prompt(season, recall, aha, lang="en"):
    """Sistem talimatı + izoloji recall + (varsa) çağrışım. Döner: system metni (dile göre)."""
    lang = lang if lang in SYSTEM_BASE else "en"
    lbl = _LBL[lang]
    parts = [SYSTEM_BASE[lang]]
    if recall.get("chunks"):
        parts.append("\n" + lbl["mem"])
        for c in recall["chunks"]:
            tag = time.strftime("%Y-%m-%d", time.localtime(c["ts"]))
            parts.append(f"- [{tag} · {c['role']}] {c['text']}")
    if recall.get("hubs"):
        parts.append("\n" + lbl["focus"] + ", ".join(recall["hubs"]))
    if aha:
        parts.append("\n" + lbl["assoc"] + aha["text"])
    return "\n".join(parts)


def history_messages(season):
    msgs = []
    for h in season.get("history", [])[-HISTORY_INJECT:]:
        msgs.append({"role": "user", "content": h["q"]})
        msgs.append({"role": "assistant", "content": h["a"]})
    return msgs


# ----------------------------------------------------------------- ÖĞRENME + otonom (ARKA PLAN, paralel)
def _learn_bg(sid, message, answer, ts, exclude, hops):
    """Cevap aktıktan SONRA arka planda: graf öğrenme (zemberek lemma user+assistant) + kayıt + otonom tarama.
    Cevabı/isteği BLOKLAMAZ. Anlık süreklilik sezon-history ile zaten senkron sağlanır."""
    def run():
        try:
            with _lock:
                MEM.remember(message, ts=ts, season=sid, role="user")
                MEM.remember(answer, ts=ts, season=sid, role="assistant")
                MEM.save(CKPT)
        except Exception as e:
            print("[öğrenme HATA]", repr(e), flush=True)
        try:
            with _lock:                      # otonom: güncellenmiş bellekte tara (10 adım)
                res = autonomous.scan(MEM, message, exclude_texts=exclude, steps=10, hops=max(2, hops - 1))
            s = load_season(sid)
            s["pending_aha"] = res
            save_season(s)
        except Exception as e:
            print("[otonom HATA]", repr(e), flush=True)
    threading.Thread(target=run, daemon=True).start()


# ----------------------------------------------------------------- HTTP
class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body, ctype="application/json; charset=utf-8", code=200):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False)
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _static(self, rel, ctype):
        p = os.path.join(HERE, "web", rel)
        if not os.path.exists(p):
            self._send("not found", "text/plain", 404); return
        self._send(open(p, "rb").read(), ctype)

    def _chat_stream(self, sid, message):
        """RECALL -> prompt -> LLM'i STREAM'le; bitince belleğe yaz + otonom tetikle."""
        cfg = load_config()
        hops = int(cfg.get("hops", 4))
        season = load_season(sid)
        aha = season.get("pending_aha")
        with _lock:
            recall = MEM.recall(message, hops=hops, k=8)
        system = build_prompt(season, recall, aha, cfg.get("lang", "en"))
        messages = history_messages(season) + [{"role": "user", "content": message}]

        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")   # proxy buffering kapat
        self.end_headers()
        full = []
        try:
            for chunk in router.chat_stream(cfg.get("model"), system, messages, cfg.get("api_key")):
                full.append(chunk)
                self.wfile.write(chunk.encode("utf-8"))
                self.wfile.flush()
        except Exception as e:
            try:
                self.wfile.write(f"\n[stream hata] {e}".encode("utf-8"))
            except Exception:
                pass
        answer = "".join(full)

        ts = time.time()
        # SENKRON (ucuz JSON): sezon-history -> sonraki tur ANLIK süreklilik kaybetmesin
        season["history"].append({"q": message, "a": answer, "ts": ts})
        season["pending_aha"] = None
        save_season(season)
        # ARKA PLAN (PARALEL): graf öğrenme (zemberek) + kayıt + otonom -> cevabı/isteği BLOKLAMAZ
        _learn_bg(sid, message, answer, ts, [c["text"] for c in recall.get("chunks", [])], hops)

    def do_GET(self):
        from urllib.parse import urlparse, parse_qs
        path = urlparse(self.path).path
        if path == "/" or path == "/index.html":
            self._static("index.html", "text/html; charset=utf-8")
        elif path == "/app.js":
            self._static("app.js", "application/javascript; charset=utf-8")
        elif path == "/style.css":
            self._static("style.css", "text/css; charset=utf-8")
        elif path == "/seasons":
            self._send(list_seasons())
        elif path == "/season":
            sid = parse_qs(urlparse(self.path).query).get("id", ["default"])[0]
            self._send(load_season(sid))
        elif path == "/settings":
            c = load_config(); c = {**c, "api_key": "***" if c.get("api_key") else ""}
            self._send(c)
        elif path == "/models":
            self._send(router.list_models(load_config().get("api_key")))
        elif path == "/stats":
            self._send(MEM.stats())
        else:
            self._send("not found", "text/plain", 404)

    def do_POST(self):
        ln = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(ln).decode("utf-8") or "{}")
        from urllib.parse import urlparse
        path = urlparse(self.path).path
        if path == "/chat":
            sid = body.get("season", "default")
            msg = (body.get("message") or "").strip()
            if not msg:
                self._send({"error": "boş mesaj"}, code=400); return
            self._chat_stream(sid, msg)
        elif path == "/season/new":
            sid = body.get("id") or f"sezon-{int(time.time())}"
            s = {"id": sid, "name": body.get("name", sid), "history": [],
                 "pending_aha": None, "created": time.time()}
            save_season(s)
            self._send(s)
        elif path == "/season/delete":
            sid = body.get("id")
            if sid:
                p = season_path(sid)
                if os.path.exists(p):
                    os.remove(p)         # NOT: global bellek (öğrenilen) SİLİNMEZ — yalnız sohbet geçmişi
            self._send({"ok": True})
        elif path == "/settings":
            c = load_config()
            for kbar in ("hops", "model", "lang"):
                if kbar in body:
                    c[kbar] = body[kbar]
            if body.get("api_key"):                  # boş/maskeli ise dokunma
                c["api_key"] = body["api_key"]
            save_config(c)
            self._send({"ok": True})
        else:
            self._send("not found", "text/plain", 404)


if __name__ == "__main__":
    os.makedirs(SEASON_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(CKPT), exist_ok=True)
    if not os.path.exists(CONFIG):
        save_config(load_config())
    print(f"[Cognitive Layer] hafıza: {MEM.stats()}  ->  http://localhost:{PORT}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
