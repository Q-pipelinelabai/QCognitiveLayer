# -*- coding: utf-8 -*-
"""
Cognitive Layer — LLM ROUTER
============================
Tek arayüz, çok sağlayıcı. Şimdilik: Gemini (API key) + Ollama (yerel).
Model adı biçimi:
  * "gemini:<model>"   örn. gemini:gemini-2.0-flash   (GEMINI_API_KEY gerekir)
  * "ollama:<model>"   örn. ollama:llama3.1           (yerel ollama çalışıyorsa)

chat(model, system, messages, api_key) -> str
list_models(api_key) -> {"gemini":[...], "ollama":[...]}
"""
import json
import re
import urllib.request
import urllib.error

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
OLLAMA_BASE = "http://127.0.0.1:11434"   # 'localhost' Windows'ta ::1'e (IPv6) gidip boş örneğe düşebiliyor -> IPv4 sabit

# Curated YENİ-nesil fallback (key yoksa/canlı liste alınamazsa). Haziran 2026: 2.0 kapatıldı.
GEMINI_DEFAULTS = [
    "gemini-3.5-flash", "gemini-3.1-flash-lite", "gemini-3.1-pro-preview",
    "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro",
]


def _post(url, payload, headers=None, timeout=120):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _get(url, timeout=8):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


# ----------------------------------------------------------------- Gemini
def _gemini_chat(model, system, messages, api_key):
    if not api_key:
        return "[HATA] Gemini API key girilmemiş (Ayarlar)."
    contents = []
    for m in messages:
        role = "model" if m["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": m["content"]}]})
    payload = {"contents": contents}
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}
    url = f"{GEMINI_BASE}/models/{model}:generateContent?key={api_key}"
    try:
        r = _post(url, payload)
        return r["candidates"][0]["content"]["parts"][0]["text"]
    except urllib.error.HTTPError as e:
        return f"[Gemini HATA {e.code}] {e.read().decode('utf-8', 'ignore')[:300]}"
    except Exception as e:
        return f"[Gemini HATA] {e}"


# ----------------------------------------------------------------- Ollama
def _ollama_chat(model, system, messages):
    msgs = ([{"role": "system", "content": system}] if system else []) + messages
    payload = {"model": model, "messages": msgs, "stream": False}
    try:
        r = _post(f"{OLLAMA_BASE}/api/chat", payload)
        return r["message"]["content"]
    except Exception as e:
        return f"[Ollama HATA] {e} (ollama çalışıyor mu? `ollama serve`)"


def _ollama_info():
    """Döner: (model_listesi, erişilebilir_mi). Ayrım: kapalı vs açık-ama-model-yok."""
    try:
        r = _get(f"{OLLAMA_BASE}/api/tags")
        return [m["name"] for m in r.get("models", [])], True
    except Exception:
        return [], False


# ----------------------------------------------------------------- Gemini canlı model listesi
def _gemini_models(api_key):
    """Google'dan CANLI model listesi (key'in erişebildikleri). Metin-üretim modellerini süzer."""
    if not api_key:
        return None
    try:
        r = _get(f"{GEMINI_BASE}/models?key={api_key}&pageSize=200", timeout=10)
    except Exception:
        return None
    names = []
    for m in r.get("models", []):
        if "generateContent" not in m.get("supportedGenerationMethods", []):
            continue
        names.append(m.get("name", "").split("/")[-1])
    return _clean_gemini(names) or None


_BAD = ("embedding", "aqa", "image", "tts", "audio", "vision", "veo", "imagen", "lyria",
        "clip", "antigravity", "deep-research", "customtools", "learnlm", "computer-use",
        "robotics", "-001", "-002", "exp", "thinking", "-8b", "-latest")


def _g_ver(n):
    m = re.search(r"^(?:gemini|gemma)-(\d+(?:\.\d+)?)", n.lower())
    return float(m.group(1)) if m else 0.0


def _clean_gemini(names):
    """Yalnız SOHBET (gemini/gemma text) modelleri; müzik/görsel/agent/eski-nesil/dated ele; yeni-sürüm önce."""
    out = set()
    for n in names:
        low = n.lower()
        if any(x in low for x in _BAD):
            continue
        if not (low.startswith("gemini-") or low.startswith("gemma-")):
            continue
        if low.startswith("gemini") and 0 < _g_ver(n) < 2.5:      # 2.0/1.5 kapandı
            continue
        out.add(n)
    # aynı modelin "-preview" ikizini, kararlı sürümü varsa ele
    bases = {n for n in out if not n.endswith("-preview")}
    out = [n for n in out if not (n.endswith("-preview") and n[: -len("-preview")] in bases)]
    # önce TÜM gemini (sürüm desc), sonra gemma; aynı sürümde pro→flash→flash-lite
    return sorted(out, key=lambda n: ("gemma" in n, -_g_ver(n), "pro" not in n, "lite" in n, n))


# ----------------------------------------------------------------- genel
def chat(model, system, messages, api_key=None):
    if model.startswith("ollama:"):
        return _ollama_chat(model.split(":", 1)[1], system, messages)
    if model.startswith("gemini:"):
        return _gemini_chat(model.split(":", 1)[1], system, messages, api_key)
    # öntanımlı: gemini
    return _gemini_chat(model, system, messages, api_key)


# ----------------------------------------------------------------- STREAMING
def chat_stream(model, system, messages, api_key=None):
    """Cevabı parça parça (delta) üreten generator. Gemini SSE / Ollama stream."""
    if model.startswith("ollama:"):
        yield from _ollama_stream(model.split(":", 1)[1], system, messages)
        return
    name = model.split(":", 1)[1] if model.startswith("gemini:") else model
    yield from _gemini_stream(name, system, messages, api_key)


def _gemini_stream(model, system, messages, api_key):
    if not api_key:
        yield "[HATA] Gemini API key girilmemiş (Ayarlar)."
        return
    contents = [{"role": "model" if m["role"] == "assistant" else "user",
                 "parts": [{"text": m["content"]}]} for m in messages]
    payload = {"contents": contents}
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}
    url = f"{GEMINI_BASE}/models/{model}:streamGenerateContent?alt=sse&key={api_key}"
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 method="POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            for raw in r:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                chunk = line[5:].strip()
                if not chunk or chunk == "[DONE]":
                    continue
                try:
                    j = json.loads(chunk)
                    for part in j["candidates"][0]["content"].get("parts", []):
                        t = part.get("text", "")
                        if t:
                            yield t
                except Exception:
                    continue
    except urllib.error.HTTPError as e:
        yield f"[Gemini HATA {e.code}] {e.read().decode('utf-8', 'ignore')[:300]}"
    except Exception as e:
        yield f"[Gemini HATA] {e}"


def _ollama_stream(model, system, messages):
    msgs = ([{"role": "system", "content": system}] if system else []) + messages
    payload = {"model": model, "messages": msgs, "stream": True}
    req = urllib.request.Request(f"{OLLAMA_BASE}/api/chat", data=json.dumps(payload).encode("utf-8"),
                                 method="POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            for raw in r:
                line = raw.decode("utf-8", "ignore").strip()
                if not line:
                    continue
                try:
                    j = json.loads(line)
                    t = j.get("message", {}).get("content", "")
                    if t:
                        yield t
                    if j.get("done"):
                        break
                except Exception:
                    continue
    except Exception as e:
        yield f"[Ollama HATA] {e} (ollama çalışıyor mu? `ollama serve`)"


def list_models(api_key=None):
    gem = _gemini_models(api_key) or GEMINI_DEFAULTS     # canlı liste; alınamazsa curated yeni-nesil
    omodels, oup = _ollama_info()
    return {"gemini": [f"gemini:{m}" for m in gem],
            "ollama": [f"ollama:{m}" for m in omodels],
            "ollama_up": oup}
