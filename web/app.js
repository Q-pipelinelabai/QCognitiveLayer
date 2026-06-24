// Qcognitive Layer — chat UI (i18n: en default / tr)
let SEASON = "default";
const $ = id => document.getElementById(id);

const I18N = {
  en: {
    newChat: "+ New chat", send: "Send", settings: "Settings", language: "Language",
    hops: "Active neuron depth (hops)", model: "LLM model", apiKey: "Gemini API key",
    save: "Save", ollamaNote: "If Ollama is running locally, its models are listed automatically.",
    composer: "Type a message…", thinking: "thinking…",
    noMsg: "No messages in this chat yet.\nAsk something — memory recalls from past chats.",
    chatPrefix: "Chat ", emptyAnswer: "(empty answer)", error: "[error] ", delTitle: "Delete chat",
    ollamaEmpty: "no model — `ollama pull llama3.1`", ollamaOff: "not running — `ollama serve`",
    ollamaGroup: "Ollama (local)",
    mem: (c, r) => `memory: ${c} concepts · ${r} records`, msgs: n => `${n} messages`,
  },
  tr: {
    newChat: "+ Yeni sohbet", send: "Gönder", settings: "Ayarlar", language: "Dil",
    hops: "Aktif nöron derinliği (hop)", model: "LLM modeli", apiKey: "Gemini API key",
    save: "Kaydet", ollamaNote: "Ollama yereldeyse modelleri otomatik listelenir.",
    composer: "Mesaj yaz…", thinking: "düşünüyor…",
    noMsg: "Bu sohbette henüz mesaj yok.\nBir şey sor — hafıza geçmiş sohbetlerden hatırlar.",
    chatPrefix: "Sohbet ", emptyAnswer: "(boş cevap)", error: "[hata] ", delTitle: "Sohbeti sil",
    ollamaEmpty: "model yok — `ollama pull llama3.1`", ollamaOff: "çalışmıyor — `ollama serve`",
    ollamaGroup: "Ollama (yerel)",
    mem: (c, r) => `bellek: ${c} kavram · ${r} kayıt`, msgs: n => `${n} mesaj`,
  },
};
let LANG = localStorage.getItem("qclang") || "en";
const T = () => I18N[LANG];
function applyLang() {
  document.documentElement.lang = LANG;
  document.querySelectorAll("[data-i18n]").forEach(el => {
    const v = T()[el.dataset.i18n]; if (v) el.textContent = v;
  });
  document.querySelectorAll("[data-i18n-ph]").forEach(el => {
    const v = T()[el.dataset.i18nPh]; if (v) el.placeholder = v;
  });
}

async function jget(u) { return (await fetch(u)).json(); }
async function jpost(u, b) { return (await fetch(u, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(b) })).json(); }

// ---------------- chats (seasons)
async function loadSeasons() {
  let list = await jget("/seasons");
  if (!list.length) { await jpost("/season/new", { id: "default", name: "Chat 1" }); list = await jget("/seasons"); }
  const el = $("seasonList"); el.innerHTML = "";
  list.forEach(s => {
    const d = document.createElement("div");
    d.className = "season" + (s.id === SEASON ? " active" : "");
    d.innerHTML = `<span class="s-name">${s.name}<small>${T().msgs(s.turns)}</small></span>` +
      `<button class="s-del" title="${T().delTitle}">✕</button>`;
    d.querySelector(".s-name").onclick = () => selectSeason(s.id);
    d.querySelector(".s-del").onclick = (e) => { e.stopPropagation(); delSeason(s.id); };
    el.appendChild(d);
  });
}
async function selectSeason(id) {
  SEASON = id;
  const s = await jget("/season?id=" + encodeURIComponent(id));
  $("seasonTitle").textContent = s.name || id;
  renderHistory(s.history || []);
  loadSeasons();
}
async function newSeason() {
  const name = T().chatPrefix + new Date().toLocaleString(LANG, { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" });
  try {
    const s = await jpost("/season/new", { name });
    await loadSeasons(); await selectSeason(s.id);
  } catch (e) { alert("Could not create chat: " + e); }
}
async function delSeason(id) {
  await jpost("/season/delete", { id });
  const list = await jget("/seasons");
  if (!list.length) { await newSeason(); return; }
  if (id === SEASON) await selectSeason(list[0].id); else loadSeasons();
}

// ---------------- chat
function renderHistory(h) {
  const c = $("chat"); c.innerHTML = "";
  if (!h.length) { c.innerHTML = `<div class="empty">${T().noMsg.replace(/\n/g, "<br>")}</div>`; return; }
  h.forEach(t => { addMsg("user", t.q); addMsg("bot", t.a); });
  c.scrollTop = c.scrollHeight;
}
function addMsg(who, text) {
  const c = $("chat");
  const d = document.createElement("div");
  d.className = "msg " + (who === "user" ? "user" : "bot");
  d.innerHTML = `<div class="who">${who === "user" ? "U" : "Q"}</div><div class="body"></div>`;
  d.querySelector(".body").textContent = text;
  c.appendChild(d); c.scrollTop = c.scrollHeight;
  return d;
}
async function refreshStats() {
  try { const s = await jget("/stats"); $("memStats").textContent = T().mem(s.kelime, s["öbek"]); } catch (e) {}
}
async function send() {
  const inp = $("input"); const msg = inp.value.trim();
  if (!msg) return;
  inp.value = ""; inp.style.height = "auto";
  addMsg("user", msg);
  $("send").disabled = true;
  const bot = addMsg("bot", ""); const body = bot.querySelector(".body");
  body.innerHTML = `<span class="typing">${T().thinking}</span>`;
  try {
    const resp = await fetch("/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ season: SEASON, message: msg }) });
    const reader = resp.body.getReader(); const dec = new TextDecoder();
    let acc = "", first = true;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      acc += dec.decode(value, { stream: true });
      if (first) { body.textContent = ""; first = false; }
      body.textContent = acc;
      $("chat").scrollTop = $("chat").scrollHeight;
    }
    if (!acc) body.textContent = T().emptyAnswer;
    refreshStats(); loadSeasons();
  } catch (e) { body.textContent = T().error + e; }
  $("send").disabled = false; inp.focus();
}

// ---------------- settings
async function openSettings() {
  const cfg = await jget("/settings");
  const models = await jget("/models");
  const sel = $("model"); sel.innerHTML = "";
  const add = (label, arr) => { if (!arr.length) return; const og = document.createElement("optgroup"); og.label = label;
    arr.forEach(m => { const o = document.createElement("option"); o.value = m; o.textContent = m.replace(/^(gemini|ollama):/, ""); og.appendChild(o); }); sel.appendChild(og); };
  add("Gemini", models.gemini || []);
  if (models.ollama && models.ollama.length) { add(T().ollamaGroup, models.ollama); }
  else {
    const og = document.createElement("optgroup"); og.label = T().ollamaGroup;
    const o = document.createElement("option"); o.disabled = true;
    o.textContent = models.ollama_up ? T().ollamaEmpty : T().ollamaOff;
    og.appendChild(o); sel.appendChild(og);
  }
  sel.value = cfg.model || (models.gemini[0] || "");
  $("hops").value = cfg.hops || 3; $("hopVal").textContent = cfg.hops || 3;
  $("lang").value = LANG;
  $("apiKey").value = ""; $("apiKey").placeholder = cfg.api_key ? "•••• (saved)" : "(Gemini only)";
  $("settingsOverlay").classList.remove("hidden");
}
async function saveSettings() {
  const b = { hops: parseInt($("hops").value), model: $("model").value, lang: $("lang").value };
  const ak = $("apiKey").value.trim(); if (ak) b.api_key = ak;
  await jpost("/settings", b);
  if ($("lang").value !== LANG) { LANG = $("lang").value; localStorage.setItem("qclang", LANG); applyLang(); }
  $("settingsOverlay").classList.add("hidden");
  loadSeasons();
}

// ---------------- bind
$("send").onclick = send;
$("input").addEventListener("keydown", e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } });
$("input").addEventListener("input", e => { e.target.style.height = "auto"; e.target.style.height = Math.min(200, e.target.scrollHeight) + "px"; });
$("newSeason").onclick = newSeason;
$("settingsBtn").onclick = openSettings;
$("settingsClose").onclick = () => $("settingsOverlay").classList.add("hidden");
$("saveSettings").onclick = saveSettings;
$("hops").addEventListener("input", e => $("hopVal").textContent = e.target.value);

(async () => { applyLang(); await loadSeasons(); await selectSeason("default"); })();
