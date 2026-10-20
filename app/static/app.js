"use strict";

const $ = (id) => document.getElementById(id);
const state = {
  notes: [], note: null, dirty: false, savePromise: null, saveTimer: null,
  chatController: null, chatBusy: false, rewriteController: null,
  rewriteOriginal: null, rewriteNoteId: null, rewriteReady: false,
  navSeq: 0, searchSeq: 0,
};

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  if (!response.ok) {
    let message;
    try { const data = await response.json(); message = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail); }
    catch { message = response.statusText; }
    throw new Error(message || `HTTP ${response.status}`);
  }
  if (response.status === 204) return null;
  return response.json();
}

function notify(message, error = false) {
  const el = $("app-status");
  el.textContent = message;
  el.style.color = error ? "#a33434" : "";
}
function formatDate(date) {
  return new Date(date).toLocaleString("ru-RU", { day:"2-digit", month:"2-digit", hour:"2-digit", minute:"2-digit" });
}
function markDirty() {
  if (!state.note) return;
  state.dirty = true;
  $("save-status").textContent = "Не сохранено";
  clearTimeout(state.saveTimer);
  state.saveTimer = setTimeout(() => { saveCurrent().catch(err => notify(err.message, true)); }, 800);
}

async function saveCurrent() {
  clearTimeout(state.saveTimer);
  if (state.savePromise) await state.savePromise;
  if (!state.note || !state.dirty) return;
  const id = state.note.id;
  const content = $("note-content").value;
  const title = $("note-title").value.trim() || "Без названия";
  state.dirty = false;
  $("save-status").textContent = "Сохранение…";
  const pending = api(`/api/notes/${id}`, {method:"PUT", body:JSON.stringify({title, content})});
  state.savePromise = pending;
  try {
    const saved = await pending;
    if (state.note?.id === id) {
      state.note = saved;
      $("save-status").textContent = state.dirty ? "Не сохранено" : "Сохранено";
      // Do not clobber the user's typing while a save was in flight.
      if (!state.dirty) $("note-title").value = saved.title;
    }
    state.notes = state.notes.map(n => n.id === id ? {id, title:saved.title, updated_at:saved.updated_at} : n);
    renderNotes();
  } catch (err) {
    if (state.note?.id === id) { state.dirty = true; $("save-status").textContent = "Ошибка сохранения"; }
    throw err;
  } finally {
    if (state.savePromise === pending) state.savePromise = null;
  }
  if (state.dirty && state.note?.id === id) return saveCurrent();
}

async function refreshNotes() {
  state.notes = await api("/api/notes");
  if ($("search-input").value.trim()) return showSearchResults();
  renderNotes();
}
function renderNotes(items = state.notes) {
  const list = $("notes-list");
  list.replaceChildren();
  if (!items.length) { const p = document.createElement("p"); p.className = "empty"; p.textContent = "Ничего не найдено"; list.append(p); return; }
  for (const note of items) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "note-item" + (note.id === state.note?.id ? " active" : "");
    const title = document.createElement("span"); title.className = "note-title"; title.textContent = note.title;
    const date = document.createElement("span"); date.className = "note-date";
    date.textContent = note.updated_at ? formatDate(note.updated_at) : (note.snippet || "").slice(0, 65);
    button.append(title, date);
    button.onclick = () => openNote(note.id);
    list.append(button);
  }
}
async function showSearchResults() {
  const query = $("search-input").value.trim();
  const seq = ++state.searchSeq;
  if (!query) return renderNotes();
  try {
    const results = await api(`/api/search?q=${encodeURIComponent(query)}`);
    if (seq === state.searchSeq) renderNotes(results);
  } catch (err) { notify(err.message, true); }
}
function enableEditor(enabled) {
  for (const id of ["note-title","note-content","save-btn","delete-btn","rewrite-btn","related-btn","chat-input","send-btn","clear-chat"]) {
    $(id).disabled = !enabled;
  }
  $("send-btn").disabled = !enabled || state.chatBusy;
}

async function openNote(id) {
  if (state.note?.id === id) return;
  const seq = ++state.navSeq;
  try {
    await saveCurrent();
    if (seq !== state.navSeq) return;
    stopChat();
    closeRewrite();
    $("related-panel").hidden = true;
    const [note, messages] = await Promise.all([
      api(`/api/notes/${id}`), api(`/api/notes/${id}/messages`),
    ]);
    if (seq !== state.navSeq) return;
    state.note = note;
    state.dirty = false;
    $("note-title").value = note.title;
    $("note-content").value = note.content;
    $("save-status").textContent = "Сохранено";
    $("search-input").value = "";
    enableEditor(true);
    renderNotes();
    renderMessages(messages);
  } catch (err) { notify(err.message, true); }
}

async function createNote() {
  try {
    await saveCurrent();
    const note = await api("/api/notes", {method:"POST", body:JSON.stringify({title:"Новая заметка", content:""})});
    await refreshNotes();
    await openNote(note.id);
    $("note-title").focus();
    $("note-title").select();
  } catch (err) { notify(err.message, true); }
}
async function deleteNote() {
  if (!state.note || !window.confirm(`Удалить заметку «${state.note.title}» и её историю чата?`)) return;
  const id = state.note.id;
  clearTimeout(state.saveTimer);
  state.dirty = false;
  stopChat();
  try {
    await api(`/api/notes/${id}`, {method:"DELETE"});
    state.note = null;
    $("note-title").value = "";
    $("note-content").value = "";
    $("save-status").textContent = "—";
    $("chat-messages").replaceChildren();
    enableEditor(false);
    await refreshNotes();
    if (state.notes.length) await openNote(state.notes[0].id);
  } catch (err) { notify(err.message, true); }
}

function addMessage(role, message, sources = []) {
  const list = $("chat-messages");
  if (list.querySelector(".empty")) list.replaceChildren();
  const bubble = document.createElement("div"); bubble.className = "bubble " + role;
  const who = document.createElement("span"); who.className = "who";
  who.textContent = role === "user" ? "Вы" : "Ollama";
  const content = document.createElement("div"); content.className = "content";
  content.textContent = message;
  bubble.append(who, content);
  addSources(bubble, sources);
  list.append(bubble);
  list.scrollTop = list.scrollHeight;
  return { bubble, content };
}
function addSources(bubble, sources) {
  bubble.querySelector(".sources")?.remove();
  if (!sources?.length) return;
  const area = document.createElement("div"); area.className = "sources";
  for (const source of sources) {
    const button = document.createElement("button"); button.type = "button";
    button.textContent = `↗ ${source.title}`;
    button.onclick = () => openNote(source.id);
    area.append(button);
  }
  bubble.append(area);
}
function renderMessages(messages) {
  $("chat-messages").replaceChildren();
  if (!messages.length) {
    const div = document.createElement("div"); div.className = "empty";
    div.textContent = "Задайте вопрос о текущей заметке. При включённом поиске подключатся связанные заметки.";
    $("chat-messages").append(div); return;
  }
  for (const item of messages) addMessage(item.role, item.content, item.sources);
}

// Parse SSE over a POST fetch stream (EventSource supports GET only).
async function readSSE(path, body, signal, onEvent) {
  const response = await fetch(path, {
    method:"POST", headers:{"Content-Type":"application/json","Accept":"text/event-stream"},
    body:JSON.stringify(body), signal,
  });
  if (!response.ok) {
    const details = await response.json().catch(() => ({}));
    throw new Error(typeof details.detail === "string" ? details.detail : `HTTP ${response.status}`);
  }
  if (!response.body) throw new Error("Браузер не поддерживает потоковый ответ");
  const traceId = response.headers.get("X-Request-ID") || "-";
  console.info(`[Notes AI ${traceId}] ${path} HTTP ${response.status}: SSE открыто (ожидаем модель)`);
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "", doneReceived = false;
  const dispatch = (block) => {
    const lines = block.split("\n");
    const name = lines.find(x => x.startsWith("event: "))?.slice(7) || "message";
    const data = lines.filter(x => x.startsWith("data: ")).map(x => x.slice(6)).join("\n");
    if (!data) return;
    const item = JSON.parse(data);
    // Browser DevTools: metadata only, never print note/prompts/model text.
    if (["status", "heartbeat", "first_token", "usage", "done", "error"].includes(name)) {
      console.debug(`[Notes AI ${traceId}] ${name}`, {
        phase: item.phase, elapsed_ms: item.elapsed_ms, idle_ms: item.idle_ms,
        eval_count: item.eval_count, prompt_eval_count: item.prompt_eval_count,
        ok: item.ok, has_error: name === "error",
      });
    }
    onEvent(name, item);
    if (name === "done") doneReceived = true;
    if (name === "error") throw new Error(item.message || "Ошибка генерации");
  };
  while (true) {
    const {done, value} = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, {stream:true});
    buffer = buffer.replace(/\r\n/g, "\n");
    let position;
    while ((position = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, position);
      buffer = buffer.slice(position + 2);
      dispatch(block);
    }
  }
  if (buffer.trim()) dispatch(buffer);
  if (!doneReceived) {
    console.warn(`[Notes AI ${traceId}] Поток завершился без SSE done`);
    throw new Error(`Поток Ollama прерван до завершения ответа (запрос ${traceId})`);
  }
  console.info(`[Notes AI ${traceId}] Генерация завершена`);
}
// A model stream consists of text *fragments*, not guaranteed tokenizer tokens.
// Usage.eval_count from Ollama's final NDJSON record is the accurate token count.
function makeActivity(container) {
  container.replaceChildren();
  const root = document.createElement("details");
  root.className = "generation-activity running";
  root.open = true;
  const summary = document.createElement("summary");
  const dot = document.createElement("span"); dot.className = "activity-dot";
  const title = document.createElement("span"); title.className = "activity-title"; title.textContent = "Подготовка…";
  const timer = document.createElement("span"); timer.className = "activity-timer"; timer.textContent = "0.0 с";
  const metrics = document.createElement("span"); metrics.className = "activity-metrics"; metrics.textContent = "0 фрагментов";
  const toggle = document.createElement("span"); toggle.className = "activity-toggle"; toggle.textContent = "Журнал ▾";
  summary.append(dot, title, timer, metrics, toggle);
  const log = document.createElement("div"); log.className = "activity-log";
  root.append(summary, log); container.append(root);

  const since = performance.now();
  let chunks = 0, chars = 0, thinking = 0, exactTokens = null, ended = false;
  let lastProgressLog = since;
  let lastHeartbeatLog = 0, lastHeartbeatPhase = "", traceShown = false;
  const showTime = (milliseconds) => `${(milliseconds / 1000).toFixed(1)} с`;
  const tick = () => { if (!ended) timer.textContent = showTime(performance.now() - since); };
  const interval = setInterval(tick, 100);
  function addLine(message, error = false) {
    const line = document.createElement("div"); line.className = "activity-line" + (error ? " error" : "");
    const when = document.createElement("span"); when.className = "at";
    when.textContent = showTime(performance.now() - since);
    const text = document.createElement("span"); text.className = "msg"; text.textContent = message;
    line.append(when, text); log.append(line);
    while (log.childElementCount > 80) log.firstElementChild.remove();
    log.scrollTop = log.scrollHeight;
  }
  function updateMetrics() {
    const amount = exactTokens === null ? `${chunks} фрагм.` : `${exactTokens} токенов`;
    metrics.textContent = `${amount} · ${chars} симв.` + (thinking ? ` · thinking ${thinking}` : "");
  }
  addLine("Запрос готовится к отправке");
  return {
    handle(name, data) {
      if (name === "status") {
        if (data.request_id && !traceShown) {
          addLine(`ID запроса: ${data.request_id}`);
          traceShown = true;
        }
        title.textContent = data.message;
        addLine(data.message);
      } else if (name === "heartbeat") {
        const phases = {
          connecting: "Ожидание HTTP-ответа Ollama (возможно, загрузка модели)",
          waiting_first_token: "Соединение есть, ожидаем первый фрагмент модели",
          thinking: "Модель обдумывает ответ, пока нет новых данных",
          generating: "Ожидаем следующую порцию генерации",
        };
        const message = phases[data.phase] || "Ожидание данных от Ollama";
        title.textContent = message;
        const now = performance.now();
        if (data.phase !== lastHeartbeatPhase || now - lastHeartbeatLog >= 10000) {
          addLine(`${message} · пауза ${(data.idle_ms / 1000).toFixed(0)} с`);
          lastHeartbeatLog = now;
          lastHeartbeatPhase = data.phase;
        }
      } else if (name === "token") {
        chunks++; chars += data.text.length;
        updateMetrics();
      } else if (name === "thinking") {
        thinking++; updateMetrics();
      } else if (name === "progress") {
        chunks = data.chunks ?? chunks;
        thinking = data.thinking_chunks ?? thinking;
        chars = data.characters ?? chars;
        updateMetrics();
        if (chunks && performance.now() - lastProgressLog > 1300) {
          addLine(`Получено ${chunks} фрагментов · ${chars} символов (точные токены — после ответа)`);
          lastProgressLog = performance.now();
        }
      } else if (name === "first_token") {
        addLine(`Первый фрагмент ответа через ${(data.after_ms / 1000).toFixed(2)} с после отправки`);
      } else if (name === "sources") {
        addLine(`Контекст: ${data.sources?.length || 0} связанных заметок`);
      } else if (name === "usage") {
        if (typeof data.eval_count === "number") exactTokens = data.eval_count;
        updateMetrics();
        const parts = [];
        if (data.prompt_eval_count != null) parts.push(`вход: ${data.prompt_eval_count} токенов`);
        if (data.eval_count != null) parts.push(`выход: ${data.eval_count} токенов`);
        if (data.tokens_per_second != null) parts.push(`${data.tokens_per_second} ток./с`);
        if (parts.length) addLine(parts.join(" · "));
        if (data.load_duration != null) addLine(`Загрузка модели: ${(data.load_duration / 1e9).toFixed(2)} с`);
        if (data.prompt_eval_duration != null) addLine(`Обработка промпта: ${(data.prompt_eval_duration / 1e9).toFixed(2)} с`);
        if (data.eval_duration != null) addLine(`Генерация на сервере: ${(data.eval_duration / 1e9).toFixed(2)} с`);
        if (data.done_reason) addLine(`Причина завершения: ${data.done_reason}`);
      }
    },
    finish(message, failed = false) {
      if (ended) return;
      ended = true;
      clearInterval(interval);
      timer.textContent = showTime(performance.now() - since);
      title.textContent = message;
      root.classList.remove("running");
      root.classList.add(failed ? "failed" : "completed");
      addLine(message, failed);
      if (!failed) root.open = false;
    },
  };
}

// Animation is deliberately decoupled from fetch: a burst of network chunks
// becomes a smooth stream of text, without delaying backend consumption.
function smoothWriter(append) {
  let pending = "", frame = 0;
  const waiters = [];
  function paint() {
    frame = 0;
    if (!pending.length) {
      while (waiters.length) waiters.shift()();
      return;
    }
    const take = Math.min(160, Math.max(3, Math.ceil(pending.length / 8)));
    append(pending.slice(0, take));
    pending = pending.slice(take);
    if (pending.length) frame = requestAnimationFrame(paint);
    else while (waiters.length) waiters.shift()();
  }
  return {
    push(text) { pending += text; if (!frame) frame = requestAnimationFrame(paint); },
    finish() { return pending.length ? new Promise(resolve => waiters.push(resolve)) : Promise.resolve(); },
  };
}

function createThinking(container) {
  const details = document.createElement("details");
  details.className = "thinking-block";
  details.open = true;
  const summary = document.createElement("summary"); summary.textContent = "Thinking модели (поток)";
  const output = document.createElement("div"); output.className = "thinking-content";
  const node = document.createTextNode(""); output.append(node);
  details.append(summary, output);
  container.before(details);
  const writer = smoothWriter(text => { node.appendData(text); output.scrollTop = output.scrollHeight; });
  return {writer, finish: async () => { await writer.finish(); details.open = false; summary.textContent = "Thinking модели"; }};
}

function stopChat() { state.chatController?.abort(); }
function setChatBusy(isBusy) {
  state.chatBusy = isBusy;
  $("send-btn").disabled = isBusy || !state.note;
  $("chat-input").disabled = isBusy || !state.note;
  $("stop-btn").hidden = !isBusy;
}
async function sendChat(event) {
  event.preventDefault();
  if (!state.note || state.chatBusy) return;
  const message = $("chat-input").value.trim();
  if (!message) return;
  try { await saveCurrent(); } catch (err) { notify(err.message, true); return; }
  const noteId = state.note.id;
  const controller = new AbortController();
  state.chatController = controller;
  $("chat-input").value = "";
  addMessage("user", message);
  const reply = addMessage("assistant", "");
  reply.bubble.classList.add("streaming");
  const contentNode = document.createTextNode("");
  reply.content.replaceChildren(contentNode);
  const activityArea = document.createElement("div"); reply.bubble.append(activityArea);
  const activity = makeActivity(activityArea);
  const writer = smoothWriter(text => {
    const list = $("chat-messages");
    const atBottom = list.scrollHeight - list.scrollTop - list.clientHeight < 85;
    contentNode.appendData(text);
    if (atBottom) list.scrollTop = list.scrollHeight;
  });
  let thinking = null, succeeded = false, stopped = false;
  setChatBusy(true);
  try {
    await readSSE(`/api/notes/${noteId}/chat/stream`, {message, ignore_history: $("chat-no-history").checked}, controller.signal, (name, data) => {
      activity.handle(name, data);
      if (name === "token") writer.push(data.text);
      else if (name === "thinking") {
        if (!thinking) thinking = createThinking(reply.content);
        thinking.writer.push(data.text);
      } else if (name === "sources") addSources(reply.bubble, data.sources);
    });
    succeeded = true;
  } catch (err) {
    stopped = err.name === "AbortError";
    if (!stopped) {
      notify(err.message, true);
      activity.finish(`Ошибка: ${err.message}`, true);
      if (!$('chat-input').value.trim()) $('chat-input').value = message;
    }
  } finally {
    await writer.finish();
    if (thinking) await thinking.finish();
    if (stopped) activity.finish("Остановлено пользователем", true);
    else if (succeeded) activity.finish("Ответ завершён");
    reply.bubble.classList.remove("streaming");
    if (state.chatController === controller) state.chatController = null;
    setChatBusy(false);
  }
}
async function clearChat() {
  if (!state.note || !window.confirm("Очистить историю чата для этой заметки?")) return;
  stopChat();
  try {
    await api(`/api/notes/${state.note.id}/messages`, {method:"DELETE"});
    renderMessages([]);
  } catch (err) { notify(err.message, true); }
}

async function showRelated() {
  if (!state.note) return;
  const panel = $("related-panel");
  if (!panel.hidden) { panel.hidden = true; return; }
  panel.hidden = false;
  $("related-items").textContent = "Поиск…";
  try {
    await saveCurrent();
    const noteId = state.note.id;
    const items = await api(`/api/notes/${noteId}/related`);
    if (state.note?.id !== noteId) return;
    const area = $("related-items"); area.replaceChildren();
    if (!items.length) { area.textContent = "Связанных заметок пока нет. Добавьте другие заметки или настройте поиск."; return; }
    for (const item of items) {
      const button = document.createElement("button");
      button.textContent = item.title;
      button.title = item.snippet;
      button.onclick = () => openNote(item.id);
      area.append(button);
    }
  } catch (err) { $("related-items").textContent = err.message; }
}

function closeRewrite() {
  state.rewriteController?.abort();
  state.rewriteController = null;
  $("rewrite-modal").hidden = true;
}
async function openRewrite() {
  if (!state.note) return;
  try { await saveCurrent(); } catch (err) { notify(err.message, true); return; }
  state.rewriteOriginal = state.note.content;
  state.rewriteNoteId = state.note.id;
  state.rewriteReady = false;
  $("rewrite-preview").value = "";
  $("rewrite-activity").hidden = true;
  $("rewrite-apply").disabled = true;
  $("rewrite-run").disabled = false;
  $("rewrite-status").textContent = "Изменения не применяются автоматически.";
  $("rewrite-modal").hidden = false;
}
async function runRewrite() {
  if (!state.rewriteNoteId) return;
  state.rewriteController?.abort();
  const controller = new AbortController();
  state.rewriteController = controller;
  state.rewriteReady = false;
  $("rewrite-preview").value = "";
  $("rewrite-run").disabled = true;
  $("rewrite-apply").disabled = true;
  $("rewrite-status").textContent = "Генерация…";
  $("rewrite-activity").hidden = false;
  const activity = makeActivity($("rewrite-activity"));
  const writer = smoothWriter(text => { $("rewrite-preview").value += text; });
  let succeeded = false, stopped = false;
  try {
    await readSSE(`/api/notes/${state.rewriteNoteId}/rewrite/stream`,
      {instruction:$("rewrite-instruction").value}, controller.signal, (name, data) => {
        activity.handle(name, data);
        if (name === "token") writer.push(data.text);
      });
    succeeded = true;
  } catch (err) {
    stopped = err.name === "AbortError";
    if (!stopped) {
      $("rewrite-status").textContent = err.message;
      activity.finish(`Ошибка: ${err.message}`, true);
    }
  } finally {
    await writer.finish();
    if (state.rewriteController === controller) state.rewriteController = null;
    $("rewrite-run").disabled = false;
    if (stopped) { activity.finish("Остановлено пользователем", true); $("rewrite-status").textContent = "Генерация остановлена"; }
    else if (succeeded) {
      activity.finish("Редактирование завершено");
      state.rewriteReady = Boolean($("rewrite-preview").value.trim());
      $("rewrite-apply").disabled = !state.rewriteReady;
      $("rewrite-status").textContent = state.rewriteReady ? "Проверьте текст перед применением." : "Модель вернула пустой текст.";
    }
  }
}
async function applyRewrite() {
  if (!state.rewriteReady || state.note?.id !== state.rewriteNoteId ||
      $("note-content").value !== state.rewriteOriginal) {
    notify("Заметка изменилась после запуска AI-редактора. Повтори редактирование.", true);
    return;
  }
  $("note-content").value = $("rewrite-preview").value;
  markDirty();
  closeRewrite();
  try { await saveCurrent(); } catch (err) { notify(err.message, true); }
}

const SETTING_KEYS = ["ollama_url","chat_model","embedding_model","temperature","top_p","num_ctx","num_predict","read_timeout_seconds","think_mode","system_prompt","search_mode","context_chunks","history_messages"];
let currentSettings = null;
function readSettingsForm() {
  const form = $("settings-form");
  const result = {};
  for (const key of SETTING_KEYS) result[key] = form.elements.namedItem(key).value;
  for (const key of ["temperature","top_p","num_ctx","num_predict","read_timeout_seconds","context_chunks","history_messages"]) result[key] = Number(result[key]);
  return result;
}
function updateModelGuidance() {
  const model = $("settings-form").elements.namedItem("chat_model").value.toLowerCase();
  $("model-quality-warning").hidden = !model.startsWith("tinyllama");
}
function populateModelSelect(name, models, selected) {
  const select = $("settings-form").elements.namedItem(name);
  select.replaceChildren();
  const empty = new Option(name === "chat_model" ? "Выбрать модель…" : "Без embeddings", "");
  select.append(empty);
  if (selected && !models.includes(selected)) select.add(new Option(`${selected} (нет на сервере)`, selected));
  for (const model of models) select.add(new Option(model, model));
  select.value = selected || "";
  if (name === "chat_model") updateModelGuidance();
}
async function loadSettings() {
  currentSettings = await api("/api/settings");
  const form = $("settings-form");
  for (const [key, value] of Object.entries(currentSettings)) {
    const input = form.elements.namedItem(key);
    if (input && key !== "chat_model" && key !== "embedding_model") input.value = value;
  }
  populateModelSelect("chat_model", [], currentSettings.chat_model);
  populateModelSelect("embedding_model", [], currentSettings.embedding_model);
  const thinkSelect = $("think-mode");
  if (currentSettings.think_mode && !Array.from(thinkSelect.options).some(o => o.value === currentSettings.think_mode)) {
    thinkSelect.add(new Option(`Уровень: ${currentSettings.think_mode}`, currentSettings.think_mode));
  }
  thinkSelect.value = currentSettings.think_mode || "off";
  await refreshModels();
}
async function saveSettings() {
  const settings = readSettingsForm();
  const saved = await api("/api/settings", {method:"PUT", body:JSON.stringify(settings)});
  currentSettings = saved;
  notify("Настройки сохранены");
  await refreshIndex();
}
async function refreshModels() {
  const label = $("ollama-status");
  label.textContent = "Запрашиваем список моделей…";
  const form = $("settings-form");
  // Probe the URL entered in the form, even before saving the other settings.
  const requestedUrl = form.elements.namedItem("ollama_url").value.trim();
  try {
    if (requestedUrl !== currentSettings?.ollama_url) {
      // Persist the complete form to make the server-side Ollama URL explicit.
      await saveSettings();
    }
    const data = await api("/api/ollama/models");
    for (const field of ["chat_model","embedding_model"]) {
      const sel = form.elements.namedItem(field);
      populateModelSelect(field, data.models, sel.value);
    }
    label.textContent = `Соединение установлено · моделей: ${data.models.length}`;
    await refreshThinkingOptions();
  } catch (err) { label.textContent = `Нет подключения: ${err.message}`; }
}
async function refreshThinkingOptions() {
  const select = $("think-mode");
  const model = $("settings-form").elements.namedItem("chat_model").value;
  const previous = select.value || currentSettings?.think_mode || "off";
  const known = ["default", "off", "on"];
  let values = null, defaults = null, infoWarning = null;
  if (model) {
    try {
      const info = await api(`/api/ollama/thinking?model=${encodeURIComponent(model)}`);
      values = info.values;
      defaults = info.default;
      infoWarning = info.warning;
    } catch (err) { infoWarning = `Метаданные thinking недоступны: ${err.message}`; }
  }
  const options = values === null ? known : ["default", ...values.map(v => v === true ? "on" : v === false ? "off" : String(v))];
  if (!options.includes(previous)) options.push(previous);
  select.replaceChildren();
  for (const option of options) {
    const label = ({default:"По умолчанию модели",off:"Выключено",on:"Включено"})[option] || `Уровень: ${option}`;
    select.add(new Option(label + (values !== null && !["default",...values.map(v => v === true ? "on" : v === false ? "off" : String(v))].includes(option) ? " (может не поддерживаться)" : ""), option));
  }
  select.value = previous;
  $("thinking-hint").textContent = values === null
    ? (typeof infoWarning === "string" ? infoWarning : "Ollama не сообщает возможности thinking; результат зависит от модели.")
    : `Поддерживается: ${values.map(v => String(v)).join(", ") || "нет"}${defaults === null ? "" : ` · по умолчанию: ${defaults}`}`;
}

async function checkOllamaDiagnostics() {
  const field = $("ollama-diagnostics");
  field.textContent = "Проверка Ollama (/api/version, /api/ps)…";
  try {
    const data = await api("/api/ollama/diagnostics");
    if (!data.reachable) {
      field.textContent = "Ollama недоступна. " + (data.warnings || []).join("; ");
      return;
    }
    const loaded = data.loaded_models?.length ? data.loaded_models.join(", ") : "нет загруженных моделей";
    const selected = data.selected_model
      ? (data.model_loaded ? " · выбранная модель загружена" : " · выбранная модель НЕ загружена в память") : "";
    field.textContent = `Ollama ${data.version || "версия неизвестна"} · ${loaded}${selected}`
      + (data.warnings?.length ? ` · ${data.warnings.join("; ")}` : "");
  } catch (err) { field.textContent = `Ошибка диагностики: ${err.message}`; }
}

async function refreshIndex() {
  try {
    const info = await api("/api/index/status");
    $("index-status").textContent = info.embedding_model
      ? `${info.ready}/${info.total} фрагментов · осталось ${info.pending}${info.indexing ? " · индексация…" : ""}`
      : "Embedding-модель не выбрана. FTS5 работает независимо.";
  } catch (err) { $("index-status").textContent = err.message; }
}
async function rebuildIndex() {
  try {
    await saveSettings();
    await api("/api/index/rebuild", {method:"POST"});
    notify("Индексация запущена");
    await refreshIndex();
  } catch (err) { notify(err.message, true); }
}

$("new-note").onclick = createNote;
$("save-btn").onclick = () => saveCurrent().catch(err => notify(err.message, true));
$("delete-btn").onclick = deleteNote;
$("related-btn").onclick = showRelated;
$("related-close").onclick = () => $("related-panel").hidden = true;
$("rewrite-btn").onclick = openRewrite;
$("note-content").addEventListener("input", markDirty);
$("note-title").addEventListener("input", markDirty);
$("search-input").addEventListener("input", showSearchResults);
$("chat-form").addEventListener("submit", sendChat);
$("chat-input").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); $("chat-form").requestSubmit(); }
});
$("stop-btn").onclick = stopChat;
$("clear-chat").onclick = clearChat;
$("rewrite-close").onclick = closeRewrite;
$("rewrite-cancel").onclick = closeRewrite;
$("rewrite-run").onclick = runRewrite;
$("rewrite-apply").onclick = applyRewrite;
$("settings-form").addEventListener("submit", (event) => { event.preventDefault(); saveSettings().catch(err => notify(err.message, true)); });
$("load-models").onclick = refreshModels;
$("ollama-diagnostics-btn").onclick = checkOllamaDiagnostics;
$("settings-form").elements.namedItem("chat_model").addEventListener("change", () => {
  updateModelGuidance();
  refreshThinkingOptions();
});
$("refresh-index").onclick = refreshIndex;
$("rebuild-index").onclick = rebuildIndex;
document.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") {
    event.preventDefault(); saveCurrent().catch(err => notify(err.message, true));
  }
  if (event.key === "Escape" && !$("rewrite-modal").hidden) closeRewrite();
});

async function boot() {
  try {
    await refreshNotes();
    if (state.notes.length) await openNote(state.notes[0].id);
    notify("Готово");
    // A missing Ollama server must not prevent the local notes UI from loading.
    loadSettings().catch(err => $("ollama-status").textContent = err.message);
    refreshIndex();
    // Avoid incessant /api/index/status polling when indexing is inactive.
    setInterval(() => { if (currentSettings?.embedding_model) refreshIndex(); }, 60000);
  } catch (err) { notify(err.message, true); }
}
boot();
