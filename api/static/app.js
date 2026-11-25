// VidyaRAG Tutor page. Talks to the same-origin FastAPI endpoints:
//   POST /search   the lecture moments for a question
//   POST /ask      a cited answer from the top moments (Server-Sent Events);
//                  its [n] citations play moment n in the player
//   POST /rewrite  turns a follow-up ("what is its worst case?") into a
//                  standalone question, using the recent turns
// The conversation lives here in the page; the server keeps no state.
// All text (results and the LLM's answer) is inserted with textContent / text
// nodes, never innerHTML, so neither transcripts nor model output can inject markup.
"use strict";

(() => {
  const $ = (sel) => document.querySelector(sel);
  const form = $("#search-form");
  const input = $("#q");
  const button = $("#search-btn");
  const settings = $("#settings");
  const searchUnder = $(".search-under");
  const sessionActions = $("#session-actions");
  const topk = $("#topk");
  const useBm25 = $("#opt-bm25");
  const useRerank = $("#opt-rerank");
  const session = $("#session");
  const statusEl = $("#status");
  const shareBtn = $("#share-btn");
  const newBtn = $("#new-btn");
  const sideEl = $(".side");
  const threadEl = $("#thread");
  const followForm = $("#followup-form");
  const followInput = $("#fq");
  const followBtn = $("#fq-btn");
  const frameEl = $("#player-frame");
  const npKicker = $("#np-kicker");
  const npTitle = $("#np-title");
  const npSub = $("#np-sub");
  const npYt = $("#np-yt");
  const playerClose = $("#player-close");
  const tabTranscript = $("#tab-transcript");
  const tabSlide = $("#tab-slide");
  const panelTranscript = $("#panel-transcript");
  const panelSlide = $("#panel-slide");

  // /info says whether this instance has an LLM key (answers and follow-ups).
  const infoReady = fetch("/info").then((r) => (r.ok ? r.json() : null)).catch(() => null);
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const HISTORY_TURNS = 3;   // the server only uses the last 3 turns anyway

  const STOPWORDS = new Set((
    "the and but for with from into about how what why when which who does did are was were " +
    "this that these those can will would should explain define difference between using use its"
  ).split(" "));

  let thread = [];         // turns of the current conversation, oldest first
  let active = null;       // { turn, index } shown in the player
  let playing = null;      // { turn, index } whose video is loaded
  let inflight = null;     // AbortController of the turn in progress
  let busy = false;
  let answersEnabled = false;
  infoReady.then((info) => {
    answersEnabled = !!(info && info.followups_enabled);
    // Without answers there is nothing to follow up on: keep the top search box.
    document.body.dataset.answers = answersEnabled ? "on" : "off";
    updateComposer();
  });

  // ── helpers ──────────────────────────────────────────────────────────────
  function el(tag, attrs = {}, ...children) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs)) {
      if (value === null || value === undefined || value === false) continue;
      if (key === "class") node.className = value;
      else if (key === "text") node.textContent = value;
      else node.setAttribute(key, value === true ? "" : String(value));
    }
    for (const child of children) if (child !== null && child !== undefined) node.append(child);
    return node;
  }

  const SVG_NS = "http://www.w3.org/2000/svg";
  function icon(shapes, size = 15) {
    const svg = document.createElementNS(SVG_NS, "svg");
    for (const [k, v] of Object.entries({ width: size, height: size, viewBox: "0 0 24 24", "aria-hidden": "true" })) {
      svg.setAttribute(k, String(v));
    }
    for (const [tag, attrs] of shapes) {
      const shape = document.createElementNS(SVG_NS, tag);
      for (const [k, v] of Object.entries(attrs)) shape.setAttribute(k, String(v));
      svg.append(shape);
    }
    return svg;
  }
  const COPY_ICON = [["rect", { x: 9, y: 9, width: 11, height: 11, rx: 2 }], ["path", { d: "M5 15V6a2 2 0 0 1 2-2h8" }]];

  function fmtTime(seconds) {
    const total = Math.max(0, Math.floor(Number(seconds) || 0));
    const h = Math.floor(total / 3600);
    const m = Math.floor((total % 3600) / 60);
    const s = String(total % 60).padStart(2, "0");
    return h ? `${h}:${String(m).padStart(2, "0")}:${s}` : `${m}:${s}`;
  }

  function videoId(url) {
    const match = /(?:[?&]v=|youtu\.be\/)([\w-]{11})/.exec(url || "");
    return match ? match[1] : null;
  }

  // Lecture titles carry their series name and numbering in several styles
  // ("Lecture 20:…", "#12 … | Introduction to Operating Systems", "Deep
  // Learning(CS7015): Lec 1.1 …"). Show the topic; the number is shown separately.
  function cleanTitle(raw) {
    let title = String(raw || "").trim();
    const parts = title.split(/\s+\|\s+/);
    if (parts.length > 1) title = parts.slice(0, -1).join(" · ");
    title = title
      .replace(/^Deep Learning\s*\(CS\d+\):\s*/i, "")
      .replace(/^#\d+\s+/, "")
      .replace(/^Lec(?:ture)?\s*[\d.]+\s*[:.–-]\s*/i, "")
      .replace(/^Lec\s*[\d.]+\s+/i, "");
    return title || String(raw || "") || "Untitled lecture";
  }

  // The lecture's own number, as its title states it ("Lecture 20", "#12",
  // "Lec 1.1"). The index's lecture_number is a playlist position, which can
  // differ from the course's numbering, so it isn't shown.
  function lectureNumber(raw) {
    const m = /^(?:Deep Learning\s*\(CS\d+\):\s*)?(?:#|Lec(?:ture)?\s*)(\d+(?:\.\d+)?)/i.exec(String(raw || "").trim());
    return m ? `Lecture ${m[1].replace(/^0+(?=\d)/, "")}` : null;
  }

  function queryTerms(query) {
    const words = query.toLowerCase().match(/[a-z0-9]{3,}/g) || [];
    return [...new Set(words)].filter((w) => !STOPWORDS.has(w)).slice(0, 12);
  }

  // Wrap occurrences of the query terms in <mark>, building text nodes only.
  function highlighted(text, terms) {
    const frag = document.createDocumentFragment();
    if (!terms.length) { frag.append(text); return frag; }
    const escaped = terms.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
    const re = new RegExp(`\\b(${escaped.join("|")})\\w*`, "gi");
    let last = 0;
    for (const match of text.matchAll(re)) {
      if (match.index > last) frag.append(text.slice(last, match.index));
      frag.append(el("mark", { text: match[0] }));
      last = match.index + match[0].length;
    }
    frag.append(text.slice(last));
    return frag;
  }

  function plainAnswer(text) {
    return text.replace(/\s*\[\d{1,2}(?:\s*,\s*\d{1,2})*\]/g, "").replace(/\*+/g, "").trim();
  }

  function setStatus(message, kind = "") {
    statusEl.textContent = message;
    statusEl.dataset.kind = kind;
  }

  function setView(view) {
    document.body.dataset.view = view;
    // The settings menu lives under the search box on the landing view and in
    // the toolbar above the thread on the question view.
    if (view === "session") sessionActions.prepend(settings);
    else searchUnder.append(settings);
  }

  // The address bar always holds a standalone question, so "Copy link" shares
  // something that makes sense on its own, even for a follow-up.
  function setUrl(question, mode) {
    const url = new URL(window.location.href);
    url.searchParams.set("q", question);
    if (url.href !== window.location.href) {
      if (mode === "push") history.pushState({ q: question }, "", url);
      else if (mode === "replace") history.replaceState({ q: question }, "", url);
    }
    document.title = `${question} · VidyaRAG Tutor`;
  }

  function scrollBehavior() { return reducedMotion.matches ? "auto" : "smooth"; }

  async function copyText(text, btn, done = "Copied") {
    const label = btn.querySelector(".btn-label");
    const original = label.textContent;
    try {
      await navigator.clipboard.writeText(text);
      label.textContent = done;
      btn.classList.add("is-done");
    } catch {
      label.textContent = "Couldn't copy";
    }
    setTimeout(() => { label.textContent = original; btn.classList.remove("is-done"); }, 1600);
  }

  function updateComposer() {
    followForm.hidden = !(answersEnabled && thread.length && !session.classList.contains("no-results"));
    followInput.disabled = busy;
    followBtn.disabled = busy;
  }

  // ── player ───────────────────────────────────────────────────────────────
  function showTab(which) {
    const transcript = which === "transcript";
    tabTranscript.setAttribute("aria-selected", String(transcript));
    tabSlide.setAttribute("aria-selected", String(!transcript));
    panelTranscript.hidden = !transcript;
    panelSlide.hidden = transcript;
  }

  const same = (a, b) => !!(a && b && a.turn === b.turn && a.index === b.index);

  function poster(turn, index, id, start, title) {
    const btn = el("button", { class: "poster", type: "button", "aria-label": `Play “${title}” from ${fmtTime(start)}` },
      el("img", { src: `https://i.ytimg.com/vi/${id}/hqdefault.jpg`, alt: "", width: 480, height: 360 }),
      el("span", { class: "poster-play", "aria-hidden": "true" }),
      el("span", { class: "poster-ts", text: `Starts at ${fmtTime(start)}` }),
    );
    btn.addEventListener("click", () => select(turn, index, { play: true }));
    return btn;
  }

  // Show moment `index` of `turn` in the player. play=true starts the video at
  // the moment; reveal=true opens the docked player on small screens.
  function select(turn, index, { play = false, reveal = false } = {}) {
    const r = turn.results[index];
    if (!r) return;
    const target = { turn, index };
    const changed = !same(active, target);
    const id = videoId(r.youtube_url);
    const start = Math.floor(Number(r.start_sec) || 0);
    const title = cleanTitle(r.lecture_title);
    active = target;
    for (const t of thread) {
      t.rows.forEach((row, i) => {
        if (t === turn && i === index) row.setAttribute("aria-current", "true");
        else row.removeAttribute("aria-current");
      });
    }

    if (changed) {
      const prefix = thread.length > 1 ? `Q${turn.n} · ` : "";
      npKicker.textContent = `${prefix}Moment ${r.rank ?? index + 1} · ${r.course_name || "Unknown course"}`;
      npTitle.textContent = title;
      npSub.textContent = [r.instructor, lectureNumber(r.lecture_title), `${fmtTime(r.start_sec)}–${fmtTime(r.end_sec)}`]
        .filter(Boolean).join(" · ");
      npYt.hidden = !r.youtube_deep_link;
      if (r.youtube_deep_link) npYt.href = r.youtube_deep_link;
      panelTranscript.replaceChildren(highlighted((r.transcript || "").trim(), turn.terms));
      panelTranscript.scrollTop = 0;
      const ocr = (r.ocr_text || "").trim();
      tabSlide.disabled = !ocr;
      tabSlide.title = ocr ? "" : "No slide text was read for this moment";
      panelSlide.replaceChildren(ocr ? highlighted(ocr, turn.terms) : "");
      panelSlide.scrollTop = 0;
      if (!ocr) showTab("transcript");
    }

    if (!id) {
      frameEl.replaceChildren(el("div", { class: "frame-empty", text: "No video link for this lecture" }));
      playing = null;
    } else if (play) {
      if (!same(playing, target)) {
        frameEl.replaceChildren(el("iframe", {
          src: `https://www.youtube-nocookie.com/embed/${id}?start=${start}&autoplay=1&rel=0`,
          title: `Lecture video: ${title}`,
          allow: "autoplay; encrypted-media; picture-in-picture; fullscreen",
          allowfullscreen: true,
        }));
        playing = target;
      }
    } else if (changed) {
      frameEl.replaceChildren(poster(turn, index, id, start, title));
      playing = null;
    }
    if (reveal || play) session.classList.add("player-open");
  }

  function closePlayer() {
    session.classList.remove("player-open");
    if (playing && active) {               // stop the video, keep the moment selected
      const { turn, index } = active;
      const r = turn.results[index];
      frameEl.replaceChildren(poster(turn, index, videoId(r.youtube_url), Math.floor(Number(r.start_sec) || 0),
        cleanTitle(r.lecture_title)));
      playing = null;
    }
  }

  // ── a turn: one question, its answer and its lecture moments ────────────
  function createTurn(question) {
    const n = thread.length + 1;
    const turn = {
      n, question, standalone: question, rewritten: false, terms: queryTerms(question),
      results: [], rows: [], sources: new Map(), answer: "",
    };
    turn.state = el("span", { class: "state" });
    turn.body = el("div", { class: "prose" });
    turn.meta = el("p", { class: "answer-meta" });
    turn.copyBtn = el("button", { class: "btn btn-quiet btn-sm", type: "button", hidden: true },
      icon(COPY_ICON), el("span", { class: "btn-label", text: "Copy answer" }));
    turn.copyBtn.addEventListener("click", () => copyText(answerAsText(turn), turn.copyBtn));
    turn.answerEl = el("section", { class: "answer", "aria-labelledby": `answer-h-${n}` },
      el("header", { class: "answer-head" }, el("h3", { id: `answer-h-${n}`, class: "label", text: "Answer" }), turn.state),
      turn.body,
      el("footer", { class: "answer-foot" }, turn.meta, turn.copyBtn));
    turn.countEl = el("span", { class: "count" });
    turn.listEl = el("ol", { class: "moment-list" });
    turn.momentsEl = el("details", { class: "moments", open: true },
      el("summary", { class: "moments-head" }, el("span", { class: "label", text: "Lecture moments" }), turn.countEl),
      turn.listEl);

    turn.el = el("article", { class: "turn" });
    const head = el("header", { class: "turn-head" },
      el("p", { class: "turn-tag", text: n === 1 ? "Question" : "Follow-up" }),
      el("h2", { class: "turn-q", text: question }));
    if (n > 1) {
      turn.note = el("p", { class: "turn-note is-busy", text: "Reading the conversation…" });
      head.append(turn.note);
    }
    turn.el.append(head);
    turn.el.append(turn.answerEl, turn.momentsEl);
    for (const t of thread) t.momentsEl.open = false;   // keep earlier turns compact
    thread.push(turn);
    threadEl.append(turn.el);
    return turn;
  }

  function setNote(turn, { standalone, rewritten, fallback }) {
    turn.standalone = standalone;
    turn.rewritten = rewritten;
    turn.terms = queryTerms(standalone);
    if (!turn.note) return;
    turn.note.className = "turn-note";
    if (rewritten) turn.note.replaceChildren("Searched for ", el("q", { text: standalone }));
    else if (fallback) {
      turn.note.classList.add("is-warn");
      turn.note.textContent = "Couldn't use the conversation for this one, so it was searched as typed.";
    } else turn.note.remove();
  }

  function setState(turn, state, text) {
    turn.state.dataset.state = state;
    turn.state.textContent = text;
  }

  function skeleton(turn, withPlayer) {
    const line = (w) => {
      const s = el("span", { class: "sk sk-line" });
      s.style.setProperty("--w", `${w}%`);
      return s;
    };
    turn.body.replaceChildren(line(94), line(100), line(97), line(58));
    turn.listEl.replaceChildren(...[0, 1, 2].map(() => el("li", { class: "moment is-skeleton", "aria-hidden": "true" },
      el("span", { class: "sk sk-thumb" }),
      el("div", { class: "m-body" }, line(36), line(78), line(100), line(88)))));
    if (withPlayer) frameEl.replaceChildren(el("span", { class: "sk sk-frame" }));
  }

  function renderRow(turn, r, index) {
    const id = videoId(r.youtube_url);
    const start = Math.floor(Number(r.start_sec) || 0);
    const title = cleanTitle(r.lecture_title);
    const row = el("li", { class: "moment" });
    row.style.setProperty("--i", String(index));

    let thumb;
    if (id) {
      thumb = el("button", { class: "m-thumb", type: "button", "aria-label": `Play “${title}” from ${fmtTime(start)}` },
        el("img", { src: `https://i.ytimg.com/vi/${id}/mqdefault.jpg`, alt: "", loading: "lazy", width: 320, height: 180 }),
        el("span", { class: "m-ts", text: fmtTime(start) }));
      thumb.addEventListener("click", () => select(turn, index, { play: true }));
    } else {
      thumb = el("div", { class: "m-thumb is-empty", text: "No video" });
    }

    const pick = el("button", { class: "m-pick", type: "button", text: title });
    pick.addEventListener("click", () => select(turn, index, { reveal: true }));
    const number = lectureNumber(r.lecture_title);

    row.append(thumb, el("div", { class: "m-body" },
      el("div", { class: "m-top" },
        el("span", { class: "m-n", text: String(r.rank ?? index + 1) }),
        el("span", { class: "m-course", text: r.course_name || "Unknown course" }),
        r.instructor ? el("span", { class: "m-prof", text: r.instructor }) : null,
      ),
      el("h3", { class: "m-title" }, pick),
      el("p", { class: "m-snippet" }, highlighted((r.transcript || "").trim(), turn.terms)),
      el("div", { class: "m-foot" },
        el("span", { text: `${fmtTime(r.start_sec)}–${fmtTime(r.end_sec)}` }),
        number ? el("span", { text: number }) : null,
        r.content_type ? el("span", { class: `m-type m-type-${r.content_type}`, text: r.content_type }) : null,
        el("span", { class: "m-score", title: "Cross-encoder relevance score", text: `score ${Number(r.retrieval_score).toFixed(2)}` }),
      ),
    ));
    return row;
  }

  // ── answer ───────────────────────────────────────────────────────────────
  // Read a Server-Sent Events stream from a fetch() response (EventSource is GET-only).
  async function* sseEvents(response) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) return;
      buffer += decoder.decode(value, { stream: true });
      let cut;
      while ((cut = buffer.indexOf("\n\n")) >= 0) {
        const block = buffer.slice(0, cut);
        buffer = buffer.slice(cut + 2);
        let event = "message";
        let data = "";
        for (const line of block.split("\n")) {
          if (line.startsWith("event: ")) event = line.slice(7);
          else if (line.startsWith("data: ")) data += line.slice(6);
        }
        if (data) yield { event, data: JSON.parse(data) };
      }
    }
  }

  function resultIndex(turn, n) {
    const src = turn.sources.get(n);
    return src ? turn.results.findIndex((r) => r.segment_id === src.segment_id) : -1;
  }

  function citeButton(turn, n) {
    const src = turn.sources.get(n);
    const index = resultIndex(turn, n);
    if (!src || index < 0) {
      return el("span", { class: "cite bad", "data-tip": "Doesn't match any source", text: String(n) });
    }
    const label = `${cleanTitle(src.lecture)} · ${src.timestamp}`;
    const btn = el("button", {
      class: "cite", type: "button", text: String(n), "data-tip": label,
      "aria-label": `Play source ${n}: ${label}`,
    });
    btn.addEventListener("click", () => {
      select(turn, index, { play: true });
      const row = turn.rows[index];
      row.classList.remove("flash");
      void row.offsetWidth;           // restart the highlight animation
      row.classList.add("flash");
    });
    return btn;
  }

  // Minimal, safe rendering of the model's plain text: paragraphs, bullet or
  // numbered lists, **bold**, *italic*, and [n] / [n, m] citations as play buttons.
  function renderAnswer(turn, text) {
    const frag = document.createDocumentFragment();
    const inline = (node, line) => {
      let last = 0;
      // **bold**, *italic* (not "a * b"), and [n] / [n, m] citations.
      const re = /\*\*([^*]+)\*\*|\*(?!\s)([^*\n]+?)(?<!\s)\*|\[(\d{1,2}(?:\s*,\s*\d{1,2})*)\]/g;
      for (const m of line.matchAll(re)) {
        if (m.index > last) node.append(line.slice(last, m.index));
        if (m[1] !== undefined) node.append(el("strong", { text: m[1] }));
        else if (m[2] !== undefined) node.append(el("em", { text: m[2] }));
        else for (const part of m[3].split(",")) node.append(citeButton(turn, Number(part)));
        last = m.index + m[0].length;
      }
      node.append(line.slice(last));
      return node;
    };
    let list = null;
    for (const raw of text.split(/\n+/)) {
      const line = raw.trim().replace(/^#+\s*/, "");
      if (!line) continue;
      const bullet = /^([-*•]|\d+[.)])\s+/.exec(line);
      if (bullet) {
        if (!list) frag.append(list = el(/^\d/.test(bullet[1]) ? "ol" : "ul"));
        list.append(inline(el("li"), line.slice(bullet[0].length)));
      } else {
        list = null;
        frag.append(inline(el("p"), line));
      }
    }
    return frag;
  }

  function notice(title, body, kind = "") {
    return el("div", { class: `notice ${kind}`.trim() }, el("strong", { text: title }), el("span", { text: body }));
  }

  // Plain-text copy of an answer that keeps its citations meaningful.
  function answerAsText(turn) {
    const cited = [...new Set([...turn.answer.matchAll(/\[(\d{1,2})\]/g)].map((m) => Number(m[1])))].sort((a, b) => a - b);
    const lines = cited.map((n) => {
      const s = turn.sources.get(n);
      return s ? `[${n}] ${cleanTitle(s.lecture)} (${s.course}), ${s.timestamp}: ${s.deep_link}` : null;
    }).filter(Boolean);
    return `Q: ${turn.standalone}\n\n${turn.answer.trim()}${lines.length ? `\n\nSources\n${lines.join("\n")}` : ""}`;
  }

  async function ask(turn, signal) {
    const segmentIds = turn.results.slice(0, 5).map((r) => r.segment_id).filter(Boolean);
    if (!segmentIds.length) return;

    setState(turn, "busy", `Reading ${segmentIds.length} moments`);
    turn.answerEl.classList.add("is-streaming");
    turn.answerEl.setAttribute("aria-busy", "true");

    let text = "";
    let frame = 0;
    const paint = () => { frame = 0; turn.body.replaceChildren(renderAnswer(turn, text)); };

    try {
      const response = await fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: turn.standalone.slice(0, 300), segment_ids: segmentIds }),
        signal,
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(typeof body.detail === "string" ? body.detail : `HTTP ${response.status}`);
      }
      for await (const { event, data } of sseEvents(response)) {
        if (event === "sources") {
          turn.sources = new Map(data.sources.map((s) => [s.n, s]));
        } else if (event === "token") {
          if (!text) setState(turn, "writing", "Writing");
          text += data.t;
          if (!frame) frame = requestAnimationFrame(paint);
        } else if (event === "done") {
          if (frame) cancelAnimationFrame(frame);
          if (data.not_covered) {
            turn.body.replaceChildren(notice("Not covered in these lectures",
              "The tutor only answers from the lectures, and none of the closest moments explains this. They're listed below in case they still help."));
            setState(turn, "muted", "No answer");
            turn.meta.textContent = data.gated ? "Relevance gate: the language model was not called" : "";
          } else {
            text = text || data.answer || "";
            turn.answer = text;
            paint();
            for (const n of data.cited) {
              const i = resultIndex(turn, n);
              if (i >= 0) turn.rows[i].classList.add("is-cited");
            }
            setState(turn, data.grounded ? "ok" : "warn", data.grounded ? "Citations checked" : "Check the citations");
            const parts = [data.model ? String(data.model).replace(/^openai\//, "") : null];
            if (data.citation_coverage != null) parts.push(`${Math.round(data.citation_coverage * 100)}% of sentences cited`);
            if (data.total_ms != null) parts.push(`${(data.total_ms / 1000).toFixed(1)} s`);
            if (data.cached) parts.push("cached");
            turn.meta.textContent = parts.filter(Boolean).join(" · ");
            if (!data.grounded) {
              turn.body.append(notice("Some citations couldn't be checked",
                "Verify these points against the lecture moments before relying on them.", "is-warn"));
            }
            turn.copyBtn.hidden = false;
          }
        } else if (event === "error") {
          throw new Error(data.message || "The answer failed.");
        }
      }
    } catch (err) {
      if (err.name === "AbortError") return;
      if (!text) turn.body.replaceChildren();
      turn.body.append(notice("The answer isn't available right now",
        `${err.message} The lecture moments below are still the best matches.`, "is-error"));
      setState(turn, "error", "Unavailable");
    } finally {
      turn.answerEl.classList.remove("is-streaming");
      turn.answerEl.removeAttribute("aria-busy");
    }
  }

  // ── running a turn: search, then answer ─────────────────────────────────
  async function runTurn(turn, controller) {
    const first = turn.n === 1;
    busy = true;
    updateComposer();
    setState(turn, "busy", "Finding lecture moments");
    setStatus("Searching 16,050 passages…", "busy");
    const slowHint = setTimeout(
      () => setStatus("Waking the server. The first question after a quiet spell takes a few extra seconds…", "busy"),
      6000,
    );
    turn.listEl.setAttribute("aria-busy", "true");

    try {
      const response = await fetch("/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: turn.standalone,
          top_k: Number(topk.value),
          use_bm25: useBm25.checked,
          use_rerank: useRerank.checked,
        }),
        signal: controller.signal,
      });
      const body = await response.json().catch(() => ({}));
      if (!response.ok) {
        const detail = typeof body.detail === "string"
          ? body.detail
          : response.status === 422 ? "Please enter a question between 2 and 512 characters." : `HTTP ${response.status}`;
        throw new Error(detail);
      }
      clearTimeout(slowHint);

      turn.results = body.results;
      turn.rows = body.results.map((r, i) => renderRow(turn, r, i));
      turn.listEl.replaceChildren(...turn.rows);
      turn.listEl.removeAttribute("aria-busy");
      if (first) session.classList.remove("is-loading");
      button.disabled = false;
      const n = body.results.length;
      turn.countEl.textContent = n ? `${n} found` : "";

      // Server-side pipeline time (embed → retrieve → fuse → rerank), not network.
      const seconds = Number(body.latency_sec ?? 0).toFixed(1);
      const intent = body.results[0] && body.results[0].query_intent;
      setStatus(`${n} moment${n === 1 ? "" : "s"} · retrieved in ${seconds} s${intent ? ` · ${intent} question` : ""}`, "ok");

      if (!n) {
        turn.listEl.replaceChildren(el("li", { class: "empty", text: "No matching lecture moments. Try rephrasing the question." }));
        if (first) session.classList.add("no-results");
        else turn.answerEl.hidden = true;
        return;
      }
      if (first || !playing) select(turn, 0);

      const info = await infoReady;
      if (info && info.answers_enabled) await ask(turn, controller.signal);
    } catch (err) {
      if (err.name === "AbortError") return;
      const network = err instanceof TypeError;
      if (first) {
        session.classList.remove("is-loading");
        session.classList.add("no-results");
      }
      turn.answerEl.hidden = true;
      turn.listEl.replaceChildren(el("li", { class: "empty", text: "Nothing to show for this question." }));
      setStatus(
        network
          ? "Couldn't reach the server. It may still be waking up, so try again in a few seconds."
          : `Search failed: ${err.message}`,
        "error",
      );
    } finally {
      clearTimeout(slowHint);
      if (inflight === controller) {
        inflight = null;
        busy = false;
        button.disabled = false;
        turn.listEl.removeAttribute("aria-busy");
        updateComposer();
      }
    }
  }

  // A new question from the top search box starts a new conversation.
  function startThread(query, { updateUrl = true } = {}) {
    query = query.trim();
    if (query.length < 2) {
      input.focus();
      return;
    }
    if (inflight) inflight.abort();
    const controller = new AbortController();
    inflight = controller;

    thread = [];
    active = null;
    playing = null;
    threadEl.replaceChildren();
    session.classList.remove("no-results", "player-open");
    session.classList.add("is-loading");
    infoReady.then((info) => session.classList.toggle("no-answers", !(info && info.answers_enabled)));
    setView("session");
    button.disabled = true;
    setUrl(query, updateUrl ? "push" : null);
    window.scrollTo({ top: 0, behavior: "auto" });

    const turn = createTurn(query);
    skeleton(turn, true);
    runTurn(turn, controller);
  }

  // A follow-up continues the conversation: resolve it against the recent
  // turns, then search and answer the standalone question.
  async function followUp(question) {
    question = question.trim();
    if (question.length < 2 || busy || !thread.length) return;
    const controller = new AbortController();
    inflight = controller;
    busy = true;
    updateComposer();
    followInput.value = "";

    const history = thread.slice(-HISTORY_TURNS).map((t) => ({
      question: t.standalone.slice(0, 300),
      answer: plainAnswer(t.answer).slice(0, 1500),
    }));
    const turn = createTurn(question);
    skeleton(turn, false);
    setState(turn, "busy", "Reading the conversation");
    setStatus("Working out what the follow-up refers to…", "busy");
    turn.el.scrollIntoView({ behavior: scrollBehavior(), block: "start" });

    let resolved = { standalone: question, rewritten: false, fallback: null };
    try {
      const response = await fetch("/rewrite", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: question.slice(0, 300), history }),
        signal: controller.signal,
      });
      if (response.ok) {
        const body = await response.json();
        resolved = { standalone: body.standalone || question, rewritten: !!body.rewritten, fallback: body.fallback };
      } else {
        resolved.fallback = `http_${response.status}`;
      }
    } catch (err) {
      if (err.name === "AbortError") return;
      resolved.fallback = "network";
    }
    if (inflight !== controller) return;
    setNote(turn, resolved);
    setUrl(resolved.standalone, "replace");
    await runTurn(turn, controller);
  }

  // Changing a retrieval setting re-runs the latest question, so the effect of
  // each pipeline stage is visible immediately.
  function rerunLatest() {
    const last = thread[thread.length - 1];
    if (!last) return;
    if (thread.length === 1) {
      startThread(last.question, { updateUrl: false });
      return;
    }
    if (inflight) inflight.abort();
    const controller = new AbortController();
    inflight = controller;
    thread.pop();
    last.el.remove();
    if (active && active.turn === last) active = null;
    if (playing && playing.turn === last) playing = null;
    const turn = createTurn(last.question);
    setNote(turn, { standalone: last.standalone, rewritten: last.rewritten, fallback: null });
    skeleton(turn, false);
    runTurn(turn, controller);
  }

  // ── wiring ───────────────────────────────────────────────────────────────
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    input.blur();
    startThread(input.value);
  });

  followForm.addEventListener("submit", (event) => {
    event.preventDefault();
    followUp(followInput.value);
  });

  for (const example of document.querySelectorAll(".example")) {
    example.addEventListener("click", () => {
      input.value = example.textContent;
      startThread(input.value);
    });
  }

  for (const control of [topk, useBm25, useRerank]) control.addEventListener("change", rerunLatest);

  tabTranscript.addEventListener("click", () => showTab("transcript"));
  tabSlide.addEventListener("click", () => showTab("slide"));
  playerClose.addEventListener("click", closePlayer);
  shareBtn.addEventListener("click", () => copyText(window.location.href, shareBtn, "Link copied"));
  newBtn.addEventListener("click", () => {
    history.pushState({}, "", "/");
    runFromUrl();
    window.scrollTo({ top: 0, behavior: "auto" });
    input.focus();
  });

  // On small screens the player docks to the bottom; keep the follow-up box
  // just above it by publishing the dock's height as a CSS variable.
  // Shorter placeholder where the input is narrow.
  const phone = window.matchMedia("(max-width: 640px)");
  const setPlaceholder = () => { followInput.placeholder = phone.matches ? "Ask a follow-up…" : "Ask a follow-up or a new question"; };
  phone.addEventListener("change", setPlaceholder);
  setPlaceholder();

  if ("ResizeObserver" in window) {
    new ResizeObserver(() => session.style.setProperty("--dock-h", `${sideEl.offsetHeight}px`)).observe(sideEl);
  }

  // Close the settings menu on an outside click or Escape.
  document.addEventListener("click", (event) => {
    if (settings.open && !settings.contains(event.target)) settings.open = false;
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && settings.open) {
      settings.open = false;
      settings.querySelector("summary").focus();
      return;
    }
    const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement.tagName);
    if (event.key === "/" && !typing) {
      event.preventDefault();
      const target = document.body.dataset.view === "session" && !followForm.hidden ? followInput : input;
      target.focus();
      target.select();
    }
  });

  // Shareable questions: /?q=backpropagation runs the search on load.
  function runFromUrl() {
    const q = new URL(window.location.href).searchParams.get("q");
    if (q) {
      input.value = q;
      startThread(q, { updateUrl: false });
    } else {
      if (inflight) inflight.abort();
      thread = [];
      active = null;
      playing = null;
      threadEl.replaceChildren();
      frameEl.replaceChildren();          // a hidden iframe would keep playing audio
      session.classList.remove("player-open");
      input.value = "";
      setView("landing");
      document.title = "VidyaRAG Tutor";
      updateComposer();
    }
  }
  window.addEventListener("popstate", runFromUrl);
  runFromUrl();

  infoReady.then((info) => { if (info && info.version) $("#build").textContent = String(info.version).slice(0, 7); });
})();
