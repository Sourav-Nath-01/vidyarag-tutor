// VidyaRAG Tutor page. Talks to the same-origin FastAPI endpoints:
// POST /search for the lecture moments, then POST /ask (Server-Sent Events) for
// an answer written from the top results, whose [n] citations play result n in
// the player. All text (results and the LLM's answer) is inserted with
// textContent / text nodes, never innerHTML, so neither transcripts nor model
// output can inject markup.
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
  const answerEl = $("#answer");
  const answerBody = $("#answer-body");
  const answerState = $("#answer-state");
  const answerMeta = $("#answer-meta");
  const copyBtn = $("#copy-btn");
  const listEl = $("#results");
  const countEl = $("#moments-count");
  const playerEl = $("#player");
  const frameEl = $("#player-frame");
  const npKicker = $("#np-kicker");
  const npTitle = $("#np-title");
  const npSub = $("#np-sub");
  const npYt = $("#np-yt");
  const tabTranscript = $("#tab-transcript");
  const tabSlide = $("#tab-slide");
  const panelTranscript = $("#panel-transcript");
  const panelSlide = $("#panel-slide");

  // /info says whether this instance has an LLM key (answers enabled).
  const infoReady = fetch("/info").then((r) => (r.ok ? r.json() : null)).catch(() => null);
  const narrow = window.matchMedia("(max-width: 1020px)");
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

  const STOPWORDS = new Set((
    "the and but for with from into about how what why when which who does did are was were " +
    "this that these those can will would should explain define difference between using use"
  ).split(" "));

  // The current question: its results, rows, sources and answer text.
  let current = { query: "", terms: [], results: [], rows: [], sources: new Map(), answer: "", active: -1, playing: -1 };

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

  function setStatus(message, kind = "") {
    statusEl.textContent = message;
    statusEl.dataset.kind = kind;
  }

  function setView(view) {
    document.body.dataset.view = view;
    // The settings menu lives under the search box on the landing view and in
    // the toolbar above the results on the question view.
    if (view === "session") sessionActions.prepend(settings);
    else searchUnder.append(settings);
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

  // ── player ───────────────────────────────────────────────────────────────
  function showTab(which) {
    const transcript = which === "transcript";
    tabTranscript.setAttribute("aria-selected", String(transcript));
    tabSlide.setAttribute("aria-selected", String(!transcript));
    panelTranscript.hidden = !transcript;
    panelSlide.hidden = transcript;
  }

  function poster(id, start, title, index) {
    const btn = el("button", { class: "poster", type: "button", "aria-label": `Play “${title}” from ${fmtTime(start)}` },
      el("img", { src: `https://i.ytimg.com/vi/${id}/hqdefault.jpg`, alt: "", width: 480, height: 360 }),
      el("span", { class: "poster-play", "aria-hidden": "true" }),
      el("span", { class: "poster-ts", text: `Starts at ${fmtTime(start)}` }),
    );
    btn.addEventListener("click", () => select(index, { play: true }));
    return btn;
  }

  // Show result `index` in the player. play=true starts the video at the moment.
  function select(index, { play = false, scroll = false } = {}) {
    const r = current.results[index];
    if (!r) return;
    const id = videoId(r.youtube_url);
    const start = Math.floor(Number(r.start_sec) || 0);
    const title = cleanTitle(r.lecture_title);
    const changed = current.active !== index;
    current.active = index;
    current.rows.forEach((row, i) => {
      if (i === index) row.setAttribute("aria-current", "true");
      else row.removeAttribute("aria-current");
    });

    if (changed) {
      npKicker.textContent = `Moment ${r.rank ?? index + 1} · ${r.course_name || "Unknown course"}`;
      npTitle.textContent = title;
      npSub.textContent = [
        r.instructor,
        lectureNumber(r.lecture_title),
        `${fmtTime(r.start_sec)}–${fmtTime(r.end_sec)}`,
      ].filter(Boolean).join(" · ");
      npYt.hidden = !r.youtube_deep_link;
      if (r.youtube_deep_link) npYt.href = r.youtube_deep_link;
      panelTranscript.replaceChildren(highlighted((r.transcript || "").trim(), current.terms));
      panelTranscript.scrollTop = 0;
      const ocr = (r.ocr_text || "").trim();
      tabSlide.disabled = !ocr;
      tabSlide.title = ocr ? "" : "No slide text was read for this moment";
      panelSlide.replaceChildren(ocr ? highlighted(ocr, current.terms) : "");
      panelSlide.scrollTop = 0;
      if (!ocr) showTab("transcript");
    }

    if (!id) {
      frameEl.replaceChildren(el("div", { class: "frame-empty", text: "No video link for this lecture" }));
      current.playing = -1;
    } else if (play) {
      if (current.playing !== index) {
        frameEl.replaceChildren(el("iframe", {
          src: `https://www.youtube-nocookie.com/embed/${id}?start=${start}&autoplay=1&rel=0`,
          title: `Lecture video: ${title}`,
          allow: "autoplay; encrypted-media; picture-in-picture; fullscreen",
          allowfullscreen: true,
        }));
        current.playing = index;
      }
    } else if (changed) {
      frameEl.replaceChildren(poster(id, start, title, index));
      current.playing = -1;
    }

    if (scroll && narrow.matches) playerEl.scrollIntoView({ behavior: scrollBehavior(), block: "start" });
  }

  // ── moments list ─────────────────────────────────────────────────────────
  function renderRow(r, index) {
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
      thumb.addEventListener("click", () => select(index, { play: true, scroll: true }));
    } else {
      thumb = el("div", { class: "m-thumb is-empty", text: "No video" });
    }

    const pick = el("button", { class: "m-pick", type: "button", text: title });
    pick.addEventListener("click", () => select(index, { scroll: true }));

    row.append(thumb, el("div", { class: "m-body" },
      el("div", { class: "m-top" },
        el("span", { class: "m-n", text: String(r.rank ?? index + 1) }),
        el("span", { class: "m-course", text: r.course_name || "Unknown course" }),
        r.instructor ? el("span", { class: "m-prof", text: r.instructor }) : null,
      ),
      el("h3", { class: "m-title" }, pick),
      el("p", { class: "m-snippet" }, highlighted((r.transcript || "").trim(), current.terms)),
      el("div", { class: "m-foot" },
        el("span", { text: `${fmtTime(r.start_sec)}–${fmtTime(r.end_sec)}` }),
        lectureNumber(r.lecture_title) ? el("span", { text: lectureNumber(r.lecture_title) }) : null,
        r.content_type ? el("span", { class: `m-type m-type-${r.content_type}`, text: r.content_type }) : null,
        el("span", { class: "m-score", title: "Cross-encoder relevance score", text: `score ${Number(r.retrieval_score).toFixed(2)}` }),
      ),
    ));
    return row;
  }

  function skeleton() {
    const line = (w, extra = "") => {
      const s = el("span", { class: `sk sk-line ${extra}`.trim() });
      s.style.setProperty("--w", `${w}%`);
      return s;
    };
    answerBody.replaceChildren(line(94), line(100), line(97), line(58));
    answerMeta.textContent = "";
    copyBtn.hidden = true;
    listEl.replaceChildren(...[0, 1, 2].map(() => el("li", { class: "moment is-skeleton", "aria-hidden": "true" },
      el("span", { class: "sk sk-thumb" }),
      el("div", { class: "m-body" }, line(36), line(78), line(100), line(88)))));
    frameEl.replaceChildren(el("span", { class: "sk sk-frame" }));
    countEl.textContent = "";
  }

  // ── answer ───────────────────────────────────────────────────────────────
  function setAnswerState(state, text) {
    answerState.dataset.state = state;
    answerState.textContent = text;
  }

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

  function citeButton(n) {
    const src = current.sources.get(n);
    const index = src ? current.results.findIndex((r) => r.segment_id === src.segment_id) : -1;
    if (!src || index < 0) {
      return el("span", { class: "cite bad", "data-tip": "Doesn't match any source", text: String(n) });
    }
    const label = `${cleanTitle(src.lecture)} · ${src.timestamp}`;
    const btn = el("button", {
      class: "cite", type: "button", text: String(n), "data-tip": label,
      "aria-label": `Play source ${n}: ${label}`,
    });
    btn.addEventListener("click", () => {
      select(index, { play: true, scroll: true });
      const row = current.rows[index];
      row.classList.remove("flash");
      void row.offsetWidth;           // restart the highlight animation
      row.classList.add("flash");
    });
    return btn;
  }

  // Minimal, safe rendering of the model's plain text: paragraphs, bullet or
  // numbered lists, **bold**, and [n] / [n, m] citations as play buttons.
  function renderAnswer(text) {
    const frag = document.createDocumentFragment();
    const inline = (node, line) => {
      let last = 0;
      for (const m of line.matchAll(/\*\*([^*]+)\*\*|\[(\d{1,2}(?:\s*,\s*\d{1,2})*)\]/g)) {
        if (m.index > last) node.append(line.slice(last, m.index));
        if (m[1] !== undefined) node.append(el("strong", { text: m[1] }));
        else for (const part of m[2].split(",")) node.append(citeButton(Number(part)));
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

  // Plain-text copy of the answer that keeps its citations meaningful.
  function answerAsText() {
    const cited = [...new Set([...current.answer.matchAll(/\[(\d{1,2})\]/g)].map((m) => Number(m[1])))].sort((a, b) => a - b);
    const lines = cited.map((n) => {
      const s = current.sources.get(n);
      return s ? `[${n}] ${cleanTitle(s.lecture)} (${s.course}), ${s.timestamp}: ${s.deep_link}` : null;
    }).filter(Boolean);
    return `Q: ${current.query}\n\n${current.answer.trim()}${lines.length ? `\n\nSources\n${lines.join("\n")}` : ""}`;
  }

  async function ask(query, results, signal) {
    const segmentIds = results.slice(0, 5).map((r) => r.segment_id).filter(Boolean);
    if (!segmentIds.length) return;

    setAnswerState("busy", `Reading ${segmentIds.length} moments`);
    answerEl.classList.add("is-streaming");
    answerEl.setAttribute("aria-busy", "true");

    let text = "";
    let frame = 0;
    const paint = () => { frame = 0; answerBody.replaceChildren(renderAnswer(text)); };

    try {
      const response = await fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: query.slice(0, 300), segment_ids: segmentIds }),
        signal,
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(typeof body.detail === "string" ? body.detail : `HTTP ${response.status}`);
      }
      for await (const { event, data } of sseEvents(response)) {
        if (event === "sources") {
          current.sources = new Map(data.sources.map((s) => [s.n, s]));
        } else if (event === "token") {
          if (!text) setAnswerState("writing", "Writing");
          text += data.t;
          if (!frame) frame = requestAnimationFrame(paint);
        } else if (event === "done") {
          if (frame) cancelAnimationFrame(frame);
          if (data.not_covered) {
            answerBody.replaceChildren(notice("Not covered in these lectures",
              "The tutor only answers from the lectures, and none of the closest moments explains this. They're listed below in case they still help."));
            setAnswerState("muted", "No answer");
            answerMeta.textContent = data.gated ? "Relevance gate: the language model was not called" : "";
          } else {
            text = text || data.answer || "";
            current.answer = text;
            paint();
            for (const n of data.cited) {
              const src = current.sources.get(n);
              const i = src ? current.results.findIndex((r) => r.segment_id === src.segment_id) : -1;
              if (i >= 0) current.rows[i].classList.add("is-cited");
            }
            setAnswerState(data.grounded ? "ok" : "warn", data.grounded ? "Citations checked" : "Check the citations");
            const parts = [data.model ? String(data.model).replace(/^openai\//, "") : null];
            if (data.citation_coverage != null) parts.push(`${Math.round(data.citation_coverage * 100)}% of sentences cited`);
            if (data.total_ms != null) parts.push(`${(data.total_ms / 1000).toFixed(1)} s`);
            if (data.cached) parts.push("cached");
            answerMeta.textContent = parts.filter(Boolean).join(" · ");
            if (!data.grounded) {
              answerBody.append(notice("Some citations couldn't be checked",
                "Verify these points against the lecture moments before relying on them.", "is-warn"));
            }
            copyBtn.hidden = false;
          }
        } else if (event === "error") {
          throw new Error(data.message || "The answer failed.");
        }
      }
    } catch (err) {
      if (err.name === "AbortError") return;
      if (!text) answerBody.replaceChildren();
      answerBody.append(notice("The answer isn't available right now",
        `${err.message} The lecture moments below are still the best matches.`, "is-error"));
      setAnswerState("error", "Unavailable");
    } finally {
      answerEl.classList.remove("is-streaming");
      answerEl.removeAttribute("aria-busy");
    }
  }

  // ── search ───────────────────────────────────────────────────────────────
  let inflight = null;

  async function search(query, { updateUrl = true } = {}) {
    query = query.trim();
    if (query.length < 2) {
      input.focus();
      return;
    }
    if (inflight) inflight.abort();
    const controller = new AbortController();
    inflight = controller;

    current = { query, terms: queryTerms(query), results: [], rows: [], sources: new Map(), answer: "", active: -1, playing: -1 };
    setView("session");
    session.classList.remove("no-results");
    session.classList.add("is-loading");
    infoReady.then((info) => session.classList.toggle("no-answers", !(info && info.answers_enabled)));
    skeleton();
    setAnswerState("busy", "Finding lecture moments");
    button.disabled = true;
    listEl.setAttribute("aria-busy", "true");
    setStatus("Searching 16,050 passages…", "busy");
    const slowHint = setTimeout(
      () => setStatus("Waking the server. The first question after a quiet spell takes a few extra seconds…", "busy"),
      6000,
    );
    if (updateUrl) {
      const url = new URL(window.location.href);
      url.searchParams.set("q", query);
      if (url.href !== window.location.href) history.pushState({ q: query }, "", url);
    }
    document.title = `${query} · VidyaRAG Tutor`;
    window.scrollTo({ top: 0, behavior: "auto" });

    try {
      const response = await fetch("/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
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

      current.results = body.results;
      current.rows = body.results.map((r, i) => renderRow(r, i));
      listEl.replaceChildren(...current.rows);
      listEl.removeAttribute("aria-busy");
      session.classList.remove("is-loading");
      button.disabled = false;              // results are usable while the answer streams
      const n = body.results.length;
      countEl.textContent = n ? `${n} found` : "";

      // Server-side pipeline time (embed → retrieve → fuse → rerank), not network.
      const seconds = Number(body.latency_sec ?? 0).toFixed(1);
      const intent = body.results[0] && body.results[0].query_intent;
      setStatus(`${n} moment${n === 1 ? "" : "s"} · retrieved in ${seconds} s${intent ? ` · ${intent} question` : ""}`, "ok");

      if (!n) {
        session.classList.add("no-results");
        listEl.replaceChildren(el("li", { class: "empty", text: "No matching lecture moments. Try rephrasing the question." }));
        return;
      }
      select(0);

      const info = await infoReady;
      if (info && info.answers_enabled) await ask(query, body.results, controller.signal);
    } catch (err) {
      if (err.name === "AbortError") return;
      const network = err instanceof TypeError;
      session.classList.remove("is-loading");
      session.classList.add("no-results");
      listEl.replaceChildren(el("li", { class: "empty", text: "Nothing to show yet." }));
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
        button.disabled = false;
        listEl.removeAttribute("aria-busy");
      }
    }
  }

  // ── wiring ───────────────────────────────────────────────────────────────
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    input.blur();
    search(input.value);
  });

  for (const example of document.querySelectorAll(".example")) {
    example.addEventListener("click", () => {
      input.value = example.textContent;
      search(input.value);
    });
  }

  // Changing a retrieval setting re-runs the current question, so the effect of
  // each pipeline stage is visible immediately.
  for (const control of [topk, useBm25, useRerank]) {
    control.addEventListener("change", () => {
      if (current.query) search(current.query, { updateUrl: false });
    });
  }

  tabTranscript.addEventListener("click", () => showTab("transcript"));
  tabSlide.addEventListener("click", () => showTab("slide"));
  shareBtn.addEventListener("click", () => copyText(window.location.href, shareBtn, "Link copied"));
  copyBtn.addEventListener("click", () => copyText(answerAsText(), copyBtn));

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
      input.focus();
      input.select();
    }
  });

  // Shareable questions: /?q=backpropagation runs the search on load.
  function runFromUrl() {
    const q = new URL(window.location.href).searchParams.get("q");
    if (q) {
      input.value = q;
      search(q, { updateUrl: false });
    } else {
      if (inflight) inflight.abort();
      current = { ...current, query: "" };
      input.value = "";
      setView("landing");
      document.title = "VidyaRAG Tutor";
    }
  }
  window.addEventListener("popstate", runFromUrl);
  runFromUrl();

  infoReady.then((info) => { if (info && info.version) $("#build").textContent = String(info.version).slice(0, 7); });
})();
