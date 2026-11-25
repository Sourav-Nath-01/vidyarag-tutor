// VidyaRAG Tutor page — talks to the same-origin FastAPI endpoints:
// POST /search for the lecture moments, then POST /ask (Server-Sent Events) for
// an answer written from the top results, whose [n] citations play result n.
// All text — results and the LLM's answer — is inserted with textContent /
// text nodes (never innerHTML), so neither transcripts nor model output can
// inject markup.
"use strict";

(() => {
  const $ = (sel) => document.querySelector(sel);
  const form = $("#search-form");
  const input = $("#q");
  const button = $("#search-btn");
  const topk = $("#topk");
  const useBm25 = $("#opt-bm25");
  const useRerank = $("#opt-rerank");
  const statusEl = $("#status");
  const resultsEl = $("#results");
  const introEl = $("#intro");
  const answerEl = $("#answer");
  const answerBody = $("#answer-body");
  const answerFoot = $("#answer-foot");

  // /info says whether this instance has an LLM key (answers enabled).
  const infoReady = fetch("/info").then((r) => (r.ok ? r.json() : null)).catch(() => null);

  const STOPWORDS = new Set((
    "the and but for with from into about how what why when which who does did are was were " +
    "this that these those can will would should explain define difference between using use"
  ).split(" "));

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

  // ── result card ─────────────────────────────────────────────────────────
  function playInline(card, media, id, start) {
    const src = `https://www.youtube-nocookie.com/embed/${id}?start=${start}&autoplay=1&rel=0`;
    media.replaceChildren(el("iframe", {
      src,
      title: "Lecture video",
      allow: "autoplay; encrypted-media; picture-in-picture; fullscreen",
      allowfullscreen: true,
    }));
    card.classList.add("is-playing");
  }

  function renderCard(result, terms) {
    const id = videoId(result.youtube_url);
    const start = Math.floor(Number(result.start_sec) || 0);
    const title = result.lecture_title || "Untitled lecture";
    const media = el("div", { class: "media" });
    const card = el("article", { class: "card" });

    if (id) {
      const thumb = el("button", {
        class: "thumb", type: "button",
        "aria-label": `Play “${title}” from ${fmtTime(start)}`,
      },
        el("img", { src: `https://i.ytimg.com/vi/${id}/mqdefault.jpg`, alt: "", loading: "lazy", width: 320, height: 180 }),
        el("span", { class: "play-icon", "aria-hidden": "true" }),
        el("span", { class: "ts", text: fmtTime(start) }),
      );
      thumb.addEventListener("click", () => playInline(card, media, id, start));
      media.append(thumb);
    } else {
      media.append(el("div", { class: "thumb thumb-empty", text: "No video link" }));
    }

    const chips = el("div", { class: "chips" },
      el("span", { class: "chip", text: result.course_name || "Unknown course" }),
      result.content_type
        ? el("span", { class: `chip type type-${result.content_type}`, text: result.content_type })
        : null,
    );

    const heading = el("h3", { class: "title" }, el("span", { class: "rank", text: `#${result.rank}` }), title);

    const metaParts = [result.instructor, `${fmtTime(result.start_sec)} – ${fmtTime(result.end_sec)}`].filter(Boolean);
    const meta = el("p", { class: "meta", text: metaParts.join(" · ") });

    const transcriptText = (result.transcript || "").trim();
    const transcript = el("p", { class: "transcript clamp" }, highlighted(transcriptText, terms));
    let toggle = null;
    if (transcriptText.length > 240) {
      toggle = el("button", { class: "linklike", type: "button", "aria-expanded": "false", text: "Show full transcript" });
      toggle.addEventListener("click", () => {
        const expanded = !transcript.classList.toggle("clamp");
        toggle.textContent = expanded ? "Show less" : "Show full transcript";
        toggle.setAttribute("aria-expanded", String(expanded));
      });
    }

    const ocrText = (result.ocr_text || "").trim();
    const ocr = ocrText
      ? el("details", { class: "ocr" }, el("summary", { text: "Slide text (OCR)" }), el("p", {}, highlighted(ocrText, terms)))
      : null;

    const actions = el("div", { class: "actions" });
    if (id) {
      const play = el("button", { class: "btn small", type: "button", text: `▶ Play from ${fmtTime(start)}` });
      play.addEventListener("click", () => playInline(card, media, id, start));
      actions.append(play);
    }
    if (result.youtube_deep_link) {
      actions.append(el("a", {
        class: "btn small ghost", href: result.youtube_deep_link,
        target: "_blank", rel: "noopener noreferrer", text: "Open on YouTube ↗",
      }));
    }
    actions.append(el("span", {
      class: "score", title: "Cross-encoder relevance score",
      text: `score ${Number(result.retrieval_score).toFixed(2)}`,
    }));

    card.append(media, el("div", { class: "body" }, chips, heading, meta, transcript, toggle, ocr, actions));
    card.play = () => { if (id) playInline(card, media, id, start); };
    return card;
  }

  // ── answer ──────────────────────────────────────────────────────────────
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

  function citeButton(n, sourceByN, cardBySegment) {
    const src = sourceByN.get(n);
    if (!src) return el("span", { class: "cite bad", title: "Doesn't match any source", text: `${n}` });
    const btn = el("button", {
      class: "cite", type: "button", text: String(n),
      title: `${src.lecture} · ${src.timestamp}`,
      "aria-label": `Play source ${n}: ${src.lecture} from ${src.timestamp}`,
    });
    btn.addEventListener("click", () => {
      const card = cardBySegment.get(src.segment_id);
      if (!card) return;
      card.scrollIntoView({ behavior: "smooth", block: "center" });
      card.play();
    });
    return btn;
  }

  // Minimal, safe rendering of the model's plain text: paragraphs, bullet or
  // numbered lists, **bold**, and [n] / [n, m] citations as play buttons.
  function renderAnswer(text, sourceByN, cardBySegment) {
    const frag = document.createDocumentFragment();
    const inline = (node, line) => {
      let last = 0;
      for (const m of line.matchAll(/\*\*([^*]+)\*\*|\[(\d{1,2}(?:\s*,\s*\d{1,2})*)\]/g)) {
        if (m.index > last) node.append(line.slice(last, m.index));
        if (m[1] !== undefined) node.append(el("strong", { text: m[1] }));
        else for (const part of m[2].split(",")) node.append(citeButton(Number(part), sourceByN, cardBySegment));
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

  function setAnswerFoot(message, kind = "") {
    answerFoot.textContent = message;
    answerFoot.dataset.kind = kind;
  }

  async function ask(query, results, cards, signal) {
    const cardBySegment = new Map(results.map((r, i) => [r.segment_id, cards[i]]));
    const segmentIds = results.slice(0, 5).map((r) => r.segment_id).filter(Boolean);
    if (!segmentIds.length) return;

    answerEl.hidden = false;
    answerEl.classList.add("is-streaming");
    answerBody.replaceChildren();
    setAnswerFoot("Writing an answer from these lectures…", "busy");

    let text = "";
    let sourceByN = new Map();
    let frame = 0;
    const paint = () => { frame = 0; answerBody.replaceChildren(renderAnswer(text, sourceByN, cardBySegment)); };

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
          sourceByN = new Map(data.sources.map((s) => [s.n, s]));
        } else if (event === "token") {
          text += data.t;
          if (!frame) frame = requestAnimationFrame(paint);
        } else if (event === "done") {
          if (frame) cancelAnimationFrame(frame);
          if (data.not_covered) {
            answerBody.replaceChildren(el("p", { class: "not-covered",
              text: "These lectures don't seem to cover this question, so the tutor won't guess." }));
            setAnswerFoot("The closest lecture moments are below in case they still help.", "warn");
          } else {
            text = text || data.answer || "";
            paint();
            const n = data.cited.length;
            const coverage = data.citation_coverage == null ? "" : ` · ${Math.round(data.citation_coverage * 100)}% of statements cited`;
            setAnswerFoot(data.grounded
              ? `Built from ${n} lecture moment${n === 1 ? "" : "s"}${coverage} · ${data.model}${data.cached ? " · cached" : ""}`
              : "⚠ Some citations couldn't be checked against the lectures. Verify with the moments below.",
              data.grounded ? "ok" : "warn");
          }
        } else if (event === "error") {
          throw new Error(data.message || "The answer failed.");
        }
      }
    } catch (err) {
      if (err.name === "AbortError") return;
      if (!text) answerBody.replaceChildren();
      setAnswerFoot(`${err.message} The lecture moments below are still the best matches.`, "error");
    } finally {
      answerEl.classList.remove("is-streaming");
    }
  }

  // ── search ──────────────────────────────────────────────────────────────
  let inflight = null;

  async function search(query, { updateUrl = true } = {}) {
    query = query.trim();
    if (query.length < 2) {
      setStatus("Type at least 2 characters.", "warn");
      input.focus();
      return;
    }
    if (inflight) inflight.abort();
    const controller = new AbortController();
    inflight = controller;

    button.disabled = true;
    resultsEl.setAttribute("aria-busy", "true");
    setStatus("Searching…", "busy");
    const slowHint = setTimeout(
      () => setStatus("Still working — if nobody has searched for a while, the server takes a few seconds to wake up…", "busy"),
      6000,
    );
    const started = performance.now();

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

      const terms = queryTerms(query);
      introEl.hidden = true;
      answerEl.hidden = true;
      const cards = body.results.map((r) => renderCard(r, terms));
      resultsEl.replaceChildren(...cards);
      resultsEl.removeAttribute("aria-busy");
      button.disabled = false;              // results are usable while the answer streams
      if (!body.results.length) {
        resultsEl.append(el("p", { class: "empty", text: "No matching lecture segments. Try rephrasing the question." }));
      }

      // Server-side pipeline time (embed → retrieve → fuse → rerank), not network.
      clearTimeout(slowHint);   // results are in; the answer has its own progress line
      const seconds = Number(body.latency_sec ?? (performance.now() - started) / 1000).toFixed(1);
      const intent = body.results[0] && body.results[0].query_intent;
      setStatus(
        `${body.n_results} result${body.n_results === 1 ? "" : "s"} · searched in ${seconds} s` +
        (intent ? ` · detected intent: ${intent}` : ""),
        "ok",
      );

      if (updateUrl) {
        const url = new URL(window.location.href);
        url.searchParams.set("q", query);
        if (url.href !== window.location.href) history.pushState({ q: query }, "", url);
      }
      document.title = `${query} — VidyaRAG Tutor`;

      const info = await infoReady;
      if (info && info.answers_enabled && body.results.length) {
        await ask(query, body.results, cards, controller.signal);
      }
    } catch (err) {
      if (err.name === "AbortError") return;
      const network = err instanceof TypeError;
      setStatus(
        network
          ? "Couldn't reach the server — it may still be waking up. Try again in a few seconds."
          : `Search failed: ${err.message}`,
        "error",
      );
    } finally {
      clearTimeout(slowHint);
      if (inflight === controller) {
        inflight = null;
        button.disabled = false;
        resultsEl.removeAttribute("aria-busy");
      }
    }
  }

  // ── wiring ──────────────────────────────────────────────────────────────
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    search(input.value);
  });

  for (const example of document.querySelectorAll(".example")) {
    example.addEventListener("click", () => {
      input.value = example.textContent;
      search(input.value);
    });
  }

  document.addEventListener("keydown", (event) => {
    const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement.tagName);
    if (event.key === "/" && !typing) {
      event.preventDefault();
      input.focus();
      input.select();
    }
  });

  // Shareable searches: /?q=backpropagation runs the search on load.
  function runFromUrl() {
    const q = new URL(window.location.href).searchParams.get("q");
    if (q) {
      input.value = q;
      search(q, { updateUrl: false });
    }
  }
  window.addEventListener("popstate", runFromUrl);
  runFromUrl();

  // Build identifier in the footer.
  infoReady.then((info) => { if (info && info.version) $("#build").textContent = String(info.version).slice(0, 7); });
})();
