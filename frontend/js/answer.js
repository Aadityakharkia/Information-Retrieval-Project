/**
 * Dedicated Answer Page Logic (answer.js).
 */

import { askQuestion } from "./api.js";

const escapeHtml = (value) =>
  String(value ?? "").replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));

const GENERATOR_LABELS = { groq: "Groq", ollama: "Ollama", mock: "Offline extractive", verbatim: "Word for word" };
const ORIGIN_LABELS = { prescription: "From your prescription", report: "From your report", database: "From the HealthNest database", wikipedia: "From Wikipedia" };
const SUPPORT_LABELS = { verbatim: "Word for word", supported: "Supported" };

document.addEventListener("DOMContentLoaded", async () => {
  const urlParams = new URLSearchParams(window.location.search);
  const question = urlParams.get("q") || sessionStorage.getItem("hn_question");

  const queryTitle = document.getElementById("queryTitle");
  const tagRetriever = document.getElementById("tagRetriever");
  const tagModel = document.getElementById("tagModel");
  const tagStatus = document.getElementById("tagStatus");
  const skeletonLoader = document.getElementById("skeletonLoader");
  const resultsSection = document.getElementById("resultsSection");
  const emergencyBanner = document.getElementById("emergencyBanner");
  const answerStatusBadge = document.getElementById("answerStatusBadge");
  const sentencesContainer = document.getElementById("sentencesContainer");
  const sourcesContainer = document.getElementById("sourcesContainer");

  if (!question) {
    queryTitle.textContent = "No question specified.";
    skeletonLoader.classList.remove("active");
    return;
  }

  queryTitle.textContent = question;

  // Retrieve options from localStorage
  const retriever = localStorage.getItem("hn_retriever") || "hybrid";
  const topK = parseInt(localStorage.getItem("hn_topK") || "5", 10);
  const model = localStorage.getItem("hn_model") || null; // null -> server default
  const enableAbstain = localStorage.getItem("hn_abstain") !== "false";
  const enableChecker = localStorage.getItem("hn_checker") !== "false";

  const linkSelect = document.getElementById("linkMode");
  const tagMemory = document.getElementById("tagMemory");
  const permissionContainer = document.getElementById("permissionContainer");
  const savedMode = localStorage.getItem("hn_link");
  if ([...linkSelect.options].some((o) => o.value === savedMode)) linkSelect.value = savedMode;
  linkSelect.addEventListener("change", () => {
    localStorage.setItem("hn_link", linkSelect.value);
    if (!urlParams.get("q")) sessionStorage.setItem("hn_question", question);
    window.location.reload();
  });

  const payload = {
    question,
    retriever,
    top_k: topK,
    model,
    enable_abstain: enableAbstain,
    enable_checker: enableChecker,
    link_mode: linkSelect.value
  };

  tagRetriever.textContent = `Retriever: ${retriever.toUpperCase()}`;
  tagModel.textContent = "Model: loading...";

  const minLoadTime = 1750;
  const startTime = Date.now();

  const skeletonStatusText = document.getElementById("skeletonStatusText");
  const skeletonStatusSub = document.getElementById("skeletonStatusSub");
  const stepNode1 = document.getElementById("stepNode1");
  const stepNode2 = document.getElementById("stepNode2");
  const stepNode3 = document.getElementById("stepNode3");
  const trackFill1 = document.getElementById("trackFill1");
  const trackFill2 = document.getElementById("trackFill2");

  // Multi-phase progress pipeline step simulation
  const step2Timer = setTimeout(() => {
    if (skeletonStatusText) skeletonStatusText.textContent = "Phase 2: Reciprocal Rank Fusion (RRF k=60)...";
    if (skeletonStatusSub) skeletonStatusSub.textContent = "Merging sparse keyword rankings and dense vector cosine similarities...";
    if (trackFill1) trackFill1.style.width = "100%";
    if (stepNode2) stepNode2.classList.add("active");
  }, 550);

  const step3Timer = setTimeout(() => {
    if (skeletonStatusText) skeletonStatusText.textContent = "Phase 3: Clinical Evidence Grounding & Verification...";
    if (skeletonStatusSub) skeletonStatusSub.textContent = "Validating sentence claims against retrieved medical evidence...";
    if (trackFill2) trackFill2.style.width = "100%";
    if (stepNode3) stepNode3.classList.add("active");
  }, 1150);

  try {
    const data = await askQuestion(payload);

    const elapsed = Date.now() - startTime;
    if (elapsed < minLoadTime) {
      await new Promise((resolve) => setTimeout(resolve, minLoadTime - elapsed));
    }

    renderAnswerData(data);
  } catch (err) {
    clearTimeout(step2Timer);
    clearTimeout(step3Timer);
    queryTitle.textContent = "Error Loading Answer";
    sentencesContainer.innerHTML = `<p style="color:#c5221f; font-weight:700;">${escapeHtml(err.message)}</p>`;
  } finally {
    // Smooth transition from skeleton to results
    skeletonLoader.classList.add("fade-out");
    setTimeout(() => {
      skeletonLoader.classList.remove("active");
      skeletonLoader.style.display = "none";
      resultsSection.style.display = "block";
      // Force reflow for smooth opacity/translate transition
      void resultsSection.offsetHeight;
      resultsSection.classList.add("visible");
    }, 400);
  }

  function renderAnswerData(data) {
    tagStatus.textContent = `Status: ${data.status.toUpperCase()}`;
    const engine = GENERATOR_LABELS[data.generator];
    tagModel.textContent = data.local_only ? `Model: ${engine || "local"} (this computer)` : engine ? `Model: ${data.model} (${engine})` : `Model: ${data.model}`;
    tagMemory.hidden = !data.memory_used;
    renderPermission(data);

    // Emergency Banner
    if (data.emergency_banner) {
      emergencyBanner.style.display = "block";
      emergencyBanner.textContent = data.emergency_banner;
      emergencyBanner.classList.add("stagger-item");
    } else {
      emergencyBanner.style.display = "none";
    }

    // Answer Badge
    if (data.status === "abstained") {
      answerStatusBadge.className = "answer-badge badge-not-found";
      answerStatusBadge.textContent = "Abstained (I don't know)";
    } else if (data.status === "self_harm") {
      answerStatusBadge.className = "answer-badge badge-not-found";
      answerStatusBadge.textContent = "Safety Protocol";
    } else {
      answerStatusBadge.className = "answer-badge badge-supported";
      answerStatusBadge.textContent = "Answer Verified";
    }

    sentencesContainer.innerHTML = "";

    if (data.status === "self_harm") {
      sentencesContainer.innerHTML = `
        <div class="stagger-item" style="background:#fff5f5; padding:24px; border-radius:20px; border:1px solid #feb2b2;">
          <p style="color:#d63031; font-weight:700; font-size:1.15rem; line-height:1.6;">${escapeHtml(data.self_harm_message)}</p>
        </div>
      `;
      sourcesContainer.innerHTML = "<p class='stagger-item text-subtitle'>No reference context needed for safety guidance.</p>";
      return;
    }

    if (data.status === "abstained") {
      const reason = data.abstain ? data.abstain.reason : "Insufficient evidence strength.";
      sentencesContainer.innerHTML = `
        <div class="stagger-item" style="background:#f9f9fb; padding:28px; border-radius:20px; border:1px solid var(--border-light);">
          <div style="display:flex; align-items:center; gap:10px; margin-bottom:12px;">
            <i class="fa-solid fa-circle-info" style="color:#888888; font-size:1.2rem;"></i>
            <h3 style="color:var(--text-pure); font-size:1.35rem; font-weight:800; letter-spacing:-0.02em;">I don't know based on the provided pages.</h3>
          </div>
          <p style="font-size:0.95rem; color:var(--text-muted); line-height:1.6;">
            <strong>Abstention Reason:</strong> ${escapeHtml(reason)}
          </p>
          <div style="margin-top:14px; font-size:0.85rem; color:#888888;">
            ${data.wikipedia_query ? "The Wikipedia pages found do not answer this question." : "The indexed medical pages do not contain verified factual support for this specific inquiry."}
          </div>
        </div>
      `;
    } else {
      if (data.memory_used) {
        const lines = [...(data.remarks || [])];
        if (data.sentences.some((s) => s.note)) lines.push("HealthNest compares a value only with the range printed beside it. It is not a diagnosis. A doctor must interpret your report.");
        lines.forEach((text) => {
          const p = document.createElement("p");
          p.className = "report-remark";
          p.textContent = text;
          sentencesContainer.appendChild(p);
        });
      }
      let lastOrigin = null;
      data.sentences.forEach((s, idx) => {
        if (s.origin && s.origin !== lastOrigin) {
          lastOrigin = s.origin;
          const h = document.createElement("h3");
          h.className = "origin-heading";
          h.textContent = ORIGIN_LABELS[s.origin] || "";
          sentencesContainer.appendChild(h);
        }
        const sDiv = document.createElement("div");
        sDiv.className = "sentence-item stagger-item";
        sDiv.style.animationDelay = `${(idx * 0.08).toFixed(2)}s`;

        const badgeClass = s.support && s.support.supported ? "badge-supported" : "badge-not-found";
        const badgeText = s.support && s.support.supported ? (SUPPORT_LABELS[s.support.badge] || "Supported") : "Not Found In Source";

        let citationHTML = "";
        if (s.citations && s.citations.length > 0) {
          s.citations.forEach((cid) => {
            citationHTML += `<a href="#source_${cid}" class="citation-tag" data-source="${cid}">[${cid}]</a>`;
          });
        }

        sDiv.innerHTML = `
          <span>${escapeHtml(s.text)}</span>
          ${citationHTML}
          <span class="answer-badge ${badgeClass}" style="margin-left:12px; font-size:11px;">${badgeText}</span>
          ${s.note ? `<p class="report-note">${escapeHtml(s.note)}</p>` : ""}
        `;

        sentencesContainer.appendChild(sDiv);
      });
    }

    // Citation click handler for smooth scrolling & highlighting
    document.querySelectorAll(".citation-tag").forEach((tag) => {
      tag.addEventListener("click", (e) => {
        e.preventDefault();
        const cid = tag.dataset.source;
        const targetSource = document.getElementById(`source_${cid}`);
        if (targetSource) {
          document.querySelectorAll(".source-card").forEach((sc) => sc.classList.remove("highlighted"));
          void targetSource.offsetWidth;
          targetSource.classList.add("highlighted");
          targetSource.scrollIntoView({ behavior: "smooth", block: "center" });
        }
      });
    });

    // Render Sources with cascading delay
    sourcesContainer.innerHTML = "";
    if (data.sources && data.sources.length > 0) {
      data.sources.forEach((src, idx) => {
        const srcDiv = document.createElement("div");
        srcDiv.className = "source-card stagger-item";
        srcDiv.style.animationDelay = `${(0.15 + idx * 0.08).toFixed(2)}s`;
        srcDiv.id = `source_${src.n}`;

        srcDiv.innerHTML = `
          <div style="margin-bottom:10px;">
            <span style="font-weight:900; color:#111111; font-size:18px; margin-right:8px;">[${src.n}]</span>
            <strong style="font-size:18px; color:#111111;">${escapeHtml(src.question)}</strong>
            <span class="origin-tag">${escapeHtml(ORIGIN_LABELS[src.origin] || "")}</span>
          </div>
          <p class="source-text" style="font-size:15px; color:#475569; line-height:1.7;">${escapeHtml(src.text)}</p>
          ${src.url && src.url.startsWith("https://") ? `<p style="font-size:13px; margin-top:8px;"><a href="${escapeHtml(src.url)}" target="_blank" rel="noopener noreferrer">Read on Wikipedia</a> (CC BY-SA)</p>` : ""}
          ${src.origin === "prescription" || src.origin === "report" ? "" : `<div style="margin-top:14px; font-size:13px; color:#64748b; display:flex; gap:20px; flex-wrap:wrap;">
            <span>Retrieval Score: <strong>${escapeHtml(src.score)}</strong></span>
            ${src.link ? `${src.link.pagerank !== undefined ? `<span>PageRank: <strong>${escapeHtml(src.link.pagerank)}</strong></span>` : ""}
            <span>HITS authority / hub: <strong>${escapeHtml(src.link.authority)} / ${escapeHtml(src.link.hub)}</strong></span>` : ""}
            <span>Matched Terms: ${src.matched_terms && src.matched_terms.length ? escapeHtml(src.matched_terms.join(", ")) : "N/A"}</span>
          </div>`}
        `;
        sourcesContainer.appendChild(srcDiv);
      });
    }

    // Render IR Inspector with smooth delays
    if (data.inspector) {
      document.getElementById("inspectorSparseList").innerHTML = (data.inspector.sparse_top5 || []).map((id) => `<li>${id}</li>`).join("");
      document.getElementById("inspectorDenseList").innerHTML = (data.inspector.dense_top5 || []).map((id) => `<li>${id}</li>`).join("");
      document.getElementById("inspectorFusedList").innerHTML = (data.inspector.fused_top5 || []).map((id) => `<li>${id}</li>`).join("");
      const links = data.inspector.link_analysis || {};
      const pages = data.wikipedia_pages || [];
      const rows = (items) => items.length ? items.map((t) => `<li>${escapeHtml(t.topic)} <strong>${escapeHtml(t.score)}</strong></li>`).join("") : "<li>Not used for this answer</li>";
      const wiki = (key) => pages.filter((p) => p[key] > 0).sort((x, y) => y[key] - x[key]).map((p) => ({ topic: p.title, score: p[key] }));
      document.getElementById("inspectorPageRankList").innerHTML = rows(links.top_pagerank || []);
      document.getElementById("inspectorAuthorityList").innerHTML = rows(links.top_authorities || wiki("authority"));
      document.getElementById("inspectorHubList").innerHTML = rows(links.top_hubs || wiki("hub"));
    }
  }

  /** Asks before anything is sent to Wikipedia; the user sees and can edit the exact words. */
  function renderPermission(data) {
    permissionContainer.replaceChildren();
    const offer = data.wikipedia_offer;
    if (!offer) return;
    const box = document.createElement("div");
    box.className = "permission-box";
    box.innerHTML = `
      <h3>Search Wikipedia instead?</h3>
      <p id="permissionWhy"></p>
      <input id="permissionQuery" type="text" maxlength="120" aria-label="Words to send to Wikipedia">
      <p class="private-meta">Only these words go to Wikipedia. Your documents are not sent. Wikipedia is not medically reviewed.</p>
      <div class="permission-actions" style="margin-top:0.8rem;">
        <button type="button" class="btn-pill btn-outline" id="permissionYes">Yes, search Wikipedia</button>
        <button type="button" class="btn-pill btn-white" id="permissionNo">No</button>
      </div>
      <p id="permissionMessage" class="private-message" role="status" aria-live="polite"></p>`;
    permissionContainer.appendChild(box);
    const missing = data.missing_medicines || [];
    box.querySelector("#permissionWhy").textContent = missing.length
      ? `HealthNest has no information that answers this for: ${missing.join(", ")}.`
      : "HealthNest's database does not answer this question.";
    const input = box.querySelector("#permissionQuery");
    const message = box.querySelector("#permissionMessage");
    const yes = box.querySelector("#permissionYes");
    input.value = offer.query;
    box.querySelector("#permissionNo").addEventListener("click", () => permissionContainer.replaceChildren());
    yes.addEventListener("click", async () => {
      const term = input.value.trim();
      if (!term) {
        message.textContent = "Enter the words to search for.";
        return;
      }
      yes.disabled = true;
      message.textContent = "Searching Wikipedia…";
      try {
        renderAnswerData(await askQuestion({ ...payload, allow_wikipedia: true, wikipedia_query: term }));
      } catch (err) {
        yes.disabled = false;
        message.textContent = err.message;
        message.classList.add("error");
      }
    });
  }
});
