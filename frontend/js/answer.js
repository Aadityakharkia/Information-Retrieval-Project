/**
 * Dedicated Answer Page Logic (answer.js).
 */

import { askQuestion } from "./api.js";

document.addEventListener("DOMContentLoaded", async () => {
  const urlParams = new URLSearchParams(window.location.search);
  const question = urlParams.get("q");

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
  const model = localStorage.getItem("hn_model") || "llama3.2:1b";
  const enableAbstain = localStorage.getItem("hn_abstain") !== "false";
  const enableChecker = localStorage.getItem("hn_checker") !== "false";

  tagRetriever.textContent = `Retriever: ${retriever.toUpperCase()}`;
  tagModel.textContent = `Model: ${model}`;

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
    const data = await askQuestion({
      question,
      retriever,
      top_k: topK,
      model,
      enable_abstain: enableAbstain,
      enable_checker: enableChecker
    });

    const elapsed = Date.now() - startTime;
    if (elapsed < minLoadTime) {
      await new Promise((resolve) => setTimeout(resolve, minLoadTime - elapsed));
    }

    renderAnswerData(data);
  } catch (err) {
    clearTimeout(step2Timer);
    clearTimeout(step3Timer);
    queryTitle.textContent = "Error Loading Answer";
    sentencesContainer.innerHTML = `<p style="color:#c5221f; font-weight:700;">${err.message}</p>`;
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
          <p style="color:#d63031; font-weight:700; font-size:1.15rem; line-height:1.6;">${data.self_harm_message}</p>
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
            <strong>Abstention Reason:</strong> ${reason}
          </p>
          <div style="margin-top:14px; font-size:0.85rem; color:#888888;">
            The indexed medical pages do not contain verified factual support for this specific inquiry.
          </div>
        </div>
      `;
    } else {
      data.sentences.forEach((s, idx) => {
        const sDiv = document.createElement("div");
        sDiv.className = "sentence-item stagger-item";
        sDiv.style.animationDelay = `${(idx * 0.08).toFixed(2)}s`;

        const badgeClass = s.support && s.support.supported ? "badge-supported" : "badge-not-found";
        const badgeText = s.support && s.support.supported ? "Supported" : "Not Found In Source";

        let citationHTML = "";
        if (s.citations && s.citations.length > 0) {
          s.citations.forEach((cid) => {
            citationHTML += `<a href="#source_${cid}" class="citation-tag" data-source="${cid}">[${cid}]</a>`;
          });
        }

        sDiv.innerHTML = `
          <span>${s.text}</span>
          ${citationHTML}
          <span class="answer-badge ${badgeClass}" style="margin-left:12px; font-size:11px;">${badgeText}</span>
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
            <strong style="font-size:18px; color:#111111;">${src.question}</strong>
          </div>
          <p style="font-size:15px; color:#475569; line-height:1.7;">${src.text}</p>
          <div style="margin-top:14px; font-size:13px; color:#64748b; display:flex; gap:20px;">
            <span>Retrieval Score: <strong>${src.score}</strong></span>
            <span>Matched Terms: ${src.matched_terms ? src.matched_terms.join(", ") : "N/A"}</span>
          </div>
        `;
        sourcesContainer.appendChild(srcDiv);
      });
    }

    // Render IR Inspector with smooth delays
    if (data.inspector) {
      document.getElementById("inspectorSparseList").innerHTML = (data.inspector.sparse_top5 || []).map((id) => `<li>${id}</li>`).join("");
      document.getElementById("inspectorDenseList").innerHTML = (data.inspector.dense_top5 || []).map((id) => `<li>${id}</li>`).join("");
      document.getElementById("inspectorFusedList").innerHTML = (data.inspector.fused_top5 || []).map((id) => `<li>${id}</li>`).join("");
    }
  }
});
