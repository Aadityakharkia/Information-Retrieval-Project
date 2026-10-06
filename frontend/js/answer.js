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

  try {
    const data = await askQuestion({
      question,
      retriever,
      top_k: topK,
      model,
      enable_abstain: enableAbstain,
      enable_checker: enableChecker
    });

    renderAnswerData(data);
  } catch (err) {
    queryTitle.textContent = "Error Loading Answer";
    sentencesContainer.innerHTML = `<p style="color:red; font-weight:bold;">${err.message}</p>`;
  } finally {
    skeletonLoader.classList.remove("active");
    resultsSection.style.display = "block";
  }

  function renderAnswerData(data) {
    tagStatus.textContent = `Status: ${data.status.toUpperCase()}`;

    // Emergency Banner
    if (data.emergency_banner) {
      emergencyBanner.style.display = "block";
      emergencyBanner.textContent = data.emergency_banner;
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
      sentencesContainer.innerHTML = `<p style="color:#d63031; font-weight:bold; font-size:18px;">${data.self_harm_message}</p>`;
      sourcesContainer.innerHTML = "<p>No reference sources needed for safety guidance.</p>";
      return;
    }

    if (data.status === "abstained") {
      const reason = data.abstain ? data.abstain.reason : "Insufficient evidence strength.";
      sentencesContainer.innerHTML = `
        <div style="background:#fff5f5; padding:24px; border-radius:20px; border:1px solid #feb2b2;">
          <h3 style="color:#c53030; font-size:22px; margin-bottom:12px;">I don't know based on the provided pages.</h3>
          <p style="font-size:16px;"><strong>Abstention Reason:</strong> ${reason}</p>
        </div>
      `;
    } else {
      data.sentences.forEach((s) => {
        const sDiv = document.createElement("div");
        sDiv.className = "sentence-item";

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
          targetSource.classList.add("highlighted");
          targetSource.scrollIntoView({ behavior: "smooth", block: "center" });
        }
      });
    });

    // Render Sources
    sourcesContainer.innerHTML = "";
    if (data.sources && data.sources.length > 0) {
      data.sources.forEach((src) => {
        const srcDiv = document.createElement("div");
        srcDiv.className = "source-card";
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

    // Render IR Inspector
    if (data.inspector) {
      document.getElementById("inspectorSparseList").innerHTML = (data.inspector.sparse_top5 || []).map((id) => `<li>${id}</li>`).join("");
      document.getElementById("inspectorDenseList").innerHTML = (data.inspector.dense_top5 || []).map((id) => `<li>${id}</li>`).join("");
      document.getElementById("inspectorFusedList").innerHTML = (data.inspector.fused_top5 || []).map((id) => `<li>${id}</li>`).join("");
    }
  }
});
