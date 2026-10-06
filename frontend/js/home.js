/**
 * HealthNest Home Page Interactive Logic (home.js).
 */

import { askQuestion } from "./api.js";

document.addEventListener("DOMContentLoaded", () => {
  const btnSettings = document.getElementById("btnSettings");
  const settingsPopover = document.getElementById("settingsPopover");
  const btnGetStarted = document.getElementById("btnGetStarted");
  const searchInput = document.getElementById("searchInput");
  const btnSearchSubmit = document.getElementById("btnSearchSubmit");
  const skeletonLoader = document.getElementById("skeletonLoader");
  const resultsSection = document.getElementById("resultsSection");
  const emergencyBanner = document.getElementById("emergencyBanner");
  const sentencesContainer = document.getElementById("sentencesContainer");
  const sourcesContainer = document.getElementById("sourcesContainer");
  const answerStatusBadge = document.getElementById("answerStatusBadge");

  // Settings popover toggle
  btnSettings.addEventListener("click", (e) => {
    e.stopPropagation();
    settingsPopover.classList.toggle("active");
  });

  document.addEventListener("click", (e) => {
    if (!settingsPopover.contains(e.target) && e.target !== btnSettings) {
      settingsPopover.classList.remove("active");
    }
  });

  // Get Started scroll
  btnGetStarted.addEventListener("click", () => {
    searchInput.focus();
    searchInput.scrollIntoView({ behavior: "smooth", block: "center" });
  });

  // Category card click presets
  const exampleQuestions = {
    general: "What are common remedies for severe stomach ache and gastritis?",
    medications: "What is the recommended dosage for Paracetamol in adults?",
    lifestyle: "How to improve sleep quality and overcome insomnia?"
  };

  document.querySelectorAll(".cat-card").forEach((card) => {
    card.addEventListener("click", () => {
      const cat = card.dataset.category;
      if (exampleQuestions[cat]) {
        searchInput.value = exampleQuestions[cat];
        handleSearch();
      }
    });
  });

  // Search Submission
  btnSearchSubmit.addEventListener("click", handleSearch);
  searchInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") handleSearch();
  });

  async function handleSearch() {
    const question = searchInput.value.trim();
    if (!question) return;

    // Get settings values
    const retriever = document.getElementById("settingRetriever").value;
    const topK = parseInt(document.getElementById("settingTopK").value, 10);
    const model = document.getElementById("settingModel").value;
    const enableAbstain = document.getElementById("settingAbstain").checked;
    const enableChecker = document.getElementById("settingChecker").checked;
    const useChampion = document.getElementById("settingChampion").checked;

    // Show Skeleton Loader
    resultsSection.classList.remove("active");
    skeletonLoader.classList.add("active");

    try {
      const data = await askQuestion({
        question,
        retriever,
        top_k: topK,
        model,
        enable_abstain: enableAbstain,
        enable_checker: enableChecker,
        use_champion_lists: useChampion
      });

      renderResults(data);
    } catch (err) {
      alert(`Error fetching response: ${err.message}`);
    } finally {
      skeletonLoader.classList.remove("active");
      resultsSection.classList.add("active");
      resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  function renderResults(data) {
    // 1. Emergency Banner
    if (data.emergency_banner) {
      emergencyBanner.style.display = "block";
      emergencyBanner.textContent = data.emergency_banner;
    } else {
      emergencyBanner.style.display = "none";
    }

    // 2. Status Badge
    if (data.status === "abstained") {
      answerStatusBadge.className = "answer-badge badge-not-found";
      answerStatusBadge.textContent = "Abstained (I don't know)";
    } else if (data.status === "self_harm") {
      answerStatusBadge.className = "answer-badge badge-not-found";
      answerStatusBadge.textContent = "Safety Message";
    } else {
      answerStatusBadge.className = "answer-badge badge-supported";
      answerStatusBadge.textContent = "Answer Verified";
    }

    // 3. Sentences & Abstention render
    sentencesContainer.innerHTML = "";

    if (data.status === "self_harm") {
      sentencesContainer.innerHTML = `<p class="sentence-item" style="color:#d63031; font-weight:bold;">${data.self_harm_message}</p>`;
      sourcesContainer.innerHTML = "<p>No sources retrieved for safety message.</p>";
      return;
    }

    if (data.status === "abstained") {
      const reason = data.abstain ? data.abstain.reason : "Insufficient evidence strength.";
      sentencesContainer.innerHTML = `
        <div style="background:#fff5f5; padding:20px; border-radius:14px; border:1px solid #feb2b2;">
          <h3 style="color:#c53030; margin-bottom:8px;">I don't know based on the provided pages.</h3>
          <p><strong>Abstention Reason:</strong> ${reason}</p>
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
          <span class="answer-badge ${badgeClass}" style="margin-left:10px; font-size:10px;">${badgeText}</span>
        `;

        sentencesContainer.appendChild(sDiv);
      });
    }

    // Add citation click scroll handlers
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

    // 4. Render Sources
    sourcesContainer.innerHTML = "";
    if (data.sources && data.sources.length > 0) {
      data.sources.forEach((src) => {
        const srcDiv = document.createElement("div");
        srcDiv.className = "source-card";
        srcDiv.id = `source_${src.n}`;

        srcDiv.innerHTML = `
          <div>
            <span class="source-num">[${src.n}]</span>
            <strong>${src.question}</strong>
          </div>
          <p style="margin-top:8px; font-size:14px; color:#4a5568;">${src.text}</p>
          <div style="margin-top:10px; font-size:12px; color:#718096; display:flex; gap:16px;">
            <span>Score: <strong>${src.score}</strong></span>
            <span>Matched Terms: ${src.matched_terms ? src.matched_terms.join(", ") : "N/A"}</span>
          </div>
        `;
        sourcesContainer.appendChild(srcDiv);
      });
    }

    // 5. Render Inspector
    if (data.inspector) {
      const inspectorSparseList = document.getElementById("inspectorSparseList");
      const inspectorDenseList = document.getElementById("inspectorDenseList");
      const inspectorFusedList = document.getElementById("inspectorFusedList");

      inspectorSparseList.innerHTML = (data.inspector.sparse_top5 || []).map((id) => `<li>${id}</li>`).join("");
      inspectorDenseList.innerHTML = (data.inspector.dense_top5 || []).map((id) => `<li>${id}</li>`).join("");
      inspectorFusedList.innerHTML = (data.inspector.fused_top5 || []).map((id) => `<li>${id}</li>`).join("");
    }
  }
});
