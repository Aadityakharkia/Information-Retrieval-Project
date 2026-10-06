/**
 * HealthNest Home Page Interactive Logic (home.js).
 * Redirects search queries to dedicated answer page /answer.html?q=...
 */

document.addEventListener("DOMContentLoaded", () => {
  const btnSettings = document.getElementById("btnSettings");
  const settingsPopover = document.getElementById("settingsPopover");
  const btnGetStarted = document.getElementById("btnGetStarted");
  const searchInput = document.getElementById("searchInput");
  const btnSearchSubmit = document.getElementById("btnSearchSubmit");

  // Settings popover toggle
  if (btnSettings && settingsPopover) {
    btnSettings.addEventListener("click", (e) => {
      e.stopPropagation();
      settingsPopover.classList.toggle("active");
    });

    document.addEventListener("click", (e) => {
      if (!settingsPopover.contains(e.target) && e.target !== btnSettings) {
        settingsPopover.classList.remove("active");
      }
    });
  }

  // Get Started scroll
  if (btnGetStarted && searchInput) {
    btnGetStarted.addEventListener("click", () => {
      searchInput.focus();
      searchInput.scrollIntoView({ behavior: "smooth", block: "center" });
    });
  }

  // Handle Search Submission & Navigation to Dedicated Answer Page
  function triggerSearch(queryText) {
    const q = queryText || searchInput.value.trim();
    if (!q) return;

    // Save active settings to localStorage so answer page receives them
    const retriever = document.getElementById("settingRetriever") ? document.getElementById("settingRetriever").value : "hybrid";
    const topK = document.getElementById("settingTopK") ? document.getElementById("settingTopK").value : "5";
    const model = document.getElementById("settingModel") ? document.getElementById("settingModel").value : "llama3.2:1b";
    const enableAbstain = document.getElementById("settingAbstain") ? document.getElementById("settingAbstain").checked : true;
    const enableChecker = document.getElementById("settingChecker") ? document.getElementById("settingChecker").checked : true;

    localStorage.setItem("hn_retriever", retriever);
    localStorage.setItem("hn_topK", topK);
    localStorage.setItem("hn_model", model);
    localStorage.setItem("hn_abstain", enableAbstain ? "true" : "false");
    localStorage.setItem("hn_checker", enableChecker ? "true" : "false");

    // Navigate to dedicated results page /answer.html
    const targetUrl = `/answer.html?q=${encodeURIComponent(q)}`;
    window.location.href = targetUrl;
  }

  if (btnSearchSubmit) {
    btnSearchSubmit.addEventListener("click", () => triggerSearch());
  }

  if (searchInput) {
    searchInput.addEventListener("keypress", (e) => {
      if (e.key === "Enter") triggerSearch();
    });
  }

  // Preset Card Clicks
  document.querySelectorAll(".cat-card, .support-card").forEach((card) => {
    card.addEventListener("click", () => {
      const q = card.dataset.question;
      if (q) {
        triggerSearch(q);
      }
    });
  });
});
