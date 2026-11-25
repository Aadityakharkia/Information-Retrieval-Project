/**
 * HealthNest Home Page Interactive Logic (home.js).
 * Bento Box Design System (Dr. Preeti layout) + Live IR Typeahead & Answer Page Navigation.
 */

import { clearHistory, fetchHistory, fetchSuggestions, openAnswer } from "./api.js";

document.addEventListener("DOMContentLoaded", () => {
  // 1. Elite Preloader
  const loader = document.getElementById("loader");
  if (loader) {
    setTimeout(() => {
      loader.classList.add("hidden");
      setTimeout(() => {
        loader.style.display = "none";
      }, 700);
    }, 1300);
  }

  // 2. Scroll Reveal Animations
  const revealElements = document.querySelectorAll(".reveal");
  if (revealElements.length > 0) {
    const revealObserver = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("active");
            revealObserver.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.08, rootMargin: "0px 0px -40px 0px" }
    );

    revealElements.forEach((el) => revealObserver.observe(el));
  }

  // 3. Menu Overlay Toggle
  const openMenuBtn = document.getElementById("openMenuBtn");
  const closeMenuBtn = document.getElementById("closeMenuBtn");
  const menuOverlay = document.getElementById("menuOverlay");
  const menuLinks = document.querySelectorAll(".menu-link-item");

  if (openMenuBtn && closeMenuBtn && menuOverlay) {
    openMenuBtn.addEventListener("click", (e) => {
      e.preventDefault();
      menuOverlay.classList.add("active");
      document.body.style.overflow = "hidden";
    });

    closeMenuBtn.addEventListener("click", () => {
      menuOverlay.classList.remove("active");
      document.body.style.overflow = "auto";
    });

    menuLinks.forEach((link) => {
      link.addEventListener("click", () => {
        menuOverlay.classList.remove("active");
        document.body.style.overflow = "auto";
      });
    });
  }

  // 4. Info Modal Logic
  const clickableCards = document.querySelectorAll(".clickable-card");
  const infoModal = document.getElementById("infoModal");
  const closeInfoModalBtn = document.getElementById("closeInfoModalBtn");
  const infoModalTitle = document.getElementById("infoModalTitle");
  const infoModalDesc = document.getElementById("infoModalDesc");

  if (infoModal && closeInfoModalBtn) {
    clickableCards.forEach((card) => {
      card.addEventListener("click", (e) => {
        // Prevent click if clicking search bar or chip buttons inside card
        if (e.target.closest("#searchContainer") || e.target.closest("#suggestDropdown") || e.target.closest(".chip-btn")) return;

        const title = card.getAttribute("data-info-title");
        const desc = card.getAttribute("data-info-desc");

        if (title && desc) {
          infoModalTitle.textContent = title;
          infoModalDesc.textContent = desc;
          infoModal.classList.add("active");
          document.body.style.overflow = "hidden";
        }
      });
    });

    closeInfoModalBtn.addEventListener("click", () => {
      infoModal.classList.remove("active");
      document.body.style.overflow = "auto";
    });

    infoModal.addEventListener("click", (e) => {
      if (e.target === infoModal) {
        infoModal.classList.remove("active");
        document.body.style.overflow = "auto";
      }
    });
  }

  // 5. Search Submission & Navigation to Dedicated Answer Page
  const searchInput = document.getElementById("searchInput");
  const btnSearchSubmit = document.getElementById("btnSearchSubmit");
  const suggestDropdown = document.getElementById("suggestDropdown");

  function triggerSearch(queryText) {
    const q = queryText || (searchInput ? searchInput.value.trim() : "");
    if (!q) return;

    if (btnSearchSubmit) {
      btnSearchSubmit.innerHTML = `<i class="fa-solid fa-spinner fa-spin" style="font-size:16px;"></i>`;
    }
    document.body.classList.add("page-exit");
    setTimeout(() => {
      openAnswer(q);
    }, 280);
  }

  if (btnSearchSubmit) {
    btnSearchSubmit.addEventListener("click", (e) => {
      e.stopPropagation();
      triggerSearch();
    });
  }

  if (searchInput) {
    searchInput.addEventListener("keypress", (e) => {
      if (e.key === "Enter") {
        e.stopPropagation();
        triggerSearch();
      }
    });

    // 6. Recent searches (on focus) + live IR typeahead (while typing)
    let debounceTimer = null;
    let requestNo = 0; // ignore replies that arrive after a newer keystroke
    if (suggestDropdown) {
      const SEARCH_ICON = `<circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>`;
      const CLOCK_ICON = `<circle cx="12" cy="12" r="9"/><polyline points="12 7 12 12 15 14"/>`;
      const hide = () => {
        suggestDropdown.classList.remove("active");
        suggestDropdown.replaceChildren();
      };

      const addItem = (text, icon) => {
        const div = document.createElement("div");
        div.className = "suggest-item";
        div.innerHTML = `<svg class="suggest-item-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">${icon}</svg><span></span>`;
        div.querySelector("span").textContent = text; // never innerHTML: searches are user input
        div.addEventListener("click", (evt) => {
          evt.stopPropagation();
          searchInput.value = text;
          hide();
          triggerSearch(text);
        });
        suggestDropdown.appendChild(div);
      };

      const addRecentHead = () => {
        const head = document.createElement("div");
        head.className = "suggest-head";
        head.innerHTML = `<span>Recent searches</span><button type="button">Clear history</button>`;
        head.querySelector("button").addEventListener("click", async (evt) => {
          evt.stopPropagation();
          if (!window.confirm("Clear your whole search history? This cannot be undone.")) return;
          try {
            await clearHistory();
            hide();
          } catch (err) {
            console.error("Clear history error:", err);
          }
        });
        suggestDropdown.appendChild(head);
      };

      const refresh = async () => {
        const mine = ++requestNo;
        const val = searchInput.value.trim();
        const history = await fetchHistory().catch(() => []);
        if (mine !== requestNo) return;
        const recent = history.map((h) => h.query).filter((q) => q.toLowerCase().includes(val.toLowerCase())).slice(0, val ? 3 : 5);
        const show = (suggestions) => {
          const fresh = suggestions.filter((s) => !recent.some((r) => r.toLowerCase() === s.toLowerCase()));
          hide();
          if (recent.length) {
            addRecentHead();
            recent.forEach((q) => addItem(q, CLOCK_ICON));
          }
          fresh.forEach((q) => addItem(q, SEARCH_ICON));
          if (recent.length || fresh.length) suggestDropdown.classList.add("active");
        };
        show([]); // recent searches first; suggestions can take longer
        if (!val) return;
        try {
          const suggestions = await fetchSuggestions(val);
          if (mine === requestNo) show(suggestions);
        } catch (err) {
          console.error("Auto-suggest error:", err);
        }
      };

      searchInput.addEventListener("focus", refresh);
      searchInput.addEventListener("input", () => {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(refresh, 100);
      });
      searchInput.addEventListener("keydown", (e) => {
        if (e.key === "Escape") hide();
      });

      document.addEventListener("click", (e) => {
        if (!suggestDropdown.contains(e.target) && e.target !== searchInput) hide();
      });
    }
  }

  // 7. Quick Chip Buttons
  document.querySelectorAll(".chip-btn").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      const q = btn.dataset.question;
      if (q) {
        if (searchInput) searchInput.value = q;
        triggerSearch(q);
      }
    });
  });
});
