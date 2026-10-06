/**
 * HealthNest Home Page Interactive Logic (home.js).
 * Bento Box Design System (Dr. Preeti layout) + Live IR Typeahead & Answer Page Navigation.
 */

import { fetchSuggestions } from "./api.js";

document.addEventListener("DOMContentLoaded", () => {
  // 1. Elite Preloader
  const loader = document.getElementById("loader");
  if (loader) {
    setTimeout(() => {
      loader.style.opacity = "0";
      loader.style.visibility = "hidden";
    }, 1000);
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
      window.location.href = `/answer.html?q=${encodeURIComponent(q)}`;
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

    // 6. Live IR Typeahead Auto-Suggest Listener
    let debounceTimer = null;
    if (suggestDropdown) {
      searchInput.addEventListener("input", (e) => {
        clearTimeout(debounceTimer);
        const val = e.target.value.trim();
        if (val.length < 1) {
          suggestDropdown.classList.remove("active");
          suggestDropdown.innerHTML = "";
          return;
        }

        debounceTimer = setTimeout(async () => {
          try {
            const suggestions = await fetchSuggestions(val);
            if (!suggestions || suggestions.length === 0) {
              suggestDropdown.classList.remove("active");
              suggestDropdown.innerHTML = "";
              return;
            }

            suggestDropdown.innerHTML = "";
            suggestions.forEach((item) => {
              const div = document.createElement("div");
              div.className = "suggest-item";
              div.innerHTML = `
                <svg class="suggest-item-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                  <circle cx="11" cy="11" r="8"/>
                  <line x1="21" y1="21" x2="16.65" y2="16.65"/>
                </svg>
                <span>${item}</span>
              `;
              div.addEventListener("click", (evt) => {
                evt.stopPropagation();
                searchInput.value = item;
                suggestDropdown.classList.remove("active");
                triggerSearch(item);
              });
              suggestDropdown.appendChild(div);
            });

            suggestDropdown.classList.add("active");
          } catch (err) {
            console.error("Auto-suggest error:", err);
          }
        }, 100);
      });

      document.addEventListener("click", (e) => {
        if (!suggestDropdown.contains(e.target) && e.target !== searchInput) {
          suggestDropdown.classList.remove("active");
        }
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
