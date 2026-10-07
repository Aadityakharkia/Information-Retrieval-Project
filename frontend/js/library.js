import { fetchLibrary } from "./api.js";

const escapeHtml = (value) =>
  String(value ?? "").replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));

document.addEventListener("DOMContentLoaded", () => {
  let currentPage = 1;
  const limit = 12;
  let currentCategory = "";
  const libraryGrid = document.getElementById("libraryGrid");
  const libSearchInput = document.getElementById("libSearchInput");
  const btnLibSearch = document.getElementById("btnLibSearch");
  const btnPrevPage = document.getElementById("btnPrevPage");
  const btnNextPage = document.getElementById("btnNextPage");
  const pageIndicator = document.getElementById("pageIndicator");
  const topicList = document.getElementById("topicList");

  if (btnLibSearch) {
    btnLibSearch.addEventListener("click", () => {
      currentPage = 1;
      loadPage();
    });
  }

  if (libSearchInput) {
    libSearchInput.addEventListener("keypress", (e) => {
      if (e.key === "Enter") {
        currentPage = 1;
        loadPage();
      }
    });
  }

  if (btnPrevPage) {
    btnPrevPage.addEventListener("click", () => {
      if (currentPage > 1) {
        currentPage--;
        loadPage();
      }
    });
  }

  if (btnNextPage) {
    btnNextPage.addEventListener("click", () => {
      currentPage++;
      loadPage();
    });
  }

  async function loadCategories() {
    try {
      const res = await fetch("/api/library/categories");
      const data = await res.json();
      const categories = data.categories || [];
      topicList.innerHTML = `<li><button class="topic-btn active" data-category="">All Topics</button></li>` + 
        categories.map(c => `<li><button class="topic-btn" data-category="${escapeHtml(c)}">${escapeHtml(c)}</button></li>`).join("");
      
      topicList.querySelectorAll(".topic-btn").forEach(btn => {
        btn.addEventListener("click", (e) => {
          topicList.querySelectorAll(".topic-btn").forEach(b => b.classList.remove("active"));
          e.target.classList.add("active");
          currentCategory = e.target.getAttribute("data-category");
          currentPage = 1;
          loadPage();
        });
      });
    } catch(err) {
      console.error("Failed to load topics", err);
    }
  }

  async function loadPage() {
    libraryGrid.innerHTML = "<p style='color: #666;'>Loading document chunks...</p>";
    try {
      let url = `/api/library?page=${currentPage}&limit=${limit}`;
      if (libSearchInput && libSearchInput.value.trim()) {
        url += `&search=${encodeURIComponent(libSearchInput.value.trim())}`;
      }
      if (currentCategory) {
        url += `&category=${encodeURIComponent(currentCategory)}`;
      }
      const res = await fetch(url);
      const data = await res.json();

      if (pageIndicator) {
        pageIndicator.textContent = `Page ${data.page} of ${data.total_pages || 1}`;
      }
      libraryGrid.innerHTML = "";

      if (!data.chunks || data.chunks.length === 0) {
        libraryGrid.innerHTML = "<p style='color: #666;'>No Q&A document chunks matched your search.</p>";
        return;
      }

      data.chunks.forEach((chunk, idx) => {
        const card = document.createElement("div");
        card.className = "library-card stagger-item";
        card.style.animationDelay = `${(idx * 0.04).toFixed(2)}s`;
        card.innerHTML = `
          <div>
            <div class="library-card-id">Chunk #${escapeHtml(chunk.chunk_id)} • Doc #${escapeHtml(chunk.qa_id)}</div>
            <h4>${escapeHtml(chunk.question)}</h4>
            <p>${escapeHtml(chunk.text)}</p>
          </div>
        `;
        libraryGrid.appendChild(card);
      });
    } catch (err) {
      libraryGrid.innerHTML = `<p style="color:#c53030; font-weight:600;">Error: ${escapeHtml(err.message)}</p>`;
    }
  }

  loadCategories();
  loadPage();
});
