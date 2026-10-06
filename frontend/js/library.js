import { fetchLibrary } from "./api.js";

document.addEventListener("DOMContentLoaded", () => {
  let currentPage = 1;
  const limit = 12;
  const libraryGrid = document.getElementById("libraryGrid");
  const libSearchInput = document.getElementById("libSearchInput");
  const btnLibSearch = document.getElementById("btnLibSearch");
  const btnPrevPage = document.getElementById("btnPrevPage");
  const btnNextPage = document.getElementById("btnNextPage");
  const pageIndicator = document.getElementById("pageIndicator");

  btnLibSearch.addEventListener("click", () => {
    currentPage = 1;
    loadPage();
  });

  libSearchInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") {
      currentPage = 1;
      loadPage();
    }
  });

  btnPrevPage.addEventListener("click", () => {
    if (currentPage > 1) {
      currentPage--;
      loadPage();
    }
  });

  btnNextPage.addEventListener("click", () => {
    currentPage++;
    loadPage();
  });

  async function loadPage() {
    libraryGrid.innerHTML = "<p>Loading document chunks...</p>";
    try {
      const data = await fetchLibrary({
        page: currentPage,
        limit: limit,
        search: libSearchInput.value.trim()
      });

      pageIndicator.textContent = `Page ${data.page} of ${data.total_pages}`;
      libraryGrid.innerHTML = "";

      if (!data.chunks || data.chunks.length === 0) {
        libraryGrid.innerHTML = "<p>No Q&A document chunks matched your filter.</p>";
        return;
      }

      data.chunks.forEach((chunk) => {
        const card = document.createElement("div");
        card.className = "source-card";
        card.innerHTML = `
          <div style="font-size: 12px; color: var(--primary-purple); font-weight: 800; margin-bottom: 6px;">
            CHUNK ID: ${chunk.chunk_id} | QA ID: ${chunk.qa_id}
          </div>
          <h4 style="margin-bottom: 8px; font-size: 16px;">${chunk.question}</h4>
          <p style="font-size: 14px; color: #4a5568;">${chunk.text}</p>
        `;
        libraryGrid.appendChild(card);
      });
    } catch (err) {
      libraryGrid.innerHTML = `<p style="color:red;">Error: ${err.message}</p>`;
    }
  }

  loadPage();
});
