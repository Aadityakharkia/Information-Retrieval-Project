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

  async function loadPage() {
    libraryGrid.innerHTML = "<p style='color: #666;'>Loading document chunks...</p>";
    try {
      const data = await fetchLibrary({
        page: currentPage,
        limit: limit,
        search: libSearchInput ? libSearchInput.value.trim() : ""
      });

      if (pageIndicator) {
        pageIndicator.textContent = `Page ${data.page} of ${data.total_pages || 1}`;
      }
      libraryGrid.innerHTML = "";

      if (!data.chunks || data.chunks.length === 0) {
        libraryGrid.innerHTML = "<p style='color: #666;'>No Q&A document chunks matched your search.</p>";
        return;
      }

      data.chunks.forEach((chunk) => {
        const card = document.createElement("div");
        card.className = "library-card";
        card.innerHTML = `
          <div>
            <div class="library-card-id">Chunk #${chunk.chunk_id} • Doc #${chunk.qa_id}</div>
            <h4>${chunk.question}</h4>
            <p>${chunk.text}</p>
          </div>
        `;
        libraryGrid.appendChild(card);
      });
    } catch (err) {
      libraryGrid.innerHTML = `<p style="color:#c53030; font-weight:600;">Error: ${err.message}</p>`;
    }
  }

  loadPage();
});
