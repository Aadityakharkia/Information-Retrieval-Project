/**
 * HealthNest Frontend API Client Module.
 * Encapsulates fetch calls to Flask backend endpoints.
 */

const API_BASE = (typeof window !== "undefined" && (window.location.protocol === "file:" || (window.location.hostname !== "127.0.0.1" && window.location.hostname !== "localhost")))
  ? "http://127.0.0.1:5001"
  : "";

export async function askQuestion(payload) {
  const response = await fetch(`${API_BASE}/api/ask`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload)
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.error || `Server returned HTTP ${response.status}`);
  }

  return response.json();
}

export async function fetchSuggestions(prefix) {
  if (!prefix || prefix.trim().length === 0) return [];
  const response = await fetch(`${API_BASE}/api/suggest?q=${encodeURIComponent(prefix)}`);
  if (!response.ok) return [];
  const data = await response.json();
  return data.suggestions || [];
}

export async function fetchLibrary({ page = 1, limit = 12, search = "", category = "" } = {}) {
  const params = new URLSearchParams({ page, limit, search, category });
  const response = await fetch(`${API_BASE}/api/library?${params}`);
  if (!response.ok) {
    throw new Error(`Failed to load library: HTTP ${response.status}`);
  }
  return response.json();
}

export async function fetchModels() {
  const response = await fetch(`${API_BASE}/api/models`);
  if (!response.ok) {
    throw new Error(`Failed to load models: HTTP ${response.status}`);
  }
  return response.json();
}

export async function fetchEvalSummary() {
  const response = await fetch(`${API_BASE}/api/eval/summary`);
  if (!response.ok) {
    throw new Error(`Failed to load eval summary: HTTP ${response.status}`);
  }
  return response.json();
}

// ---------------------------------------------------------------------------
// Private patient memory. The browser's HttpOnly cookie identifies the patient.
// ---------------------------------------------------------------------------
async function privateCall(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || `Server returned HTTP ${response.status}`);
  return data;
}

export const fetchHistory = async (limit = 100) => (await privateCall(`/api/history?limit=${limit}`)).history;
export const clearHistory = () => privateCall("/api/history", { method: "DELETE" });
export const deleteSearch = (id) => privateCall(`/api/history/${id}`, { method: "DELETE" });
export const fetchDocuments = async () => (await privateCall("/api/documents")).documents;
export const deleteDocument = (id) => privateCall(`/api/documents/${id}`, { method: "DELETE" });
export const deleteEverything = () => privateCall("/api/me", { method: "DELETE" });
export function uploadDocument(file) {
  const body = new FormData();
  body.append("file", file);
  return privateCall("/api/documents", { method: "POST", body });
}

/** Opens the answer page. The question travels in sessionStorage, not the URL, so that
 *  health questions do not stay in the browser's own history after HealthNest history is cleared. */
export function openAnswer(question) {
  sessionStorage.setItem("hn_question", question);
  window.location.href = "/answer";
}
