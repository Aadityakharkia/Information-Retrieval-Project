/**
 * HealthNest Frontend API Client Module.
 * Encapsulates fetch calls to Flask backend endpoints.
 */

const API_BASE = "";

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
