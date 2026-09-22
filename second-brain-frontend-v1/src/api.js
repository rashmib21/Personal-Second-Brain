const API_BASE = import.meta.env.VITE_API_BASE_URL || "";

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {})
    },
    ...options
  });

  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json")
    ? await response.json()
    : await response.text();

  if (!response.ok) {
    const message =
      typeof payload === "object" && payload?.error
        ? payload.error
        : `Request failed (${response.status})`;
    throw new Error(message);
  }

  return payload;
}

/*
 * The current backend's main RAG entrypoint is /api/chat.
 * Inventory endpoints stay isolated here so backend route changes
 * never leak into the UI components.
 */
export async function askRag(question, options = {}) {
  const body = { question };

  let activeFile = null;
  if (options && typeof options === "object") {
    activeFile = options.active_file || options.source || options.activeFile || null;
  }

  let modalityValue = null;
  if (options && typeof options === "object") {
    modalityValue = options.modality || options.scope || options.type || null;
  }

  if (activeFile) {
    body.active_file = activeFile;
  }

  if (modalityValue) {
    body.modality = modalityValue;
  }

  return request("/api/chat", {
    method: "POST",
    body: JSON.stringify(body)
  });
}

export async function getFiles() {
  return request("/api/files");
}

export async function getDocuments() {
  return request("/api/documents");
}

export async function getSpreadsheets() {
  return request("/api/spreadsheets");
}

export async function getImages() {
  return request("/api/images");
}