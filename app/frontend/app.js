/* ==========================================================================
   PERSONAL SECOND BRAIN — INTERACTIVE SPA JAVASCRIPT
   ========================================================================== */

document.addEventListener("DOMContentLoaded", () => {
  initNavigation();
  loadSystemStats();
  loadRecentFiles();

  // Global Keyboard listener for Cmd+K
  document.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "k") {
      e.preventDefault();
      document.getElementById("global-header-input").focus();
    }
  });

  // Global header search listener
  const headerInput = document.getElementById("global-header-input");
  headerInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && headerInput.value.trim()) {
      switchTab("chat");
      executeChatQuery(headerInput.value.trim());
      headerInput.value = "";
    }
  });

  // Dashboard hero search
  const heroBtn = document.getElementById("hero-submit-btn");
  const heroInput = document.getElementById("hero-query-input");
  if (heroBtn && heroInput) {
    const handleHeroSubmit = () => {
      const q = heroInput.value.trim();
      if (q) {
        switchTab("chat");
        executeChatQuery(q);
        heroInput.value = "";
      }
    };
    heroBtn.addEventListener("click", handleHeroSubmit);
    heroInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") handleHeroSubmit();
    });
  }

  // Chat page listeners
  const chatSendBtn = document.getElementById("chat-send-btn");
  const chatInput = document.getElementById("chat-input");
  if (chatSendBtn && chatInput) {
    const handleChatSubmit = () => {
      const q = chatInput.value.trim();
      if (q) {
        executeChatQuery(q);
        chatInput.value = "";
      }
    };
    chatSendBtn.addEventListener("click", handleChatSubmit);
    chatInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") handleChatSubmit();
    });
  }

  // Image search page listeners
  const imageSearchBtn = document.getElementById("image-search-btn");
  const imageQueryInput = document.getElementById("image-query-input");
  if (imageSearchBtn && imageQueryInput) {
    const handleImageSearch = () => {
      const q = imageQueryInput.value.trim();
      if (q) executeImageSearch(q);
    };
    imageSearchBtn.addEventListener("click", handleImageSearch);
    imageQueryInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") handleImageSearch();
    });
  }
});


/* ==========================================================================
   NAVIGATION & TAB SWITCHING
   ========================================================================== */

function initNavigation() {
  const navItems = document.querySelectorAll(".nav-item");
  navItems.forEach((item) => {
    item.addEventListener("click", () => {
      const target = item.getAttribute("data-target");
      switchTab(target);
    });
  });
}

function switchTab(targetId) {
  // Update sidebar active nav
  document.querySelectorAll(".nav-item").forEach((nav) => {
    nav.classList.toggle("active", nav.getAttribute("data-target") === targetId);
  });

  // Update page visibility
  document.querySelectorAll(".workspace-page").forEach((page) => {
    page.classList.toggle("active", page.id === `page-${targetId}`);
  });

  // Lazy load tab data
  if (targetId === "files") loadFilesExplorer();
  else if (targetId === "images") loadImagesGallery();
  else if (targetId === "faces") loadFaceMemory();
  else if (targetId === "audio") loadAudioLibrary();
  else if (targetId === "video") loadVideoLibrary();
}

function quickPrompt(promptText) {
  switchTab("chat");
  executeChatQuery(promptText);
}


/* ==========================================================================
   API SERVICE LAYER
   ========================================================================== */

async function loadSystemStats() {
  try {
    const res = await fetch("/api/stats");
    const data = await res.json();
    document.getElementById("stat-files").textContent = data.files_indexed || 0;
    document.getElementById("stat-chunks").textContent = data.document_chunks || 0;
    document.getElementById("stat-images").textContent = data.images_indexed || 0;
    document.getElementById("stat-faces").textContent = data.faces_indexed || 0;
  } catch (err) {
    console.error("Error loading stats:", err);
  }
}

async function loadRecentFiles() {
  try {
    const res = await fetch("/api/files");
    const data = await res.json();
    const listEl = document.getElementById("recent-files-list");
    if (!data.files || data.files.length === 0) {
      listEl.innerHTML = `<div style="color: var(--text-muted); font-size: 0.9rem;">No files indexed yet. Upload files to get started.</div>`;
      return;
    }

    listEl.innerHTML = data.files.slice(0, 4).map(f => `
      <div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 14px; background: rgba(255,255,255,0.03); border: var(--border-glass); border-radius: var(--radius-sm);">
        <div style="display: flex; align-items: center; gap: 10px;">
          <span style="font-size: 1.2rem;">${getFileIcon(f.file_type)}</span>
          <div>
            <div style="font-weight: 600; font-size: 0.88rem; color: var(--text-primary);">${f.filename}</div>
            <div style="font-size: 0.75rem; color: var(--text-muted);">${f.path}</div>
          </div>
        </div>
        <span style="font-size: 0.75rem; background: rgba(0, 242, 254, 0.15); color: var(--accent-cyan); padding: 2px 8px; border-radius: 10px; font-weight: 600;">INDEXED</span>
      </div>
    `).join("");
  } catch (err) {
    console.error("Error loading files:", err);
  }
}

function getFileIcon(type) {
  switch (type) {
    case 'image': return '📸';
    case 'audio': return '🎙️';
    case 'video': return '🎬';
    case 'pdf': return '📄';
    case 'docx': return '📝';
    case 'xlsx': return '📊';
    default: return '📁';
  }
}


/* ==========================================================================
   AI CHAT & RAG QUERY EXECUTION
   ========================================================================== */

async function executeChatQuery(questionText) {
  const chatThread = document.getElementById("chat-thread");

  // 1. Append User Message
  const userRow = document.createElement("div");
  userRow.className = "msg-row user";
  userRow.innerHTML = `
    <div class="msg-avatar">R</div>
    <div class="msg-bubble">${escapeHtml(questionText)}</div>
  `;
  chatThread.appendChild(userRow);

  // 2. Append AI Loading Message
  const aiRow = document.createElement("div");
  aiRow.className = "msg-row ai";
  aiRow.innerHTML = `
    <div class="msg-avatar">⚡</div>
    <div class="msg-bubble">
      <div style="display: flex; align-items: center; gap: 8px; color: var(--accent-cyan);">
        <div class="status-dot"></div> Searching Second Brain & Running Multimodal RAG...
      </div>
    </div>
  `;
  chatThread.appendChild(aiRow);
  chatThread.scrollTop = chatThread.scrollHeight;

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: questionText })
    });
    const data = await res.json();

    // Parse Markdown Answer
    let parsedAnswer = window.marked ? marked.parse(data.answer) : data.answer;

    // Check if answer contains file paths or images to format
    let mediaHTML = "";
    if (data.images && data.images.length > 0) {
      mediaHTML = `<div class="media-preview-container">` +
        data.images.map(img => `
          <div class="media-preview-card">
            <img src="/api/media/${encodeURIComponent(img.path || img)}" alt="Retrieved Image" onclick="openImageLightbox('/api/media/${encodeURIComponent(img.path || img)}')">
            <div style="padding: 8px; font-size: 0.75rem; color: var(--text-muted); text-align: center;">${getBasename(img.path || img)}</div>
          </div>
        `).join("") +
        `</div>`;
    }

    // Replace AI Loading Bubble with Result
    aiRow.querySelector(".msg-bubble").innerHTML = `
      <div>${parsedAnswer}</div>
      ${mediaHTML}
    `;

    // Update RAG Inspector Sidebar Sources
    updateRAGInspector(data.sources, data.chunks);

  } catch (err) {
    aiRow.querySelector(".msg-bubble").innerHTML = `
      <div style="color: var(--accent-rose);">Failed to process query: ${err.message}</div>
    `;
  }
  chatThread.scrollTop = chatThread.scrollHeight;
}

function updateRAGInspector(sources, chunks) {
  const inspectorList = document.getElementById("inspector-sources-list");
  if (!sources || sources.length === 0) {
    inspectorList.innerHTML = `<div style="color: var(--text-muted); font-size: 0.82rem;">No relevant source documents retrieved.</div>`;
    return;
  }

  inspectorList.innerHTML = sources.map((src, i) => `
    <div class="source-card">
      <div class="source-file">${i + 1}. ${getBasename(src)}</div>
      <div style="font-size: 0.75rem; color: var(--text-muted); word-break: break-all;">${src}</div>
    </div>
  `).join("");
}


/* ==========================================================================
   IMAGE GALLERY & FACE MEMORY
   ========================================================================== */

async function loadImagesGallery() {
  const grid = document.getElementById("images-grid");
  grid.innerHTML = `<div style="color: var(--text-muted);">Loading image library...</div>`;

  try {
    const res = await fetch("/api/files");
    const data = await res.json();
    const images = (data.files || []).filter(f => f.file_type === "image");

    if (images.length === 0) {
      grid.innerHTML = `<div style="color: var(--text-muted);">No images indexed yet.</div>`;
      return;
    }

    grid.innerHTML = images.map(img => `
      <div class="gallery-card">
        <img class="gallery-thumb" src="/api/media/${encodeURIComponent(img.path)}" alt="${img.filename}" onclick="openImageLightbox('/api/media/${encodeURIComponent(img.path)}')">
        <div class="gallery-info">
          <div class="gallery-title">${img.filename}</div>
          <div class="gallery-meta">
            <span>Image Document</span>
            <span style="color: var(--accent-cyan); cursor: pointer;" onclick="quickPrompt('Analyze this image: ${img.filename}')">Inspect</span>
          </div>
        </div>
      </div>
    `).join("");
  } catch (err) {
    grid.innerHTML = `<div style="color: var(--accent-rose);">Failed to load images.</div>`;
  }
}

async function executeImageSearch(query) {
  switchTab("chat");
  executeChatQuery(query);
}

async function loadFaceMemory() {
  const container = document.getElementById("faces-list-container");
  container.innerHTML = `<div style="color: var(--text-muted);">Loading face memory registry...</div>`;

  try {
    const res = await fetch("/api/faces");
    const data = await res.json();

    if (!data.faces || data.faces.length === 0) {
      container.innerHTML = `<div style="color: var(--text-muted);">No face memories registered yet. Click "Register Face Memory" above to add a person.</div>`;
      return;
    }

    container.innerHTML = data.faces.map(face => `
      <div class="gallery-card">
        <img class="gallery-thumb" src="/api/media/${encodeURIComponent(face.image_path)}" alt="${face.person_name}">
        <div class="gallery-info">
          <div class="gallery-title" style="color: var(--accent-cyan); font-size: 1rem;">👤 ${face.person_name}</div>
          <div class="gallery-meta" style="margin-top: 6px;">
            <span>${face.filename}</span>
            <span style="background: rgba(0, 242, 254, 0.15); padding: 2px 6px; border-radius: 4px; font-weight: 600;">512D Vector</span>
          </div>
        </div>
      </div>
    `).join("");
  } catch (err) {
    container.innerHTML = `<div style="color: var(--accent-rose);">Failed to load face registry.</div>`;
  }
}

function openRegisterFaceModal() {
  document.getElementById("face-modal").classList.add("active");
}

function closeRegisterFaceModal() {
  document.getElementById("face-modal").classList.remove("active");
}

async function submitFaceRegistration() {
  const personName = document.getElementById("modal-person-name").value.trim();
  const fileInput = document.getElementById("modal-face-file");

  if (!personName || fileInput.files.length === 0) {
    alert("Please enter a person name and choose a reference photo.");
    return;
  }

  const formData = new FormData();
  formData.append("person_name", personName);
  formData.append("file", fileInput.files[0]);

  try {
    const res = await fetch("/api/faces/register", {
      method: "POST",
      body: formData
    });
    const result = await res.json();

    if (result.status === "SUCCESS") {
      closeRegisterFaceModal();
      loadFaceMemory();
      loadSystemStats();
      alert(`Successfully registered face memory for ${personName}!`);
    } else {
      alert(`Registration failed: ${result.message}`);
    }
  } catch (err) {
    alert(`Error: ${err.message}`);
  }
}


/* ==========================================================================
   AUDIO PLAYER & VIDEO LIBRARY
   ========================================================================== */

async function loadAudioLibrary() {
  const container = document.getElementById("audio-list-container");
  container.innerHTML = `<div style="color: var(--text-muted);">Loading audio library...</div>`;

  try {
    const res = await fetch("/api/audio");
    const data = await res.json();

    if (!data.audio || data.audio.length === 0) {
      container.innerHTML = `<div style="color: var(--text-muted);">No audio files processed yet.</div>`;
      return;
    }

    container.innerHTML = data.audio.map(aud => `
      <div class="audio-card">
        <div class="audio-header">
          <div style="font-weight: 600; font-size: 1.05rem; color: var(--accent-cyan);">🎙️ ${aud.filename}</div>
          <span style="font-size: 0.75rem; background: rgba(16, 185, 129, 0.15); color: var(--accent-emerald); padding: 2px 8px; border-radius: 10px; font-weight: 600;">SROTA ASR 16kHz</span>
        </div>

        <div class="waveform-placeholder">
          <div class="wave-bar"></div>
          <div class="wave-bar"></div>
          <div class="wave-bar"></div>
          <div class="wave-bar"></div>
          <div class="wave-bar"></div>
        </div>

        <audio controls src="/api/media/${encodeURIComponent(aud.path)}"></audio>

        <div style="background: rgba(0,0,0,0.3); border: var(--border-glass); border-radius: var(--radius-sm); padding: 14px; margin-top: 8px;">
          <div style="font-size: 0.8rem; color: var(--text-muted); font-weight: 600; margin-bottom: 6px;">SEARCHABLE TRANSCRIPT:</div>
          <div style="font-size: 0.9rem; line-height: 1.5; color: var(--text-secondary);">${escapeHtml(aud.transcripts.join(" "))}</div>
        </div>
      </div>
    `).join("");
  } catch (err) {
    container.innerHTML = `<div style="color: var(--accent-rose);">Failed to load audio files.</div>`;
  }
}

async function loadVideoLibrary() {
  const container = document.getElementById("video-list-container");
  container.innerHTML = `<div style="color: var(--text-muted);">Loading video library...</div>`;

  try {
    const res = await fetch("/api/video");
    const data = await res.json();

    if (!data.videos || data.videos.length === 0) {
      container.innerHTML = `<div style="color: var(--text-muted);">No videos processed yet.</div>`;
      return;
    }

    container.innerHTML = data.videos.map(vid => `
      <div class="gallery-card" style="grid-column: span 2;">
        <video controls src="/api/media/${encodeURIComponent(vid.path)}"></video>
        <div class="gallery-info">
          <div class="gallery-title" style="color: var(--accent-cyan);">🎬 ${vid.filename}</div>
          <div class="gallery-meta" style="margin-top: 8px;">
            <span>Extracted Keyframes & ASR Track</span>
            <span style="color: var(--accent-emerald); font-weight: 600;">PROCESSED</span>
          </div>
        </div>
      </div>
    `).join("");
  } catch (err) {
    container.innerHTML = `<div style="color: var(--accent-rose);">Failed to load video library.</div>`;
  }
}

async function loadFilesExplorer() {
  const container = document.getElementById("files-grid-container");
  container.innerHTML = `<div style="color: var(--text-muted);">Loading files...</div>`;

  try {
    const res = await fetch("/api/files");
    const data = await res.json();

    if (!data.files || data.files.length === 0) {
      container.innerHTML = `<div style="color: var(--text-muted);">No indexed files found.</div>`;
      return;
    }

    container.innerHTML = data.files.map(f => `
      <div class="gallery-card">
        <div style="height: 120px; background: rgba(255,255,255,0.03); display: flex; align-items: center; justify-content: center; font-size: 3rem;">
          ${getFileIcon(f.file_type)}
        </div>
        <div class="gallery-info">
          <div class="gallery-title">${f.filename}</div>
          <div class="gallery-meta">
            <span style="font-family: var(--font-code); font-size: 0.7rem;">${f.file_type.toUpperCase()}</span>
            <span style="color: var(--accent-cyan); font-weight: 600;">INDEXED</span>
          </div>
        </div>
      </div>
    `).join("");
  } catch (err) {
    container.innerHTML = `<div style="color: var(--accent-rose);">Failed to load files.</div>`;
  }
}

async function handleFileUpload(input) {
  if (!input.files || input.files.length === 0) return;
  const file = input.files[0];

  const formData = new FormData();
  formData.append("file", file);

  alert(`Uploading and indexing file: ${file.name}...`);
  try {
    const res = await fetch("/api/upload", {
      method: "POST",
      body: formData
    });
    const data = await res.json();
    if (data.status === "SUCCESS") {
      alert(`File ${file.name} uploaded and indexed successfully!`);
      loadFilesExplorer();
      loadSystemStats();
    } else {
      alert(`Upload failed: ${data.message}`);
    }
  } catch (err) {
    alert(`Error: ${err.message}`);
  }
}


/* ==========================================================================
   UTILITY HELPERS
   ========================================================================== */

function getBasename(pathStr) {
  if (!pathStr) return "";
  return pathStr.split(/[\\/]/).pop();
}

function escapeHtml(text) {
  if (!text) return "";
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function openImageLightbox(src) {
  window.open(src, '_blank');
}
