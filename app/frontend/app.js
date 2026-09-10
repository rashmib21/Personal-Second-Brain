/* ==========================================================================
   PERSONAL SECOND BRAIN — INTERACTIVE SPA JAVASCRIPT
   Coding Style: Beginner-Friendly, Readable, Step-by-Step, and Interview-Ready
   ========================================================================== */

// Execute initialization logic once the web page DOM structure is fully loaded
document.addEventListener("DOMContentLoaded", function () {
  // Step 1: Initialize navigation event listeners
  initNavigation();

  // Step 2: Fetch and display initial system statistics and recent files
  loadSystemStats();
  loadRecentFiles();

  // Step 3: Global Keyboard listener for Cmd+K or Ctrl+K search shortcut
  document.addEventListener("keydown", function (event) {
    const isModifierKeyPressed = event.metaKey || event.ctrlKey;
    const isKeyKPressed = event.key === "k";

    if (isModifierKeyPressed && isKeyKPressed) {
      event.preventDefault();
      const globalHeaderInput = document.getElementById("global-header-input");
      if (globalHeaderInput) {
        globalHeaderInput.focus();
      }
    }
  });

  // Step 4: Global header search listener (Press Enter to query)
  const headerInput = document.getElementById("global-header-input");
  if (headerInput) {
    headerInput.addEventListener("keydown", function (event) {
      const isEnterPressed = event.key === "Enter";
      const userQuery = headerInput.value.trim();

      if (isEnterPressed && userQuery !== "") {
        switchTab("chat");
        executeChatQuery(userQuery);
        headerInput.value = "";
      }
    });
  }

  // Step 5: Dashboard hero search section listeners
  const heroButton = document.getElementById("hero-submit-btn");
  const heroInput = document.getElementById("hero-query-input");

  if (heroButton && heroInput) {
    function handleHeroSubmit() {
      const userQuery = heroInput.value.trim();
      if (userQuery !== "") {
        switchTab("chat");
        executeChatQuery(userQuery);
        heroInput.value = "";
      }
    }

    heroButton.addEventListener("click", function () {
      handleHeroSubmit();
    });

    heroInput.addEventListener("keydown", function (event) {
      if (event.key === "Enter") {
        handleHeroSubmit();
      }
    });
  }

  // Step 6: Chat page send button and enter key listeners
  const chatSendButton = document.getElementById("chat-send-btn");
  const chatInput = document.getElementById("chat-input");

  if (chatSendButton && chatInput) {
    function handleChatSubmit() {
      const userQuery = chatInput.value.trim();
      if (userQuery !== "") {
        executeChatQuery(userQuery);
        chatInput.value = "";
      }
    }

    chatSendButton.addEventListener("click", function () {
      handleChatSubmit();
    });

    chatInput.addEventListener("keydown", function (event) {
      if (event.key === "Enter") {
        handleChatSubmit();
      }
    });
  }

  // Step 7: Image search page listeners
  const imageSearchButton = document.getElementById("image-search-btn");
  const imageQueryInput = document.getElementById("image-query-input");

  if (imageSearchButton && imageQueryInput) {
    function handleImageSearchSubmit() {
      const userQuery = imageQueryInput.value.trim();
      if (userQuery !== "") {
        executeImageSearch(userQuery);
      }
    }

    imageSearchButton.addEventListener("click", function () {
      handleImageSearchSubmit();
    });

    imageQueryInput.addEventListener("keydown", function (event) {
      if (event.key === "Enter") {
        handleImageSearchSubmit();
      }
    });
  }
});


/* ==========================================================================
   NAVIGATION & TAB SWITCHING
   ========================================================================== */

/**
 * Attaches click event listeners to all navigation items in the sidebar.
 */
function initNavigation() {
  const navigationItems = document.querySelectorAll(".nav-item");

  navigationItems.forEach(function (navItem) {
    navItem.addEventListener("click", function () {
      const targetTabId = navItem.getAttribute("data-target");
      switchTab(targetTabId);
    });
  });
}

/**
 * Switches active visible tab page and updates sidebar navigation highlight.
 * @param {string} targetTabId - The ID of the tab to switch to (e.g. 'chat', 'files', 'images')
 */
function switchTab(targetTabId) {
  // Step 1: Update active highlight on sidebar navigation links
  const navigationItems = document.querySelectorAll(".nav-item");
  navigationItems.forEach(function (navItem) {
    const itemTarget = navItem.getAttribute("data-target");
    if (itemTarget === targetTabId) {
      navItem.classList.add("active");
    } else {
      navItem.classList.remove("active");
    }
  });

  // Step 2: Show the target workspace page and hide all other pages
  const workspacePages = document.querySelectorAll(".workspace-page");
  workspacePages.forEach(function (pageElement) {
    const expectedPageId = "page-" + targetTabId;
    if (pageElement.id === expectedPageId) {
      pageElement.classList.add("active");
    } else {
      pageElement.classList.remove("active");
    }
  });

  // Step 3: Lazy load data specific to the selected tab page
  if (targetTabId === "files") {
    loadFilesExplorer();
  } else if (targetTabId === "images") {
    loadImagesGallery();
  } else if (targetTabId === "faces") {
    loadFaceMemory();
  } else if (targetTabId === "audio") {
    loadAudioLibrary();
  } else if (targetTabId === "video") {
    loadVideoLibrary();
  }
}

/**
 * Helper function to trigger a chat query from quick prompt buttons.
 * @param {string} promptText - The pre-written prompt text to send
 */
function quickPrompt(promptText) {
  switchTab("chat");
  executeChatQuery(promptText);
}


/* ==========================================================================
   API SERVICE LAYER — SYSTEM STATS & RECENT FILES
   ========================================================================== */

/**
 * Fetches system index statistics from the backend server API.
 */
async function loadSystemStats() {
  try {
    const response = await fetch("/api/stats");
    const responseData = await response.json();

    // Safely assign statistic values with zero fallbacks
    let indexedFilesCount = 0;
    if (responseData.files_indexed !== undefined && responseData.files_indexed !== null) {
      indexedFilesCount = responseData.files_indexed;
    }

    let documentChunksCount = 0;
    if (responseData.document_chunks !== undefined && responseData.document_chunks !== null) {
      documentChunksCount = responseData.document_chunks;
    }

    let indexedImagesCount = 0;
    if (responseData.images_indexed !== undefined && responseData.images_indexed !== null) {
      indexedImagesCount = responseData.images_indexed;
    }

    let indexedFacesCount = 0;
    if (responseData.faces_indexed !== undefined && responseData.faces_indexed !== null) {
      indexedFacesCount = responseData.faces_indexed;
    }

    // Update DOM elements with values
    const statFilesElement = document.getElementById("stat-files");
    const statChunksElement = document.getElementById("stat-chunks");
    const statImagesElement = document.getElementById("stat-images");
    const statFacesElement = document.getElementById("stat-faces");

    if (statFilesElement) statFilesElement.textContent = indexedFilesCount;
    if (statChunksElement) statChunksElement.textContent = documentChunksCount;
    if (statImagesElement) statImagesElement.textContent = indexedImagesCount;
    if (statFacesElement) statFacesElement.textContent = indexedFacesCount;

  } catch (error) {
    console.error("Error loading system stats from server:", error);
  }
}

/**
 * Fetches recent indexed files from the API and displays up to 4 items on the dashboard.
 */
async function loadRecentFiles() {
  const recentFilesListElement = document.getElementById("recent-files-list");
  if (!recentFilesListElement) return;

  try {
    const response = await fetch("/api/files");
    const responseData = await response.json();

    const filesArray = responseData.files;

    // Handle empty file list state explicitly
    if (!filesArray || filesArray.length === 0) {
      recentFilesListElement.innerHTML = `<div style="color: var(--text-muted); font-size: 0.9rem;">No files indexed yet. Upload files to get started.</div>`;
      return;
    }

    // Process top 4 recent files using a readable step-by-step for loop
    let recentFilesHtml = "";
    const displayLimit = Math.min(filesArray.length, 4);

    for (let index = 0; index < displayLimit; index++) {
      const fileItem = filesArray[index];
      const iconSymbol = getFileIcon(fileItem.file_type);

      recentFilesHtml += `
        <div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 14px; background: rgba(255,255,255,0.03); border: var(--border-glass); border-radius: var(--radius-sm);">
          <div style="display: flex; align-items: center; gap: 10px;">
            <span style="font-size: 1.2rem;">${iconSymbol}</span>
            <div>
              <div style="font-weight: 600; font-size: 0.88rem; color: var(--text-primary);">${escapeHtml(fileItem.filename)}</div>
              <div style="font-size: 0.75rem; color: var(--text-muted);">${escapeHtml(fileItem.path)}</div>
            </div>
          </div>
          <span style="font-size: 0.75rem; background: rgba(0, 242, 254, 0.15); color: var(--accent-cyan); padding: 2px 8px; border-radius: 10px; font-weight: 600;">INDEXED</span>
        </div>
      `;
    }

    recentFilesListElement.innerHTML = recentFilesHtml;

  } catch (error) {
    console.error("Error loading recent files from server:", error);
  }
}

/**
 * Returns a user-friendly emoji icon corresponding to a given file type.
 * @param {string} fileType - The extension or category of file (e.g. 'image', 'pdf')
 */
function getFileIcon(fileType) {
  if (!fileType) {
    return "📁";
  }

  const lowercaseType = fileType.toLowerCase();

  switch (lowercaseType) {
    case "image":
      return "📸";
    case "audio":
      return "🎙️";
    case "video":
      return "🎬";
    case "pdf":
      return "📄";
    case "docx":
      return "📝";
    case "xlsx":
      return "📊";
    default:
      return "📁";
  }
}


/* ==========================================================================
   AI CHAT & MULTIMODAL RAG QUERY EXECUTION
   ========================================================================== */

/**
 * Executes an AI chat query, displays loading state, calls backend API, and updates conversation thread.
 * @param {string} questionText - The question submitted by the user
 */
async function executeChatQuery(questionText) {
  const chatThread = document.getElementById("chat-thread");
  if (!chatThread) return;

  // Step 1: Append User Question Message Bubble
  const userRow = document.createElement("div");
  userRow.className = "msg-row user";
  userRow.innerHTML = `
    <div class="msg-avatar">R</div>
    <div class="msg-bubble">${escapeHtml(questionText)}</div>
  `;
  chatThread.appendChild(userRow);

  // Step 2: Append AI Loading / Thinking Spinner Bubble
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

  // Scroll smoothly to bottom of chat
  chatThread.scrollTop = chatThread.scrollHeight;

  // Step 3: Make backend API call to process RAG chat query
  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        question: questionText
      })
    });

    const responseData = await response.json();

    // Step 4: Safely extract answer text and handle different response structures
    let rawAnswerText = "";
    let sourcesList = [];
    let chunksList = [];

    if (responseData && typeof responseData === "object") {
      if (Array.isArray(responseData)) {
        // If server returned tuple array [answer, sources, num_chunks]
        if (responseData.length > 0 && responseData[0]) {
          rawAnswerText = responseData[0];
        }
        if (responseData.length > 1 && Array.isArray(responseData[1])) {
          sourcesList = responseData[1];
        }
      } else {
        // Standard structured dictionary response
        if (responseData.answer) {
          rawAnswerText = responseData.answer;
        } else if (responseData.detail) {
          rawAnswerText = responseData.detail;
        } else if (responseData.message) {
          rawAnswerText = responseData.message;
        }

        if (Array.isArray(responseData.sources)) {
          sourcesList = responseData.sources;
        }
        if (Array.isArray(responseData.chunks)) {
          chunksList = responseData.chunks;
        }
      }
    }

    if (!rawAnswerText || typeof rawAnswerText !== "string") {
      rawAnswerText = "No answer content was returned by the server.";
    }

    // Step 5: Parse answer markdown using marked library if available
    let parsedAnswerText = rawAnswerText;
    if (window.marked && typeof window.marked.parse === "function") {
      parsedAnswerText = window.marked.parse(rawAnswerText);
    }

    // Step 5: Format retrieved media images HTML if present in response
    let mediaGalleryHtml = "";
    if (responseData.images && responseData.images.length > 0) {
      mediaGalleryHtml += `<div class="media-preview-container">`;

      for (let index = 0; index < responseData.images.length; index++) {
        const imageItem = responseData.images[index];
        let imagePathStr = imageItem;
        if (typeof imageItem === "object" && imageItem !== null && imageItem.path) {
          imagePathStr = imageItem.path;
        }

        const encodedMediaPath = encodeURIComponent(imagePathStr);
        const mediaUrl = "/api/media/" + encodedMediaPath;
        const filenameLabel = getBasename(imagePathStr);

        mediaGalleryHtml += `
          <div class="media-preview-card">
            <img src="${mediaUrl}" alt="Retrieved Image" onclick="openImageLightbox('${mediaUrl}')">
            <div style="padding: 8px; font-size: 0.75rem; color: var(--text-muted); text-align: center;">${escapeHtml(filenameLabel)}</div>
          </div>
        `;
      }

      mediaGalleryHtml += `</div>`;
    }

    // Step 6: Update AI Loading Bubble content with final result
    const messageBubbleElement = aiRow.querySelector(".msg-bubble");
    if (messageBubbleElement) {
      messageBubbleElement.innerHTML = `
        <div>${parsedAnswerText}</div>
        ${mediaGalleryHtml}
      `;
    }

    // Step 7: Update right sidebar RAG inspector with source document references
    updateRAGInspector(sourcesList, chunksList);

  } catch (error) {
    // Explicit Error Handling State
    const messageBubbleElement = aiRow.querySelector(".msg-bubble");
    if (messageBubbleElement) {
      messageBubbleElement.innerHTML = `
        <div style="color: var(--accent-rose);">Failed to process query: ${escapeHtml(error.message)}</div>
      `;
    }
  }

  // Final scroll to keep newest content visible
  chatThread.scrollTop = chatThread.scrollHeight;
}

/**
 * Renders source documents and chunk citations in the RAG inspector sidebar panel.
 * @param {Array} sources - List of source file paths retrieved
 * @param {Array} chunks - List of text chunks retrieved
 */
function updateRAGInspector(sources, chunks) {
  const inspectorListElement = document.getElementById("inspector-sources-list");
  if (!inspectorListElement) return;

  // Handle empty state explicitly
  if (!sources || sources.length === 0) {
    inspectorListElement.innerHTML = `<div style="color: var(--text-muted); font-size: 0.82rem;">No relevant source documents retrieved.</div>`;
    return;
  }

  // Construct source list HTML using a readable for loop
  let sourcesHtml = "";
  for (let index = 0; index < sources.length; index++) {
    const sourceFilePath = sources[index];
    const sourceFilename = getBasename(sourceFilePath);
    const itemNumber = index + 1;

    sourcesHtml += `
      <div class="source-card">
        <div class="source-file">${itemNumber}. ${escapeHtml(sourceFilename)}</div>
        <div style="font-size: 0.75rem; color: var(--text-muted); word-break: break-all;">${escapeHtml(sourceFilePath)}</div>
      </div>
    `;
  }

  inspectorListElement.innerHTML = sourcesHtml;
}


/* ==========================================================================
   IMAGE GALLERY & FACE MEMORY REGISTRY
   ========================================================================== */

/**
 * Loads indexed images into grid view on the Images tab page.
 */
async function loadImagesGallery() {
  const imagesGridElement = document.getElementById("images-grid");
  if (!imagesGridElement) return;

  // Set explicit loading state UI
  imagesGridElement.innerHTML = `<div style="color: var(--text-muted);">Loading image library...</div>`;

  try {
    const response = await fetch("/api/files");
    const responseData = await response.json();

    const allFilesArray = responseData.files || [];
    const imageFilesArray = [];

    // Filter image files explicitly
    for (let index = 0; index < allFilesArray.length; index++) {
      const fileItem = allFilesArray[index];
      if (fileItem.file_type === "image") {
        imageFilesArray.push(fileItem);
      }
    }

    // Handle empty state explicitly
    if (imageFilesArray.length === 0) {
      imagesGridElement.innerHTML = `<div style="color: var(--text-muted);">No images indexed yet.</div>`;
      return;
    }

    // Build image cards HTML
    let galleryHtml = "";
    for (let index = 0; index < imageFilesArray.length; index++) {
      const imageItem = imageFilesArray[index];
      const encodedPath = encodeURIComponent(imageItem.path);
      const imageUrl = "/api/media/" + encodedPath;

      galleryHtml += `
        <div class="gallery-card">
          <img class="gallery-thumb" src="${imageUrl}" alt="${escapeHtml(imageItem.filename)}" onclick="openImageLightbox('${imageUrl}')">
          <div class="gallery-info">
            <div class="gallery-title">${escapeHtml(imageItem.filename)}</div>
            <div class="gallery-meta">
              <span>Image Document</span>
              <span style="color: var(--accent-cyan); cursor: pointer;" onclick="quickPrompt('Analyze this image: ${escapeHtml(imageItem.filename)}')">Inspect</span>
            </div>
          </div>
        </div>
      `;
    }

    imagesGridElement.innerHTML = galleryHtml;

  } catch (error) {
    imagesGridElement.innerHTML = `<div style="color: var(--accent-rose);">Failed to load images from server.</div>`;
  }
}

/**
 * Helper to execute an image-based chat query.
 * @param {string} searchQueryText - Image search text
 */
async function executeImageSearch(searchQueryText) {
  switchTab("chat");
  executeChatQuery(searchQueryText);
}

/**
 * Loads registered face vectors and user reference photos into Face Memory tab.
 */
async function loadFaceMemory() {
  const faceContainerElement = document.getElementById("faces-list-container");
  if (!faceContainerElement) return;

  // Set explicit loading state UI
  faceContainerElement.innerHTML = `<div style="color: var(--text-muted);">Loading face memory registry...</div>`;

  try {
    const response = await fetch("/api/faces");
    const responseData = await response.json();

    const facesArray = responseData.faces;

    // Handle empty state explicitly
    if (!facesArray || facesArray.length === 0) {
      faceContainerElement.innerHTML = `<div style="color: var(--text-muted);">No face memories registered yet. Click "Register Face Memory" above to add a person.</div>`;
      return;
    }

    // Build face memory gallery cards HTML
    let facesHtml = "";
    for (let index = 0; index < facesArray.length; index++) {
      const faceItem = facesArray[index];
      const encodedImagePath = encodeURIComponent(faceItem.image_path);
      const faceImageUrl = "/api/media/" + encodedImagePath;

      facesHtml += `
        <div class="gallery-card">
          <img class="gallery-thumb" src="${faceImageUrl}" alt="${escapeHtml(faceItem.person_name)}">
          <div class="gallery-info">
            <div class="gallery-title" style="color: var(--accent-cyan); font-size: 1rem;">👤 ${escapeHtml(faceItem.person_name)}</div>
            <div class="gallery-meta" style="margin-top: 6px;">
              <span>${escapeHtml(faceItem.filename)}</span>
              <span style="background: rgba(0, 242, 254, 0.15); padding: 2px 6px; border-radius: 4px; font-weight: 600;">512D Vector</span>
            </div>
          </div>
        </div>
      `;
    }

    faceContainerElement.innerHTML = facesHtml;

  } catch (error) {
    faceContainerElement.innerHTML = `<div style="color: var(--accent-rose);">Failed to load face registry.</div>`;
  }
}

/**
 * Opens modal dialog for registering a new person's face memory.
 */
function openRegisterFaceModal() {
  const modalElement = document.getElementById("face-modal");
  if (modalElement) {
    modalElement.classList.add("active");
  }
}

/**
 * Closes modal dialog for face registration.
 */
function closeRegisterFaceModal() {
  const modalElement = document.getElementById("face-modal");
  if (modalElement) {
    modalElement.classList.remove("active");
  }
}

/**
 * Submits face registration form to backend server API.
 */
async function submitFaceRegistration() {
  const nameInputElement = document.getElementById("modal-person-name");
  const fileInputElement = document.getElementById("modal-face-file");

  if (!nameInputElement || !fileInputElement) return;

  const personNameValue = nameInputElement.value.trim();
  const selectedFiles = fileInputElement.files;

  // Step 1: Input Validation
  if (personNameValue === "" || selectedFiles.length === 0) {
    alert("Please enter a person name and choose a reference photo.");
    return;
  }

  // Step 2: Prepare Multipart Form Data Payload
  const registrationFormData = new FormData();
  registrationFormData.append("person_name", personNameValue);
  registrationFormData.append("file", selectedFiles[0]);

  // Step 3: POST payload to server
  try {
    const response = await fetch("/api/faces/register", {
      method: "POST",
      body: registrationFormData
    });

    const responseResult = await response.json();

    // Step 4: Handle response status explicitly
    if (responseResult.status === "SUCCESS") {
      closeRegisterFaceModal();
      loadFaceMemory();
      loadSystemStats();
      alert("Successfully registered face memory for " + personNameValue + "!");
    } else {
      alert("Registration failed: " + responseResult.message);
    }
  } catch (error) {
    alert("Error registering face: " + error.message);
  }
}


/* ==========================================================================
   AUDIO & VIDEO MEDIA LIBRARIES
   ========================================================================== */

/**
 * Fetches and displays processed audio files with speech transcripts.
 */
async function loadAudioLibrary() {
  const audioContainerElement = document.getElementById("audio-list-container");
  if (!audioContainerElement) return;

  audioContainerElement.innerHTML = `<div style="color: var(--text-muted);">Loading audio library...</div>`;

  try {
    const response = await fetch("/api/audio");
    const responseData = await response.json();

    const audioFilesArray = responseData.audio;

    if (!audioFilesArray || audioFilesArray.length === 0) {
      audioContainerElement.innerHTML = `<div style="color: var(--text-muted);">No audio files processed yet.</div>`;
      return;
    }

    let audioCardsHtml = "";
    for (let index = 0; index < audioFilesArray.length; index++) {
      const audioItem = audioFilesArray[index];
      const encodedAudioPath = encodeURIComponent(audioItem.path);
      const audioMediaUrl = "/api/media/" + encodedAudioPath;

      // Join transcript lines cleanly into one readable paragraph
      let transcriptText = "";
      if (audioItem.transcripts && audioItem.transcripts.length > 0) {
        transcriptText = audioItem.transcripts.join(" ");
      }

      audioCardsHtml += `
        <div class="audio-card">
          <div class="audio-header">
            <div style="font-weight: 600; font-size: 1.05rem; color: var(--accent-cyan);">🎙️ ${escapeHtml(audioItem.filename)}</div>
            <span style="font-size: 0.75rem; background: rgba(16, 185, 129, 0.15); color: var(--accent-emerald); padding: 2px 8px; border-radius: 10px; font-weight: 600;">SROTA ASR 16kHz</span>
          </div>

          <div class="waveform-placeholder">
            <div class="wave-bar"></div>
            <div class="wave-bar"></div>
            <div class="wave-bar"></div>
            <div class="wave-bar"></div>
            <div class="wave-bar"></div>
          </div>

          <audio controls src="${audioMediaUrl}"></audio>

          <div style="background: rgba(0,0,0,0.3); border: var(--border-glass); border-radius: var(--radius-sm); padding: 14px; margin-top: 8px;">
            <div style="font-size: 0.8rem; color: var(--text-muted); font-weight: 600; margin-bottom: 6px;">SEARCHABLE TRANSCRIPT:</div>
            <div style="font-size: 0.9rem; line-height: 1.5; color: var(--text-secondary);">${escapeHtml(transcriptText)}</div>
          </div>
        </div>
      `;
    }

    audioContainerElement.innerHTML = audioCardsHtml;

  } catch (error) {
    audioContainerElement.innerHTML = `<div style="color: var(--accent-rose);">Failed to load audio files.</div>`;
  }
}

/**
 * Fetches and displays processed video files in video library.
 */
async function loadVideoLibrary() {
  const videoContainerElement = document.getElementById("video-list-container");
  if (!videoContainerElement) return;

  videoContainerElement.innerHTML = `<div style="color: var(--text-muted);">Loading video library...</div>`;

  try {
    const response = await fetch("/api/video");
    const responseData = await response.json();

    const videoFilesArray = responseData.videos;

    if (!videoFilesArray || videoFilesArray.length === 0) {
      videoContainerElement.innerHTML = `<div style="color: var(--text-muted);">No videos processed yet.</div>`;
      return;
    }

    let videoCardsHtml = "";
    for (let index = 0; index < videoFilesArray.length; index++) {
      const videoItem = videoFilesArray[index];
      const encodedVideoPath = encodeURIComponent(videoItem.path);
      const videoMediaUrl = "/api/media/" + encodedVideoPath;

      videoCardsHtml += `
        <div class="gallery-card" style="grid-column: span 2;">
          <video controls src="${videoMediaUrl}"></video>
          <div class="gallery-info">
            <div class="gallery-title" style="color: var(--accent-cyan);">🎬 ${escapeHtml(videoItem.filename)}</div>
            <div class="gallery-meta" style="margin-top: 8px;">
              <span>Extracted Keyframes & ASR Track</span>
              <span style="color: var(--accent-emerald); font-weight: 600;">PROCESSED</span>
            </div>
          </div>
        </div>
      `;
    }

    videoContainerElement.innerHTML = videoCardsHtml;

  } catch (error) {
    videoContainerElement.innerHTML = `<div style="color: var(--accent-rose);">Failed to load video library.</div>`;
  }
}

/**
 * Fetches all indexed files for display in the File Explorer tab page.
 */
async function loadFilesExplorer() {
  const filesContainerElement = document.getElementById("files-grid-container");
  if (!filesContainerElement) return;

  filesContainerElement.innerHTML = `<div style="color: var(--text-muted);">Loading files...</div>`;

  try {
    const response = await fetch("/api/files");
    const responseData = await response.json();

    const allFilesArray = responseData.files;

    if (!allFilesArray || allFilesArray.length === 0) {
      filesContainerElement.innerHTML = `<div style="color: var(--text-muted);">No indexed files found.</div>`;
      return;
    }

    let filesGridHtml = "";
    for (let index = 0; index < allFilesArray.length; index++) {
      const fileItem = allFilesArray[index];
      const iconSymbol = getFileIcon(fileItem.file_type);

      filesGridHtml += `
        <div class="gallery-card">
          <div style="height: 120px; background: rgba(255,255,255,0.03); display: flex; align-items: center; justify-content: center; font-size: 3rem;">
            ${iconSymbol}
          </div>
          <div class="gallery-info">
            <div class="gallery-title">${escapeHtml(fileItem.filename)}</div>
            <div class="gallery-meta">
              <span style="font-family: var(--font-code); font-size: 0.7rem;">${escapeHtml(fileItem.file_type.toUpperCase())}</span>
              <span style="color: var(--accent-cyan); font-weight: 600;">INDEXED</span>
            </div>
          </div>
        </div>
      `;
    }

    filesContainerElement.innerHTML = filesGridHtml;

  } catch (error) {
    filesContainerElement.innerHTML = `<div style="color: var(--accent-rose);">Failed to load files.</div>`;
  }
}

/**
 * Handles new file upload from the file upload input element.
 * @param {HTMLInputElement} inputElement - File input element
 */
async function handleFileUpload(inputElement) {
  if (!inputElement || !inputElement.files || inputElement.files.length === 0) {
    return;
  }

  const selectedFile = inputElement.files[0];

  // Construct form data for upload
  const uploadFormData = new FormData();
  uploadFormData.append("file", selectedFile);

  alert("Uploading and indexing file: " + selectedFile.name + "...");

  try {
    const response = await fetch("/api/upload", {
      method: "POST",
      body: uploadFormData
    });

    const responseData = await response.json();

    if (responseData.status === "SUCCESS") {
      alert("File " + selectedFile.name + " uploaded and indexed successfully!");
      loadFilesExplorer();
      loadSystemStats();
    } else {
      alert("Upload failed: " + responseData.message);
    }
  } catch (error) {
    alert("Error uploading file: " + error.message);
  }
}


/* ==========================================================================
   UTILITY & HELPER FUNCTIONS
   ========================================================================== */

/**
 * Extracts and returns the simple file name from a given file path string.
 * @param {string} pathString - The full path of the file
 */
function getBasename(pathString) {
  if (!pathString) {
    return "";
  }
  const parts = pathString.split(/[\\/]/);
  return parts.pop();
}

/**
 * Escapes unsafe HTML characters to prevent XSS issues when inserting text.
 * @param {string} textString - Raw string text
 */
function escapeHtml(textString) {
  if (!textString) {
    return "";
  }
  return textString
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

/**
 * Opens full resolution image media preview in a new browser window/tab.
 * @param {string} mediaUrl - Full URL to image media
 */
function openImageLightbox(mediaUrl) {
  window.open(mediaUrl, "_blank");
}
