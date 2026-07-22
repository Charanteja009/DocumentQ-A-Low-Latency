const API_BASE_URL =  "https://rag-backend-service-eert.onrender.com";

// Manage session_id
let sessionId = localStorage.getItem("rag_session_id");
if (!sessionId) {
  sessionId = "session_" + Math.random().toString(36).substring(2, 9);
  localStorage.setItem("rag_session_id", sessionId);
}

document.getElementById("displaySessionId").innerText = sessionId;

// DOM Elements
const chatHistory = document.getElementById("chatHistory");
const chatForm = document.getElementById("chatForm");
const promptInput = document.getElementById("promptInput");
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const fileLabelText = document.getElementById("fileLabelText");
const uploadBtn = document.getElementById("uploadBtn");
const uploadStatus = document.getElementById("uploadStatus");
const newSessionBtn = document.getElementById("newSessionBtn");

// Auto-expand Textarea
promptInput.addEventListener("input", () => {
  promptInput.style.height = "auto";
  promptInput.style.height = promptInput.scrollHeight + "px";
});

// Load Chat History
async function loadHistory() {
  try {
    const res = await fetch(`${API_BASE_URL}/history/${sessionId}`);
    const data = await res.json();
    
    if (data.history && data.history.length > 0) {
      chatHistory.innerHTML = "";
      data.history.forEach(msg => {
        const [role, ...textParts] = msg.split(":");
        const text = textParts.join(":").trim();
        appendMessageRow(role.toLowerCase() === "user" ? "user" : "assistant", text);
      });
      scrollToBottom();
    }
  } catch (err) {
    console.error("Error loading history:", err);
  }
}

function appendMessageRow(sender, text = "") {
  const emptyState = chatHistory.querySelector(".empty-state");
  if (emptyState) emptyState.remove();

  const row = document.createElement("div");
  row.classList.add("message-row", sender);

  const avatar = document.createElement("div");
  avatar.classList.add("avatar");
  avatar.innerText = sender === "user" ? "U" : "AI";

  const bubble = document.createElement("div");
  bubble.classList.add("bubble");
  bubble.innerText = text;

  row.appendChild(avatar);
  row.appendChild(bubble);
  chatHistory.appendChild(row);

  scrollToBottom();
  return bubble;
}

function scrollToBottom() {
  chatHistory.scrollTop = chatHistory.scrollHeight;
}

// ----------------------------------------------------
// 📁 DRAG & DROP EVENT LISTENERS (FIXED)
// ----------------------------------------------------

// 1. Prevent default browser file-opening behavior
["dragenter", "dragover", "dragleave", "drop"].forEach(eventName => {
  dropzone.addEventListener(eventName, (e) => {
    e.preventDefault();
    e.stopPropagation();
  }, false);
});

// 2. Visual highlighting when dragging file over dropzone
["dragenter", "dragover"].forEach(eventName => {
  dropzone.addEventListener(eventName, () => {
    dropzone.style.borderColor = "var(--accent-warm)";
    dropzone.style.backgroundColor = "var(--accent-soft)";
  }, false);
});

["dragleave", "drop"].forEach(eventName => {
  dropzone.addEventListener(eventName, () => {
    dropzone.style.borderColor = "var(--border-subtle)";
    dropzone.style.backgroundColor = "var(--bg-card)";
  }, false);
});

// 3. Handle Dropped File
dropzone.addEventListener("drop", (e) => {
  const dt = e.dataTransfer;
  const files = dt.files;

  if (files && files.length > 0) {
    fileInput.files = files; // Assign dropped file to hidden file input
    handleFileSelected(files[0]);
  }
});

// 4. Click to browse file fallback
dropzone.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", () => {
  if (fileInput.files.length > 0) {
    handleFileSelected(fileInput.files[0]);
  }
});

function handleFileSelected(file) {
  fileLabelText.innerText = `📄 ${file.name}`;
  uploadBtn.disabled = false;
}

// 5. Ingest Document Action
uploadBtn.addEventListener("click", async () => {
  if (!fileInput.files.length) return;

  const formData = new FormData();
  formData.append("session_id", sessionId);
  formData.append("file", fileInput.files[0]);

  uploadStatus.className = "toast";
  uploadStatus.innerText = "Ingesting & Embedding...";
  uploadStatus.style.display = "block";

  try {
    const res = await fetch(`${API_BASE_URL}/upload`, {
      method: "POST",
      body: formData
    });

    const data = await res.json();
    if (res.ok) {
      uploadStatus.classList.add("success");
      uploadStatus.innerText = `✅ Ingested ${data.chunks_stored} chunks cleanly.`;
    } else {
      throw new Error(data.detail || "Upload failed");
    }
  } catch (err) {
    uploadStatus.classList.add("error");
    uploadStatus.innerText = `❌ ${err.message}`;
  }
});

// ----------------------------------------------------
// 💬 STREAM CHAT EVENT LISTENERS
// ----------------------------------------------------
chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const prompt = promptInput.value.trim();
  if (!prompt) return;

  appendMessageRow("user", prompt);
  promptInput.value = "";
  promptInput.style.height = "auto";

  const assistantBubble = appendMessageRow("assistant", "");

  try {
    const response = await fetch(`${API_BASE_URL}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, prompt: prompt })
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value, { stream: true });
      assistantBubble.innerText += chunk;
      scrollToBottom();
    }
  } catch (err) {
    assistantBubble.innerText = "Error streaming response.";
    console.error(err);
  }
});

// New Session Reset
newSessionBtn.addEventListener("click", () => {
  localStorage.removeItem("rag_session_id");
  location.reload();
});

// Load History on Boot
loadHistory();