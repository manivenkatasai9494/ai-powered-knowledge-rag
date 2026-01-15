// static/script.js
async function send() {
    const inputEl = document.getElementById("input");
    const roleEl = document.getElementById("role");
    const chatEl = document.getElementById("chat");
  
    const text = inputEl.value.trim();
    const role = roleEl.value;
  
    if (!text) return;
  
    // Show user message bubble
    appendMessage(chatEl, text, "user");
    inputEl.value = "";
  
    try {
      const res = await fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: text, role })
      });
  
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        appendMessage(
          chatEl,
          err.error || `Something went wrong (HTTP ${res.status})`,
          "bot"
        );
        return;
      }
  
      const data = await res.json();
      appendMessage(chatEl, data.answer || "No answer received from server.", "bot");
    } catch (e) {
      appendMessage(
        chatEl,
        "Network error: unable to reach the server. Please try again.",
        "bot"
      );
      console.error(e);
    }
  }
  
  // Helper to create message bubbles
  function appendMessage(container, text, type) {
    const div = document.createElement("div");
    div.className = `message ${type}`;
    div.textContent = text;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
  }
  
  // Allow Enter key to send
  document.addEventListener("DOMContentLoaded", () => {
    const inputEl = document.getElementById("input");
    inputEl.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        send();
      }
    });
  });