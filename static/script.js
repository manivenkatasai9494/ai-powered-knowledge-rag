async function send() {
    const input = document.getElementById("input");
    const role = document.getElementById("role").value;
    const chat = document.getElementById("chat");
  
    if (!input.value) return;
  
    chat.innerHTML += `<div class="user">${input.value}</div>`;
  
    const res = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: input.value, role })
    });
  
    const data = await res.json();
    chat.innerHTML += `<div class="bot">${data.answer}</div>`;
    input.value = "";
  }
  