async function send() {
    const text = document.getElementById("input").value;
    const role = document.getElementById("role").value;
    const chat = document.getElementById("chat");

    if (!text) return;

    chat.innerHTML += `<div class="user">You (${role}): ${text}</div>`;
    document.getElementById("input").value = "";

    const res = await fetch("http://127.0.0.1:5000/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: text, role })
    });

    const data = await res.json();
    chat.innerHTML += `<div class="bot">Bot: ${data.answer}</div>`;
}
