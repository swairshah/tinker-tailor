"""Modal FastAPI chat playground for the hosted Tinker model.

Required Modal secrets:
  1) tinker-api-key      -> TINKER_API_KEY=...
  2) tinker-model-config -> TINKER_MODEL_PATH=tinker://.../sampler_weights/final

Deploy:
  modal deploy apps/modal/playground.py
"""

from __future__ import annotations

import os

import modal

app = modal.App("qwen35-4b-playground")
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "openai>=2.28.0,<3",
    "fastapi>=0.116,<1",
)

_tinker_secret = modal.Secret.from_name("tinker-api-key", required_keys=["TINKER_API_KEY"])
_model_secret = modal.Secret.from_name(
    "tinker-model-config",
    required_keys=["TINKER_MODEL_PATH"],
)

TINKER_OAI_BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"


STYLE = r"""
*,*::before,*::after{margin:0;padding:0;box-sizing:border-box}
:root{
  --bg:rgb(15,17,21);--bg-secondary:rgb(26,29,35);
  --border:rgba(107,114,128,0.2);--text:rgb(201,204,209);
  --text-bright:rgb(229,231,235);--text-muted:rgb(107,114,128);
  --text-dimmed:rgb(75,85,99);--accent:rgb(245,158,11);
  --green:rgb(16,185,129);--blue:rgb(96,165,250);
  --font-mono:'JetBrains Mono',monospace;
}
[data-theme="light"],[data-theme="light"] body{
  --bg:#ffffff;--bg-secondary:#f9fafb;--border:#e5e7eb;
  --text:#374151;--text-bright:#111827;--text-muted:#6b7280;
  --text-dimmed:#9ca3af;--accent:#d97706;--green:#059669;--blue:#2563eb;
}
html{font-size:15px;line-height:1.6;-webkit-font-smoothing:antialiased}
body{background:var(--bg);color:var(--text);font-family:var(--font-mono);min-height:100vh;display:flex;flex-direction:column}
nav{padding:20px 28px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid var(--border)}
.nav-brand{color:var(--text-bright);text-decoration:none;font-weight:500}
.nav-brand:hover{color:var(--accent)}
.nav-links{display:flex;align-items:center;gap:18px}
.nav-link{color:var(--text-muted);text-decoration:none;font-size:13px}
.nav-link:hover{color:var(--accent)}
.theme-toggle{background:none;border:none;color:var(--text-dimmed);font-size:14px;cursor:pointer;font-family:var(--font-mono)}
.theme-toggle:hover{color:var(--text-muted)}

.chat-wrap{max-width:820px;margin:0 auto;padding:18px 24px 26px;flex:1;display:flex;flex-direction:column;width:100%}
.model-tag{color:var(--text-dimmed);font-size:12px;margin-bottom:14px;padding:6px 10px;background:var(--bg-secondary);border:1px solid var(--border);border-radius:6px;display:inline-block}
.model-tag code{color:var(--accent)}

#messages{flex:1;overflow-y:auto;padding-bottom:16px;scroll-behavior:smooth}
.msg{margin-bottom:18px;padding-bottom:18px;border-bottom:1px solid var(--border)}
.msg:last-child{border-bottom:none}
.msg-role{font-size:12px;font-weight:600;letter-spacing:.05em;text-transform:uppercase;margin-bottom:6px}
.msg-role.user{color:var(--accent)}
.msg-role.assistant{color:var(--green)}
.msg-body{white-space:pre-wrap;word-wrap:break-word;color:var(--text-muted)}
.msg-body code{font-size:.9em;background:var(--bg-secondary);padding:2px 6px;border-radius:3px;color:var(--text-bright)}
.msg-thinking{padding:10px 12px;background:var(--bg-secondary);border:1px solid var(--border);border-radius:6px;margin-bottom:8px;font-size:13px;color:var(--text-dimmed)}
.msg-thinking summary{cursor:pointer;color:var(--blue);font-size:12px;font-weight:600;letter-spacing:.05em;text-transform:uppercase}

.typing{color:var(--text-dimmed);font-size:13px;animation:pulse 1.4s infinite}
@keyframes pulse{0%,100%{opacity:.45}50%{opacity:1}}

.input-row{display:flex;gap:10px;padding-top:14px;border-top:1px solid var(--border)}
#input{flex:1;background:var(--bg-secondary);border:1px solid var(--border);border-radius:6px;color:var(--text-bright);font-family:var(--font-mono);font-size:14px;padding:12px 14px;resize:none;outline:none;min-height:48px;max-height:220px;line-height:1.5}
#input:focus{border-color:var(--accent)}
#input::placeholder{color:var(--text-dimmed)}
#send{background:var(--accent);color:#111;border:none;border-radius:6px;padding:12px 18px;font-family:var(--font-mono);font-size:14px;font-weight:600;cursor:pointer;white-space:nowrap}
#send:hover{opacity:.9}
#send:disabled{opacity:.45;cursor:not-allowed}

.settings{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;margin-bottom:12px}
.settings label{display:block;color:var(--text-muted);font-size:12px;margin-bottom:4px}
.settings input{width:100%;background:var(--bg-secondary);border:1px solid var(--border);border-radius:6px;color:var(--text);padding:8px 10px;font-family:var(--font-mono)}

.empty-state{flex:1;display:flex;align-items:center;justify-content:center;flex-direction:column;gap:10px;color:var(--text-dimmed);padding:24px}
.empty-state h2{color:var(--text-muted);font-size:16px;font-weight:600}
.empty-state p{font-size:13px}

@media (max-width:720px){
  .settings{grid-template-columns:1fr}
  nav{padding:16px 16px}
  .chat-wrap{padding:14px 12px 18px}
}
"""


HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>llm-tuner · chat playground</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet" />
  <style>__STYLE__</style>
  <script>
    (function() {
      var t = localStorage.getItem('theme');
      if (t === 'light') document.documentElement.setAttribute('data-theme', 'light');
    })();
  </script>
</head>
<body>
  <nav>
    <a href="/" class="nav-brand">llm-tuner · chat</a>
    <div class="nav-links">
      <a href="/health" class="nav-link">health</a>
      <button class="theme-toggle" onclick="toggleTheme()" aria-label="Toggle theme">☀</button>
    </div>
  </nav>

  <div class="chat-wrap">
    <div class="model-tag">model: <code id="model_tag">loading...</code> · provider: tinker</div>

    <div class="settings">
      <div>
        <label for="max_tokens">max_tokens</label>
        <input id="max_tokens" type="number" value="512" min="1" max="4096" />
      </div>
      <div>
        <label for="temperature">temperature</label>
        <input id="temperature" type="number" value="0.7" step="0.1" min="0" max="2" />
      </div>
      <div>
        <label for="top_p">top_p</label>
        <input id="top_p" type="number" value="0.95" step="0.01" min="0" max="1" />
      </div>
    </div>

    <div id="messages">
      <div class="empty-state" id="empty">
        <h2>ask anything</h2>
        <p>chat with your tuned model, inspect reasoning tags, iterate quickly</p>
      </div>
    </div>

    <div class="input-row">
      <textarea id="input" rows="1" placeholder="Type your message..."></textarea>
      <button id="send" onclick="sendMessage()">send</button>
    </div>
  </div>

  <script>
    const messages = [];
    const msgEl = document.getElementById('messages');
    const inputEl = document.getElementById('input');
    const sendBtn = document.getElementById('send');

    function toggleTheme() {
      const html = document.documentElement;
      const btn = document.querySelector('.theme-toggle');
      if (html.getAttribute('data-theme') === 'light') {
        html.removeAttribute('data-theme');
        btn.textContent = '☀';
        localStorage.setItem('theme', 'dark');
      } else {
        html.setAttribute('data-theme', 'light');
        btn.textContent = '☾';
        localStorage.setItem('theme', 'light');
      }
    }

    (function syncThemeButton() {
      const btn = document.querySelector('.theme-toggle');
      if (!btn) return;
      btn.textContent = localStorage.getItem('theme') === 'light' ? '☾' : '☀';
    })();

    function esc(s) {
      const d = document.createElement('div');
      d.textContent = s;
      return d.innerHTML;
    }

    function renderMessage(role, content, thinking = '') {
      const empty = document.getElementById('empty');
      if (empty) empty.remove();

      const d = document.createElement('div');
      d.className = 'msg';
      let html = `<div class="msg-role ${role}">${role}</div>`;
      if (thinking) {
        html += `<details class="msg-thinking"><summary>thinking</summary><div class="msg-body">${esc(thinking)}</div></details>`;
      }
      html += `<div class="msg-body">${esc(content)}</div>`;
      d.innerHTML = html;
      msgEl.appendChild(d);
      msgEl.scrollTop = msgEl.scrollHeight;
    }

    function showTyping() {
      const d = document.createElement('div');
      d.className = 'msg';
      d.id = 'typing';
      d.innerHTML = '<div class="typing">thinking...</div>';
      msgEl.appendChild(d);
      msgEl.scrollTop = msgEl.scrollHeight;
    }

    function hideTyping() {
      const t = document.getElementById('typing');
      if (t) t.remove();
    }

    async function sendMessage() {
      const text = inputEl.value.trim();
      if (!text) return;

      inputEl.value = '';
      inputEl.style.height = 'auto';

      messages.push({ role: 'user', content: text });
      renderMessage('user', text);

      sendBtn.disabled = true;
      showTyping();

      const payload = {
        messages,
        max_tokens: Number(document.getElementById('max_tokens').value || 512),
        temperature: Number(document.getElementById('temperature').value || 0.7),
        top_p: Number(document.getElementById('top_p').value || 0.95),
      };

      try {
        const res = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });

        const data = await res.json();
        if (!res.ok) {
          throw new Error(data.error || data.detail || (`HTTP ${res.status}`));
        }

        hideTyping();

        const full = String(data.completion || '');
        document.getElementById('model_tag').textContent = data.model_path || 'unknown';

        let thinking = '';
        let answer = full;
        const match = full.match(/<think>([\s\S]*?)<\/think>/i);
        if (match) {
          thinking = match[1].trim();
          answer = full.replace(/<think>[\s\S]*?<\/think>/ig, '').trim();
        }

        messages.push({ role: 'assistant', content: full });
        renderMessage('assistant', answer || full, thinking);
      } catch (err) {
        hideTyping();
        renderMessage('assistant', 'Error: ' + (err?.message || String(err)));
      } finally {
        sendBtn.disabled = false;
        inputEl.focus();
      }
    }

    inputEl.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });

    inputEl.addEventListener('input', function () {
      this.style.height = 'auto';
      this.style.height = Math.min(this.scrollHeight, 220) + 'px';
    });

    (async function loadHealth() {
      try {
        const res = await fetch('/health');
        const data = await res.json();
        if (data.model_path) {
          document.getElementById('model_tag').textContent = data.model_path;
        }
      } catch (_) {
        // no-op
      }
    })();
  </script>
</body>
</html>
"""


@app.function(
    image=image,
    cpu=1,
    timeout=120,
    secrets=[_tinker_secret, _model_secret],
    scaledown_window=300,
)
@modal.asgi_app()
def web_app():
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import HTMLResponse
    from openai import OpenAI

    api = FastAPI(title="llm-tuner-playground")
    client = OpenAI(base_url=TINKER_OAI_BASE_URL, api_key=os.environ["TINKER_API_KEY"])

    @api.get("/", response_class=HTMLResponse)
    async def index():
        return HTMLResponse(HTML.replace("__STYLE__", STYLE))

    @api.get("/health")
    async def health():
        return {
            "status": "ok",
            "model_path": os.environ.get("TINKER_MODEL_PATH", ""),
            "provider": "tinker-openai-compatible",
        }

    def _complete(
        *,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> tuple[str, str]:
        model_path = os.environ["TINKER_MODEL_PATH"]
        try:
            resp = client.chat.completions.create(
                model=model_path,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Generation failed: {e}") from e

        completion = resp.choices[0].message.content or ""
        return completion, model_path

    @api.post("/api/chat")
    async def chat(request: Request):
        payload = await request.json()
        raw_messages = payload.get("messages")
        if not isinstance(raw_messages, list) or not raw_messages:
            raise HTTPException(status_code=400, detail="messages[] is required")

        messages: list[dict[str, str]] = []
        for row in raw_messages:
            if not isinstance(row, dict):
                continue
            role = str(row.get("role", "")).strip()
            content = str(row.get("content", ""))
            if role not in {"system", "user", "assistant"}:
                continue
            messages.append({"role": role, "content": content})

        if not messages:
            raise HTTPException(status_code=400, detail="No valid messages found")

        max_tokens = max(1, min(4096, int(payload.get("max_tokens", 512))))
        temperature = float(payload.get("temperature", 0.7))
        top_p = float(payload.get("top_p", 0.95))

        completion, model_path = _complete(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        return {
            "completion": completion,
            "model_path": model_path,
            "provider": "tinker-openai-compatible",
        }

    # Backward-compatible single-prompt endpoint.
    @api.post("/api/generate")
    async def generate(request: Request):
        payload = await request.json()
        prompt = str(payload.get("prompt", "")).strip()
        if not prompt:
            raise HTTPException(status_code=400, detail="prompt is required")

        max_tokens = max(1, min(4096, int(payload.get("max_tokens", 512))))
        temperature = float(payload.get("temperature", 0.7))
        top_p = float(payload.get("top_p", 0.95))

        completion, model_path = _complete(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        return {
            "completions": [completion],
            "model_path": model_path,
            "provider": "tinker-openai-compatible",
        }

    return api
