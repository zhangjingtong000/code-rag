/* ═══════════════════════════════════════════════
   ChatApp — core chat logic + sidebar tracking
   ═══════════════════════════════════════════════ */

var ChatApp = class ChatApp {
  constructor() {
    this.messages = [];
    this.streamHandler = null;
    this.isStreaming = false;
    this.currentAssistantMsg = null;
    this.msgIdCounter = 0;
    this.activeSidebarItem = null;

    this.els = {
      messages: document.getElementById('messages'),
      input: document.getElementById('input'),
      sendBtn: document.getElementById('send-btn'),
      statusDot: document.getElementById('status-dot'),
      statusText: document.getElementById('status-text'),
      sidebarList: document.getElementById('sidebar-list'),
    };

    this._init();
  }

  _init() {
    this.els.sendBtn.addEventListener('click', () => this.sendMessage());
    this.els.input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        this.sendMessage();
      }
    });
    this.els.input.addEventListener('input', () => this._autoResize());
    this._checkHealth();
  }

  // ── Health ────────────────────────────────────────

  async _checkHealth() {
    try {
      const resp = await fetch('/api/rag/health');
      const data = await resp.json();
      if (data.index_status === 'ready') {
        this._setStatus('ready', '就绪');
      } else if (data.index_status === 'building') {
        this._setStatus('building', '索引中...');
        this._pollIndexStatus();
      } else {
        this._setStatus('offline', '未构建索引');
      }
    } catch (e) {
      this._setStatus('offline', '服务未连接');
    }
  }

  async _pollIndexStatus() {
    try {
      const resp = await fetch('/api/rag/index/status');
      const data = await resp.json();
      if (data.status === 'building') {
        this._setStatus('building', data.progress || '索引中...');
        setTimeout(() => this._pollIndexStatus(), 2000);
      } else if (data.status === 'ready') {
        this._setStatus('ready', '就绪');
      } else {
        this._setStatus('offline', '未构建索引');
      }
    } catch (e) {
      this._setStatus('offline', '状态未知');
    }
  }

  _setStatus(state, text) {
    var dot = this.els.statusDot;
    dot.className = 'status-dot ' + state;
    this.els.statusText.textContent = text;
  }

  // ── Sidebar ────────────────────────────────────────

  _addToSidebar(question, userMsgId) {
    // Remove empty placeholder
    var emptyEl = this.els.sidebarList.querySelector('.sidebar-empty');
    if (emptyEl) emptyEl.remove();

    var btn = document.createElement('button');
    btn.className = 'sidebar-item';
    btn.textContent = question;
    btn.title = question;
    btn.setAttribute('data-msg-id', userMsgId);
    btn.addEventListener('click', () => this._scrollToMessage(userMsgId, btn));
    this.els.sidebarList.appendChild(btn);
    this.els.sidebarList.scrollTop = this.els.sidebarList.scrollHeight;
  }

  _scrollToMessage(msgId, sidebarBtn) {
    var msgEl = document.getElementById(msgId);
    if (msgEl) {
      msgEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
      // Highlight
      msgEl.style.animation = 'none';
      msgEl.offsetHeight;
      msgEl.style.animation = '';
      msgEl.style.boxShadow = '0 0 0 3px rgba(59,130,246,0.2)';
      setTimeout(function () { msgEl.style.boxShadow = ''; }, 1800);
    }
    // Update active sidebar item
    if (this.activeSidebarItem) {
      this.activeSidebarItem.classList.remove('active');
    }
    sidebarBtn.classList.add('active');
    this.activeSidebarItem = sidebarBtn;
  }

  // ── Send ───────────────────────────────────────────

  async sendMessage() {
    if (this.isStreaming) return;

    var text = this.els.input.value.trim();
    if (!text) return;

    this.els.input.value = '';
    this._autoResize();

    var userMsgId = 'msg-' + (++this.msgIdCounter);
    this.addMessage('user', text, false, userMsgId);
    this._addToSidebar(text, userMsgId);

    this.currentAssistantMsg = this.addMessage('assistant', '', true, 'msg-' + (++this.msgIdCounter));

    try {
      this.isStreaming = true;
      this.els.sendBtn.disabled = true;

      var resp = await fetch('/api/rag/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: text,
          chat_history: this._getChatHistory(),
        }),
      });

      if (!resp.ok) throw new Error('HTTP ' + resp.status);

      var data = await resp.json();
      this.streamHandler = new StreamHandler(data.stream_url, this);
      await this.streamHandler.start();

    } catch (err) {
      console.error('Send error:', err);
      this.showError('发送失败: ' + err.message);
    } finally {
      this.isStreaming = false;
      this.els.sendBtn.disabled = false;
      this.els.input.focus();
    }
  }

  _getChatHistory() {
    var history = [];
    var recent = this.messages.slice(-11, -1);
    for (var i = 0; i < recent.length; i++) {
      var msg = recent[i];
      if (msg.role !== 'system') {
        history.push({ role: msg.role, content: msg.raw || msg.content });
      }
    }
    return history;
  }

  // ── Messages ───────────────────────────────────────

  addMessage(role, content, isStreaming, msgId) {
    var msg = {
      role: role,
      content: content,
      raw: content,
      isStreaming: !!isStreaming,
      el: null,
      msgId: msgId || '',
    };

    this.messages.push(msg);

    var div = document.createElement('div');
    div.className = 'message ' + role;
    if (msgId) div.id = msgId;

    var bubble = document.createElement('div');
    bubble.className = 'msg-bubble';

    if (isStreaming) {
      bubble.classList.add('streaming-cursor');
      bubble.innerHTML = '';
    } else {
      bubble.innerHTML = window.renderMarkdown(content);
      this._postProcessCodeBlocks(bubble);
    }

    div.appendChild(bubble);
    this.els.messages.appendChild(div);
    msg.el = div;

    this._scrollToBottom();
    return msg;
  }

  // ── Streaming callbacks ────────────────────────────

  onMetadata(data) {
    console.log('Source files:', data.files);
  }

  appendStreamText(text) {
    if (!this.currentAssistantMsg) return;
    var bubble = this.currentAssistantMsg.el.querySelector('.msg-bubble');
    this.currentAssistantMsg.content += text;
    bubble.innerHTML = window.renderMarkdown(this.currentAssistantMsg.content);
    bubble.classList.add('streaming-cursor');
    this._postProcessCodeBlocks(bubble);
    this._scrollToBottom();
  }

  onStreamComplete(fullText) {
    if (!this.currentAssistantMsg) return;
    this.currentAssistantMsg.raw = fullText;
    this.currentAssistantMsg.isStreaming = false;
    var bubble = this.currentAssistantMsg.el.querySelector('.msg-bubble');
    bubble.classList.remove('streaming-cursor');
    bubble.innerHTML = window.renderMarkdown(fullText);
    this._postProcessCodeBlocks(bubble);
  }

  showError(msg) {
    if (this.currentAssistantMsg) {
      var bubble = this.currentAssistantMsg.el.querySelector('.msg-bubble');
      bubble.classList.remove('streaming-cursor');
      bubble.innerHTML = '<div class="error-msg">' + msg + '</div>';
      this.currentAssistantMsg.isStreaming = false;
    }
  }

  // ── Helpers ────────────────────────────────────────

  _postProcessCodeBlocks(bubble) {
    // no-op — fold/copy handlers are inline onclick
  }

  _scrollToBottom() {
    var el = this.els.messages;
    var distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    if (distanceFromBottom < 120) {
      requestAnimationFrame(function () {
        el.scrollTop = el.scrollHeight;
      });
    }
  }

  _autoResize() {
    var el = this.els.input;
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 120) + 'px';
  }

  abort() {
    if (this.streamHandler) this.streamHandler.abort();
    this.isStreaming = false;
    this.els.sendBtn.disabled = false;
  }
};

// ── Bootstrap ──────────────────────────────────────

function bootApp() {
  window.chatApp = new ChatApp();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', bootApp);
} else {
  bootApp();
}
