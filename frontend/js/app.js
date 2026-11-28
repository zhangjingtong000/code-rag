/* ═══════════════════════════════════════════════
   ChatApp — core chat logic + sidebar tracking
   ═══════════════════════════════════════════════ */

// 流式渲染节流间隔（ms）与长文本快路径阈值（字符数）
var RENDER_INTERVAL_MS = 100;
var LARGE_TEXT_CHARS = 4000;

var ChatApp = class ChatApp {
  constructor() {
    this.messages = [];
    this.streamHandler = null;
    this.isStreaming = false;
    this.isRecording = false;
    this.currentAssistantMsg = null;
    this.msgIdCounter = 0;
    this.activeSidebarItem = null;
    this.mediaRecorder = null;
    this.audioChunks = [];
    this.recordStartTime = 0;
    this.recordTimer = null;
    this.maxRecordMs = 30000;  // 30s max
    this.renderTimer = null;

    this.els = {
      messages: document.getElementById('messages'),
      input: document.getElementById('input'),
      sendBtn: document.getElementById('send-btn'),
      stopBtn: document.getElementById('stop-btn'),
      micBtn: document.getElementById('mic-btn'),
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
    this._initMic();
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
      } else if (data.index_status === 'error') {
        this._setStatus('offline', '索引异常');
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
      } else if (data.status === 'error') {
        this._setStatus('offline', '索引异常');
        if (data.progress) this._showToast(data.progress);
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
      this.els.sendBtn.style.display = 'none';
      this.els.stopBtn.style.display = '';
      this.els.micBtn.disabled = true;

      var resp = await fetch('/api/rag/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: text,
          chat_history: this._getChatHistory(),
        }),
      });

      if (!resp.ok) {
        // 409（索引为空 / 未构建）之类要显示后端给的说明，不能只剩 HTTP 409
        var errBody = await resp.json().catch(function () { return {}; });
        throw new Error(errBody.detail || ('HTTP ' + resp.status));
      }

      var data = await resp.json();
      this.streamHandler = new StreamHandler(data.stream_url, this);
      await this.streamHandler.start();

    } catch (err) {
      console.error('Send error:', err);
      this.showError('发送失败: ' + err.message);
    } finally {
      this.isStreaming = false;
      this.els.sendBtn.style.display = '';
      this.els.stopBtn.style.display = 'none';
      this.els.micBtn.disabled = false;
      this.els.input.focus();
    }
  }

  _getChatHistory(excludeTail) {
    // 调用发生在「当前问题」和「空的助手占位」都推进 this.messages 之后，
    // 所以末尾 2 条必须排除——否则当前问题既在 chat_history 里，又由
    // backend 单独作为 question 拼进去，同一句话被注入两次。
    var history = [];
    var end = this.messages.length - (excludeTail === undefined ? 2 : excludeTail);
    var start = Math.max(0, end - 10);  // 最多 10 条 = 5 轮
    for (var i = start; i < end; i++) {
      var msg = this.messages[i];
      if (msg.role === 'system') continue;
      var content = msg.raw || msg.content;
      if (!content) continue;
      history.push({ role: msg.role, content: content });
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
    this.currentAssistantMsg.content += text;
    this._scheduleStreamRender();
  }

  // 打字机每 16ms 一帧。原来每帧都对全文重新 marked.parse + DOMPurify +
  // innerHTML 全量替换，几千字的代码回答会持续触发全量重排，越长越卡。
  // 现在渲染节流到 RENDER_INTERVAL_MS；正文超过 LARGE_TEXT_CHARS 后走
  // textContent 快路径（不解析 Markdown、不重建代码块），完成时再整体渲染一次。
  _scheduleStreamRender() {
    if (this.renderTimer) return;
    var self = this;
    this.renderTimer = setTimeout(function () {
      self.renderTimer = null;
      self._renderStreamingNow();
    }, RENDER_INTERVAL_MS);
  }

  _renderStreamingNow() {
    var msg = this.currentAssistantMsg;
    if (!msg || !msg.isStreaming) return;
    var bubble = msg.el.querySelector('.msg-bubble');
    if (!bubble) return;

    if (msg.content.length > LARGE_TEXT_CHARS) {
      bubble.textContent = msg.content;
      bubble.style.whiteSpace = 'pre-wrap';
    } else {
      bubble.innerHTML = window.renderMarkdown(msg.content);
    }
    bubble.classList.add('streaming-cursor');
    this._scrollToBottom();
  }

  _cancelStreamRender() {
    if (this.renderTimer) {
      clearTimeout(this.renderTimer);
      this.renderTimer = null;
    }
  }

  onStreamComplete(fullText) {
    if (!this.currentAssistantMsg) return;
    this._cancelStreamRender();
    this.currentAssistantMsg.raw = fullText;
    this.currentAssistantMsg.isStreaming = false;
    var bubble = this.currentAssistantMsg.el.querySelector('.msg-bubble');
    bubble.classList.remove('streaming-cursor');
    bubble.style.whiteSpace = '';
    bubble.innerHTML = window.renderMarkdown(fullText);
  }

  showError(msg) {
    if (this.currentAssistantMsg) {
      this._cancelStreamRender();
      var bubble = this.currentAssistantMsg.el.querySelector('.msg-bubble');
      bubble.classList.remove('streaming-cursor');
      bubble.style.whiteSpace = '';
      // 错误文案可能带上后端返回的 detail，用 textContent 写入，不拼 HTML
      bubble.innerHTML = '';
      var box = document.createElement('div');
      box.className = 'error-msg';
      box.textContent = msg;
      bubble.appendChild(box);
      this.currentAssistantMsg.isStreaming = false;
    }
  }

  // ── Helpers ────────────────────────────────────────

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
    this._cancelStreamRender();
    this.isStreaming = false;
    this.els.sendBtn.disabled = false;
  }

  // ── Voice input ──────────────────────────────────

  _initMic() {
    var mic = this.els.micBtn;
    if (!mic) return;

    // Feature detection: hide if MediaRecorder not supported
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia || !window.MediaRecorder) {
      mic.style.display = 'none';
      return;
    }

    // mousedown / touchstart → start recording
    mic.addEventListener('mousedown', (e) => { e.preventDefault(); this._startRecording(); });
    mic.addEventListener('touchstart', (e) => { e.preventDefault(); this._startRecording(); });

    // mouseup / touchend → stop recording (on document to catch release outside button)
    document.addEventListener('mouseup', (e) => { if (this.isRecording) this._stopRecording(); });
    document.addEventListener('touchend', (e) => { if (this.isRecording) this._stopRecording(); });

    // Prevent context menu on long-press
    mic.addEventListener('contextmenu', function(e) { e.preventDefault(); });
  }

  _setMicActive(active) {
    var mic = this.els.micBtn;
    if (!mic) return;
    if (active) {
      mic.classList.add('recording');
    } else {
      mic.classList.remove('recording');
    }
  }

  _setTranscribing(active) {
    var mic = this.els.micBtn;
    if (!mic) return;
    if (active) {
      mic.classList.add('transcribing');
      this.els.input.placeholder = '正在识别语音...';
      // Show brief dots animation in input area
      this.els.input.style.animation = 'input-pulse 1.2s ease-in-out infinite';
    } else {
      mic.classList.remove('transcribing');
      this.els.input.placeholder = '询问关于项目代码的问题...';
      this.els.input.style.animation = '';
    }
  }

  async _startRecording() {
    if (this.isStreaming || this.isRecording) return;

    try {
      var stream = await navigator.mediaDevices.getUserMedia({
        audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: true }
      });
      this.isRecording = true;
      this.audioChunks = [];
      this._setMicActive(true);

      // Show recording indicator in input
      this.els.input.placeholder = '正在聆听...';
      this.els.input.style.color = '#ef4444';

      // Timer display on mic button
      this.recordStartTime = Date.now();
      this.recordTimer = setInterval(() => {
        var elapsed = Date.now() - this.recordStartTime;
        if (elapsed >= this.maxRecordMs) this._stopRecording();
      }, 500);

      var mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus' : 'audio/webm';

      this.mediaRecorder = new MediaRecorder(stream, { mimeType: mimeType });
      this.mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) this.audioChunks.push(e.data);
      };
      this.mediaRecorder.onstop = () => this._onRecordingStop();
      this.mediaRecorder.start();
    } catch (err) {
      console.error('Mic access denied:', err);
      this._showToast('无法访问麦克风，请检查浏览器权限');
      this.isRecording = false;
      this._setMicActive(false);
    }
  }

  _stopRecording() {
    if (!this.isRecording || !this.mediaRecorder) return;

    this.isRecording = false;
    this._setMicActive(false);
    this.els.input.placeholder = '询问关于项目代码的问题...';
    this.els.input.style.color = '';

    if (this.recordTimer) { clearInterval(this.recordTimer); this.recordTimer = null; }

    try {
      this.mediaRecorder.stop();
      if (this.mediaRecorder.stream) {
        this.mediaRecorder.stream.getTracks().forEach(function(t) { t.stop(); });
      }
    } catch(e) { /* recorder already stopped */ }
  }

  async _onRecordingStop() {
    var duration = (Date.now() - this.recordStartTime) / 1000;

    // Ignore if too short (accidental tap)
    if (duration < 0.3 || this.audioChunks.length === 0) {
      this.audioChunks = [];
      return;
    }

    var webmBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
    this.audioChunks = [];

    // Convert webm → 16kHz mono WAV (no ffmpeg needed)
    this._setTranscribing(true);
    var wavBlob;
    try {
      wavBlob = await this._webmToWav(webmBlob);
      if (!wavBlob || wavBlob.size < 500) {
        this._setTranscribing(false);
        this._showToast('音频处理失败');
        return;
      }
    } catch (e) {
      console.error('WebM→WAV conversion error:', e);
      this._setTranscribing(false);
      this._showToast('音频处理失败');
      return;
    }

    // Upload to backend
    try {
      this.els.micBtn.disabled = true;

      var formData = new FormData();
      formData.append('file', wavBlob, 'recording.wav');

      var resp = await fetch('/api/rag/audio/transcribe', {
        method: 'POST',
        body: formData,
      });

      if (!resp.ok) {
        this._setTranscribing(false);
        var errData = await resp.json().catch(function() { return {}; });
        if (resp.status === 422) {
          this._showToast('未识别到语音');
        } else {
          this._showToast('语音识别失败: ' + (errData.detail || resp.statusText));
        }
        return;
      }

      var data = await resp.json();
      var text = (data.text || '').trim();
      this._setTranscribing(false);

      if (!text) {
        this._showToast('未识别到语音');
        return;
      }

      // Fill input — user can review/edit before sending
      this.els.input.value = text;
      this._autoResize();
      this.els.input.focus();

    } catch (err) {
      this._setTranscribing(false);
      console.error('Transcribe error:', err);
      this._showToast('语音识别请求失败');
    } finally {
      this.els.micBtn.disabled = false;
    }
  }

  async _webmToWav(blob) {
    // Decode webm via AudioContext
    var arrayBuffer = await blob.arrayBuffer();
    var audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
    var audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
    audioCtx.close();

    // Resample to 16kHz mono if needed
    var targetRate = 16000;
    var srcData = audioBuffer.getChannelData(0);  // mono from channel 0
    var srcRate = audioBuffer.sampleRate;

    // Simple resampling (linear interpolation)
    var samples;
    if (srcRate !== targetRate) {
      var ratio = srcRate / targetRate;
      var newLen = Math.round(srcData.length / ratio);
      samples = new Float32Array(newLen);
      for (var i = 0; i < newLen; i++) {
        var srcIdx = i * ratio;
        var idx0 = Math.floor(srcIdx);
        var idx1 = Math.min(idx0 + 1, srcData.length - 1);
        var frac = srcIdx - idx0;
        samples[i] = srcData[idx0] * (1 - frac) + srcData[idx1] * frac;
      }
    } else {
      samples = srcData;
    }

    // Convert Float32 → 16-bit PCM
    var pcm = new Int16Array(samples.length);
    for (var i = 0; i < samples.length; i++) {
      var s = Math.max(-1, Math.min(1, samples[i]));
      pcm[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }

    // Build WAV header
    var header = new ArrayBuffer(44);
    var view = new DataView(header);
    var writeStr = function(off, str) {
      for (var i = 0; i < str.length; i++) view.setUint8(off + i, str.charCodeAt(i));
    };
    writeStr(0, 'RIFF');
    view.setUint32(4, 36 + pcm.length * 2, true);
    writeStr(8, 'WAVE');
    writeStr(12, 'fmt ');
    view.setUint32(16, 16, true);           // chunk size
    view.setUint16(20, 1, true);            // PCM
    view.setUint16(22, 1, true);            // mono
    view.setUint32(24, targetRate, true);
    view.setUint32(28, targetRate * 2, true); // byte rate
    view.setUint16(32, 2, true);            // block align
    view.setUint16(34, 16, true);           // bits per sample
    writeStr(36, 'data');
    view.setUint32(40, pcm.length * 2, true);

    // Combine header + PCM data
    var wavBytes = new Uint8Array(44 + pcm.byteLength);
    wavBytes.set(new Uint8Array(header), 0);
    wavBytes.set(new Uint8Array(pcm.buffer), 44);

    return new Blob([wavBytes], { type: 'audio/wav' });
  }

  _showToast(msg) {
    // Simple toast notification
    var toast = document.createElement('div');
    toast.className = 'toast';
    toast.textContent = msg;
    document.body.appendChild(toast);
    // Trigger reflow for animation
    toast.offsetHeight;
    toast.classList.add('show');
    setTimeout(function() {
      toast.classList.remove('show');
      setTimeout(function() { toast.remove(); }, 300);
    }, 2500);
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
