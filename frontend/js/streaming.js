/* ═══════════════════════════════════════════════
   SSE streaming handler + typewriter buffer effect
   ═══════════════════════════════════════════════ */

class StreamHandler {
  constructor(url, app) {
    this.url = url;
    this.app = app;
    this.reader = null;
    this.abortController = null;
    this.rawAnswer = '';
    this.displayedLength = 0;
    this.timer = null;
    this.isComplete = false;
    this.lastEventId = '';
  }

  async start() {
    this.abortController = new AbortController();

    try {
      const headers = {};
      if (this.lastEventId) {
        headers['Last-Event-ID'] = this.lastEventId;
      }

      const response = await fetch(this.url, {
        signal: this.abortController.signal,
        headers,
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      this.reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let currentEvent = '';

      while (true) {
        const { done, value } = await this.reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('event: ')) {
            currentEvent = line.slice(7).trim();
          } else if (line.startsWith('data: ')) {
            this._handleEvent(currentEvent, line.slice(6));
            currentEvent = '';
          } else if (line.startsWith('id: ')) {
            this.lastEventId = line.slice(4).trim();
          }
        }
      }
    } catch (err) {
      if (err.name === 'AbortError') return;
      console.error('Stream error:', err);
      this.app.showError('连接中断: ' + err.message);
    }
  }

  _handleEvent(event, dataStr) {
    try {
      const data = JSON.parse(dataStr);

      switch (event) {
        case 'metadata':
          this.app.onMetadata(data);
          break;

        case 'token':
          this.rawAnswer += data.text || '';
          this._scheduleFlush();
          break;

        case 'done':
          this.isComplete = true;
          this._flushAll();
          this.app.onStreamComplete(this.rawAnswer);
          break;

        case 'error':
          this.isComplete = true;
          this._flushAll();
          this.app.showError(data.error || '未知错误');
          break;
      }
    } catch (e) {
      // Skip malformed SSE data
    }
  }

  _scheduleFlush() {
    if (this.timer) return;

    const FRAME_MS = 16;
    const CHARS_MIN = 8;
    const CHARS_MAX = 15;

    this.timer = setInterval(() => {
      if (this.displayedLength >= this.rawAnswer.length) {
        if (this.isComplete) {
          clearInterval(this.timer);
          this.timer = null;
        }
        return;
      }

      const remaining = this.rawAnswer.length - this.displayedLength;
      const chars = Math.min(
        remaining,
        CHARS_MIN + Math.floor(Math.random() * (CHARS_MAX - CHARS_MIN))
      );
      const chunk = this.rawAnswer.slice(this.displayedLength, this.displayedLength + chars);
      this.displayedLength += chars;
      this.app.appendStreamText(chunk);
    }, FRAME_MS);
  }

  _flushAll() {
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = null;
    }
    if (this.displayedLength < this.rawAnswer.length) {
      const remaining = this.rawAnswer.slice(this.displayedLength);
      this.displayedLength = this.rawAnswer.length;
      this.app.appendStreamText(remaining);
    }
  }

  abort() {
    if (this.abortController) {
      this.abortController.abort();
    }
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = null;
    }
  }
}
