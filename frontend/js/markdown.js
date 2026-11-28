/* ═══════════════════════════════════════════════
   Markdown rendering + code highlighting + collapse
   Uses marked v4.3.0 API
   ═══════════════════════════════════════════════ */

// Configure marked
marked.setOptions({
  gfm: true,
  breaks: true,
});

// Custom renderer for code blocks
const renderer = new marked.Renderer();

// marked v4: renderer.code(code, infostring, escaped)
renderer.code = function (code, lang) {
  const language = lang || 'text';
  let highlighted;
  try {
    if (language && hljs.getLanguage(language)) {
      highlighted = hljs.highlight(code, { language }).value;
    } else {
      highlighted = hljs.highlightAuto(code).value;
    }
  } catch (e) {
    highlighted = code.replace(/&/g, '&amp;').replace(/</g, '&lt;');
  }

  const blockId = 'code-' + Math.random().toString(36).slice(2, 8);

  return [
    '<div class="code-block">',
    '  <div class="code-header">',
    '    <span class="code-lang">' + language + '</span>',
    '    <div class="code-actions">',
    '      <button class="code-btn fold-btn" onclick="toggleFold(\'' + blockId + '\')">收起</button>',
    '      <button class="code-btn copy-btn" onclick="copyCode(this, \'' + blockId + '\')">复制</button>',
    '    </div>',
    '  </div>',
    '  <div class="code-body" id="' + blockId + '">',
    '    <pre><code class="hljs language-' + language + '">' + highlighted + '</code></pre>',
    '  </div>',
    '</div>'
  ].join('\n');
};

// marked v4: renderer.paragraph(text)
renderer.paragraph = function (text) {
  if (/^\[来源:/.test(text.trim())) {
    return '<div class="source-ref">' + text + '</div>';
  }
  return '<p>' + text + '</p>';
};

marked.setOptions({ renderer: renderer });

// ── Code fold toggle ──────────────────────────────

window.toggleFold = function (blockId) {
  var body = document.getElementById(blockId);
  if (!body) return;

  var btn = body.parentElement.querySelector('.fold-btn');
  var isFolded = body.classList.toggle('folded');

  if (isFolded) {
    btn.textContent = '展开';
    body.style.maxHeight = '60px';
  } else {
    btn.textContent = '收起';
    body.style.maxHeight = '';
  }
};

// ── Code copy ─────────────────────────────────────

window.copyCode = function (btn, blockId) {
  var body = document.getElementById(blockId);
  if (!body) return;

  var codeEl = body.querySelector('code');
  if (!codeEl) return;
  var code = codeEl.textContent;

  var orig = btn.textContent;
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(code).then(function () {
      btn.textContent = '已复制';
      btn.style.color = '#3fb950';
      setTimeout(function () { btn.textContent = orig; btn.style.color = ''; }, 1800);
    });
  } else {
    // Fallback
    var ta = document.createElement('textarea');
    ta.value = code;
    ta.style.position = 'fixed';
    ta.style.left = '-9999px';
    document.body.appendChild(ta);
    ta.select();
    document.execCommand('copy');
    document.body.removeChild(ta);
    btn.textContent = '已复制';
    setTimeout(function () { btn.textContent = orig; }, 1800);
  }
};

// ── Sanitize ──────────────────────────────────────
// marked 的输出直接进 innerHTML，而它的输入是 LLM 生成的回答：模型可以复述语料
// 里的任何文本，语料又是任意源码，所以渲染前必须过一遍白名单。
// DOMPurify 从 CDN 加载（见 index.html）；拿不到时退化成标签/属性黑名单——弱一
// 点，但不会因为某个 CDN 被墙就把整页渲染搞崩。

// 渲染器内联的两个处理器是唯一允许保留的 onclick，参数只准是 code-xxxx 这类 id
var ALLOWED_ONCLICK = /^\s*(?:toggleFold|copyCode)\(\s*(?:this\s*,\s*)?'[A-Za-z0-9_-]+'\s*\)\s*;?\s*$/;

if (window.DOMPurify && !window.__domPurifyHooked) {
  window.__domPurifyHooked = true;
  window.DOMPurify.addHook('uponSanitizeAttribute', function (node, data) {
    if (!/^on/i.test(data.attrName)) return;
    var ok = data.attrName.toLowerCase() === 'onclick' &&
      ALLOWED_ONCLICK.test(data.attrValue || '');
    if (!ok) data.keepAttr = false;
  });
}

function fallbackSanitize(html) {
  return html
    // 成对出现的危险容器，连内容一起丢掉
    .replace(/<\s*(script|style|iframe|object|embed|form|svg|math)\b[^>]*>[\s\S]*?<\s*\/\s*\1\s*>/gi, '')
    // 未闭合 / 自闭合的同类标签
    .replace(/<\s*\/?\s*(script|style|iframe|object|embed|link|meta|base|form|svg|math)\b[^>]*>/gi, '')
    // 所有内联事件属性
    .replace(/\son[a-z]+\s*=\s*("[^"]*"|'[^']*'|[^\s>]+)/gi, '')
    // javascript: 协议
    .replace(/(href|src|xlink:href)\s*=\s*(["'])\s*javascript:[^"']*\2/gi, '$1=$2#$2');
}

function sanitizeHtml(html) {
  if (window.DOMPurify && window.DOMPurify.sanitize) {
    try {
      return window.DOMPurify.sanitize(html, {
        ADD_ATTR: ['onclick'],
        ALLOW_DATA_ATTR: true,
      });
    } catch (e) {
      console.warn('[Markdown] DOMPurify 失败，退回黑名单过滤:', e);
    }
  }
  return fallbackSanitize(html);
}

// ── Render markdown to HTML ───────────────────────

window.renderMarkdown = function (text) {
  if (!text) return '';
  var html;
  try {
    html = marked.parse(text);
  } catch (e) {
    return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }
  return sanitizeHtml(html);
};
