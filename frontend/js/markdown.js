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

// ── Render markdown to HTML ───────────────────────

window.renderMarkdown = function (text) {
  if (!text) return '';
  try {
    return marked.parse(text);
  } catch (e) {
    return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }
};
