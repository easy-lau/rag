import MarkdownIt from 'markdown-it'
import hljs from 'highlight.js'
import DOMPurify from 'dompurify'

const md = new MarkdownIt({
  html: false,
  linkify: true,
  typographer: true,
})

// 直接覆盖 fenced code block 的渲染规则，而不是用 `highlight` 选项。
// markdown-it 的 highlight 流程只有在返回值以 "<pre" 开头时才会跳过再包裹，
// 否则会把结果再套一层 <pre><code>，导致代码块嵌套、样式错乱。覆盖 fence
// 规则可以完全掌控输出结构，避免这个问题。
md.renderer.rules.fence = (tokens, idx) => {
  const token = tokens[idx]
  const lang = (token.info || '').trim().split(/\s+/)[0]
  const language = lang && hljs.getLanguage(lang) ? lang : 'plaintext'

  let code
  try {
    code = hljs.highlight(token.content, { language, ignoreIllegals: true }).value
  } catch {
    code = md.utils.escapeHtml(token.content)
  }

  const label = md.utils.escapeHtml(lang || 'text')
  return (
    '<div class="code-block-wrapper">' +
      '<div class="code-block-header">' +
        `<span class="code-lang">${label}</span>` +
        '<button class="copy-btn" type="button">复制</button>' +
      '</div>' +
      `<pre class="hljs"><code>${code}</code></pre>` +
    '</div>'
  )
}

export function renderMarkdown(content) {
  if (!content) return ''
  return DOMPurify.sanitize(md.render(content))
}

// 文档预览用渲染：与聊天一致地用 .code-block-wrapper 包裹（带边框 + 语言标签），
// 但不带「复制」按钮（预览场景没有对应点击处理）。
// 必须带容器边框：github.css 的 .hljs 背景为白色，裸 <pre> 在白底预览上会“看不出是代码块”。
const docMd = new MarkdownIt({ html: false, linkify: true, typographer: true })
docMd.renderer.rules.fence = (tokens, idx) => {
  const token = tokens[idx]
  const lang = (token.info || '').trim().split(/\s+/)[0]
  const language = lang && hljs.getLanguage(lang) ? lang : 'plaintext'
  let code
  try {
    code = hljs.highlight(token.content, { language, ignoreIllegals: true }).value
  } catch {
    code = docMd.utils.escapeHtml(token.content)
  }
  const label = docMd.utils.escapeHtml(lang || 'text')
  return (
    '<div class="code-block-wrapper">' +
      '<div class="code-block-header">' +
        `<span class="code-lang">${label}</span>` +
      '</div>' +
      `<pre class="hljs"><code>${code}</code></pre>` +
    '</div>'
  )
}

export function renderDocMarkdown(content) {
  if (!content) return ''
  return DOMPurify.sanitize(docMd.render(content))
}
