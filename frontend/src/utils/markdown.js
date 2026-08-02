import MarkdownIt from 'markdown-it'
import hljs from 'highlight.js/lib/core'
import bash from 'highlight.js/lib/languages/bash'
import css from 'highlight.js/lib/languages/css'
import diff from 'highlight.js/lib/languages/diff'
import dockerfile from 'highlight.js/lib/languages/dockerfile'
import xml from 'highlight.js/lib/languages/xml'
import http from 'highlight.js/lib/languages/http'
import ini from 'highlight.js/lib/languages/ini'
import java from 'highlight.js/lib/languages/java'
import javascript from 'highlight.js/lib/languages/javascript'
import json from 'highlight.js/lib/languages/json'
import markdown from 'highlight.js/lib/languages/markdown'
import python from 'highlight.js/lib/languages/python'
import sql from 'highlight.js/lib/languages/sql'
import typescript from 'highlight.js/lib/languages/typescript'
import yaml from 'highlight.js/lib/languages/yaml'
import DOMPurify from 'dompurify'

// 只注册知识库和对话中常见的代码语言。整包 highlight.js 会包含全部语言定义，
// 让 Markdown 共享块超过 1 MB；未知语言仍会安全地按纯文本展示。
const highlightLanguages = {
  bash,
  shell: bash,
  sh: bash,
  css,
  diff,
  dockerfile,
  docker: dockerfile,
  html: xml,
  xml,
  svg: xml,
  http,
  ini,
  java,
  javascript,
  js: javascript,
  json,
  markdown,
  md: markdown,
  python,
  py: python,
  sql,
  typescript,
  ts: typescript,
  yaml,
  yml: yaml,
}

for (const [name, definition] of Object.entries(highlightLanguages)) {
  hljs.registerLanguage(name, definition)
}

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
