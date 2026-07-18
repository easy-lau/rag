// 从 User-Agent 粗略解析浏览器 / 操作系统 / 设备类型，用于日志详情展示。
export function parseUA(ua) {
  if (!ua) return { browser: '—', os: '—', device: '—' }
  let browser = '其他'
  if (/Edg\//.test(ua)) browser = 'Edge'
  else if (/OPR\/|Opera/.test(ua)) browser = 'Opera'
  else if (/Firefox\//.test(ua)) browser = 'Firefox'
  else if (/Chrome\//.test(ua)) browser = 'Chrome'
  else if (/Safari\//.test(ua)) browser = 'Safari'
  const ver = ua.match(/(?:Edg|OPR|Firefox|Chrome|Version)\/(\d+)/)
  if (browser !== '其他' && ver) browser += ` ${ver[1]}`

  let os = '其他'
  if (/Windows NT 10/.test(ua)) os = 'Windows 10/11'
  else if (/Windows/.test(ua)) os = 'Windows'
  else if (/Mac OS X/.test(ua)) os = 'macOS'
  else if (/Android/.test(ua)) os = 'Android'
  else if (/(iPhone|iPad|iPod|iOS)/.test(ua)) os = 'iOS'
  else if (/Linux/.test(ua)) os = 'Linux'

  const device = /(Mobile|Android|iPhone|iPod)/.test(ua) ? '移动端'
    : /(iPad|Tablet)/.test(ua) ? '平板' : '桌面端'
  return { browser, os, device }
}
