/**
 * Remove browser-introduced wrapper rows from a one-line paste.
 *
 * Native selection of a block-level chat bubble can place a valid one-line
 * message between leading/trailing line breaks on the clipboard.  We only
 * normalize that exact shape.  Pasted code, lists and deliberately multiline
 * questions keep their original line structure.
 */
export function normalizeSingleLinePaste(value) {
  if (typeof value !== 'string') return value

  const normalizedNewlines = value.replace(/\r\n?/g, '\n')
  const nonEmptyLines = normalizedNewlines
    .split('\n')
    .filter(line => line.trim())

  if (nonEmptyLines.length !== 1) return value
  return nonEmptyLines[0].trim()
}
