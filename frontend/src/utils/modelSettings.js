export function normalizeOptionalModel(value) {
  return String(value || '').trim()
}

export function resolveOptionalLlmModel(value, chatModel) {
  return normalizeOptionalModel(value) || normalizeOptionalModel(chatModel)
}
