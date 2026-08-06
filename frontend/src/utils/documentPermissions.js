// The backend is the authorization authority.  These helpers deliberately
// consume only its per-document decision and default to deny when a response
// is missing or stale; they do not reconstruct ownership rules in the client.
export function documentAllows(document, action) {
  return document?.permissions?.[action] === true
}

export const canReadDocumentRow = document => documentAllows(document, 'read')
export const canUpdateDocumentRow = document => documentAllows(document, 'update')
export const canDeleteDocumentRow = document => documentAllows(document, 'delete')
