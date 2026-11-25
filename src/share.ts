// Copyright (C) 2026 Francisco Peña Gallardo (Reox Lab)
//
// This file is part of LinealAlRL.
//
// LinealAlRL is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// LinealAlRL is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with LinealAlRL. If not, see <https://www.gnu.org/licenses/>.

// Serialización y deserialización del estado matricial en el hash de la URL (#m=<base64url>).

import { type Matrix } from './matrix.ts'
import { type Rat, parseRat, ratToString } from './rational.ts'

export const MAX_GRID_DIM = 6
export const SHARE_SCHEMA_VERSION = 1
export const MAX_RAT_DIGITS = 60
const BASE64URL_RE = /^[A-Za-z0-9_-]+$/

export interface SharedState {
  a: Matrix
  b: Matrix | null
  vector: Rat[] | null
}

interface SerializedState {
  s: number
  a: string[][]
  b?: string[][]
  v?: string[]
}

function toBase64Url(text: string): string {
  const bytes = new TextEncoder().encode(text)
  let binary = ''
  for (let i = 0; i < bytes.length; i++) {
    binary += String.fromCharCode(bytes[i])
  }
  return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')
}

function fromBase64Url(encoded: string): string {
  const base64 = encoded.replace(/-/g, '+').replace(/_/g, '/')
  const pad = base64.length % 4 === 0 ? '' : '='.repeat(4 - (base64.length % 4))
  const binary = atob(base64 + pad)
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i)
  }
  return new TextDecoder('utf-8', { fatal: true }).decode(bytes)
}

function exceedsDigitLimit(val: Rat): boolean {
  const absN = val.n < 0n ? -val.n : val.n
  return absN.toString().length > MAX_RAT_DIGITS || val.d.toString().length > MAX_RAT_DIGITS
}

function parseMatrixField(raw: unknown): Matrix | null {
  if (!Array.isArray(raw) || raw.length === 0 || raw.length > MAX_GRID_DIM) {
    return null
  }
  const rows: Matrix = []
  let expectedCols = -1
  for (const row of raw) {
    if (!Array.isArray(row) || row.length === 0 || row.length > MAX_GRID_DIM) {
      return null
    }
    if (expectedCols === -1) {
      expectedCols = row.length
    } else if (row.length !== expectedCols) {
      return null
    }
    const cells: Rat[] = []
    for (const cell of row) {
      if (typeof cell !== 'string' || cell.length === 0 || cell.length > 64) {
        return null
      }
      const val = parseRat(cell)
      if (val === null || exceedsDigitLimit(val)) return null
      cells.push(val)
    }
    rows.push(cells)
  }
  return rows
}

function parseVectorField(raw: unknown): Rat[] | null {
  if (!Array.isArray(raw) || raw.length === 0 || raw.length > MAX_GRID_DIM) {
    return null
  }
  const out: Rat[] = []
  for (const item of raw) {
    if (typeof item !== 'string' || item.length === 0 || item.length > 64) {
      return null
    }
    const val = parseRat(item)
    if (val === null || exceedsDigitLimit(val)) return null
    out.push(val)
  }
  return out
}

/**
 * Codifica la matriz A (y opcionalmente B y el vector b) en base64url exacto
 * usando fracciones irreducibles.
 */
export function encodeState(
  a: Matrix,
  b?: Matrix | null,
  vector?: Rat[] | null,
): string {
  const payload: SerializedState = {
    s: SHARE_SCHEMA_VERSION,
    a: a.map((row) => row.map(ratToString)),
  }
  if (b && b.length > 0 && b[0].length > 0) {
    payload.b = b.map((row) => row.map(ratToString))
  }
  if (vector && vector.length > 0) {
    payload.v = vector.map(ratToString)
  }
  return toBase64Url(JSON.stringify(payload))
}

/**
 * Decodifica una cadena base64url (con o sin prefijo `#m=` / `?m=` / `m=`).
 * Devuelve `null` si el estado es inválido o está manipulado; nunca lanza excepciones.
 */
export function decodeState(texto: string): SharedState | null {
  if (typeof texto !== 'string') return null
  let raw = texto.trim()
  if (raw.startsWith('#') || raw.startsWith('?')) {
    raw = raw.slice(1)
  }
  if (raw.startsWith('m=')) {
    raw = raw.slice(2)
  }
  if (raw.length === 0 || !BASE64URL_RE.test(raw)) return null

  try {
    const parsed: unknown = JSON.parse(fromBase64Url(raw))
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
      return null
    }
    const obj = parsed as Record<string, unknown>

    if (obj.s !== undefined) {
      if (
        typeof obj.s !== 'number' ||
        !Number.isInteger(obj.s) ||
        obj.s < 1 ||
        obj.s > SHARE_SCHEMA_VERSION
      ) {
        return null
      }
    }

    const a = parseMatrixField(obj.a)
    if (!a) return null

    let b: Matrix | null = null
    if (obj.b !== undefined && obj.b !== null) {
      b = parseMatrixField(obj.b)
      if (!b) return null
    }

    let vector: Rat[] | null = null
    const rawVec = obj.v !== undefined ? obj.v : obj.vector
    if (rawVec !== undefined && rawVec !== null) {
      vector = parseVectorField(rawVec)
      if (!vector || vector.length !== a.length) return null
    }

    return { a, b, vector }
  } catch {
    return null
  }
}

export function readStateFromUrl(): SharedState | null {
  if (typeof window === 'undefined' || !window.location) return null
  const hash = window.location.hash
  if (!hash || !hash.startsWith('#m=')) return null
  return decodeState(hash)
}

export function buildShareUrl(
  a: Matrix,
  b?: Matrix | null,
  vector?: Rat[] | null,
): string {
  const encoded = encodeState(a, b, vector)
  if (typeof window === 'undefined' || !window.location) {
    return `#m=${encoded}`
  }
  const url = new URL(window.location.href)
  url.hash = `m=${encoded}`
  return url.toString()
}
