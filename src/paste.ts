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

// Parser de matrices y matrices aumentadas pegadas desde el portapapeles.

import { type Matrix } from './matrix.ts'
import { type Rat, parseRat } from './rational.ts'

export type PasteMatrixResult =
  | { ok: true; matriz: Matrix; filas: number; cols: number }
  | { ok: false; error: string }

export type PasteAugmentedResult =
  | { ok: true; a: Matrix; vector: Rat[]; filas: number; cols: number }
  | { ok: false; error: string }

type RowDelimiterMode = 'tab' | 'semi' | 'default'

const DECIMAL_COMMA_RE = /^([+-−]?\d*),(\d+)$/

function detectRowMode(line: string): RowDelimiterMode {
  if (line.includes('\t')) return 'tab'
  if (line.includes(';')) return 'semi'
  return 'default'
}

function splitSegmentByMode(segment: string, mode: RowDelimiterMode): string[] | null {
  const trimmed = segment.trim()
  if (!trimmed) return null

  if (mode === 'tab') {
    const cells = trimmed.split('\t').map((t) => t.trim())
    if (cells.some((c) => c.length === 0)) return null
    return cells.map((c) => c.replace(DECIMAL_COMMA_RE, '$1.$2'))
  }

  if (mode === 'semi') {
    if (/(?:^;|;$|[,;]\s*[,;])/.test(trimmed)) return null
    const cells = trimmed.split(';').map((t) => t.trim())
    if (cells.some((c) => c.length === 0)) return null
    return cells.map((c) => c.replace(DECIMAL_COMMA_RE, '$1.$2'))
  }

  if (/(?:^,|,$|[,;]\s*[,;])/.test(trimmed)) return null
  const tokens = trimmed
    .split(/,+|\s+/)
    .map((t) => t.trim())
    .filter((t) => t.length > 0)

  return tokens.length > 0 ? tokens : null
}

function splitRowTokens(line: string): string[] | null {
  const normalized = line.trim().replace(/\s*\/\s*/g, '/')
  if (!normalized) return null
  const mode = detectRowMode(normalized)
  return splitSegmentByMode(normalized, mode)
}

interface ParseOptions {
  maxRows: number
  maxCols: number
  allowPipe: boolean
  maxColsError?: (cols: number) => string
}

function parsePastedInternal(texto: string, opts: ParseOptions): PasteMatrixResult {
  const lines = texto
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l.length > 0)

  if (lines.length === 0) {
    return {
      ok: false,
      error:
        'El texto está vacío. Pega al menos una fila con números separados por espacios, comas, punto y coma o tabuladores.',
    }
  }

  if (lines.length > opts.maxRows) {
    return {
      ok: false,
      error: `El texto tiene ${lines.length} filas, pero el máximo permitido es ${opts.maxRows}.`,
    }
  }

  const hasPipe = opts.allowPipe && lines.some((l) => l.includes('|'))
  const matriz: Matrix = []
  let expectedCols = -1

  for (let i = 0; i < lines.length; i++) {
    let tokens: string[] | null

    if (hasPipe) {
      const normalized = lines[i].replace(/\s*\/\s*/g, '/')
      const pipeParts = normalized.split('|')
      if (pipeParts.length === 1) {
        return {
          ok: false,
          error: `La fila ${i + 1} no contiene el separador «|». Si usas «|», debe aparecer exactamente una vez en cada fila, justo antes del vector b.`,
        }
      }
      if (pipeParts.length > 2) {
        return {
          ok: false,
          error: `La fila ${i + 1} contiene más de un separador «|». Usa «|» una sola vez por fila, justo antes del vector b.`,
        }
      }
      const mode = detectRowMode(normalized)
      const leftTokens = splitSegmentByMode(pipeParts[0], mode)
      const rightTokens = splitSegmentByMode(pipeParts[1], mode)
      if (!leftTokens || !rightTokens) {
        return {
          ok: false,
          error: `La fila ${i + 1} contiene celdas vacías o separadores sobrantes alrededor de «|».`,
        }
      }
      if (rightTokens.length !== 1) {
        return {
          ok: false,
          error: `En la fila ${i + 1}, el separador «|» debe colocarse justo antes del último valor (la columna b).`,
        }
      }
      tokens = [...leftTokens, rightTokens[0]]
    } else {
      tokens = splitRowTokens(lines[i])
      if (!tokens) {
        return {
          ok: false,
          error: `La fila ${i + 1} contiene celdas vacías o separadores sobrantes.`,
        }
      }
    }

    if (expectedCols === -1) {
      expectedCols = tokens.length
      if (expectedCols > opts.maxCols) {
        return {
          ok: false,
          error: opts.maxColsError
            ? opts.maxColsError(expectedCols)
            : `La fila 1 tiene ${expectedCols} columnas, pero el máximo permitido es ${opts.maxCols}.`,
        }
      }
    } else if (tokens.length !== expectedCols) {
      return {
        ok: false,
        error: `La fila ${i + 1} tiene ${tokens.length} valores, pero la fila 1 tiene ${expectedCols}. Todas las filas deben tener la misma longitud.`,
      }
    }

    const row: Rat[] = []
    for (let j = 0; j < tokens.length; j++) {
      const val = parseRat(tokens[j])
      if (val === null) {
        return {
          ok: false,
          error: `Valor inválido «${tokens[j]}» en la fila ${i + 1}, columna ${j + 1}. Usa enteros, decimales o fracciones como 3/4.`,
        }
      }
      row.push(val)
    }
    matriz.push(row)
  }

  return {
    ok: true,
    matriz,
    filas: matriz.length,
    cols: expectedCols,
  }
}

/**
 * Parsea texto plano con filas separadas por salto de línea y valores
 * separados por tabulador, punto y coma, coma o espacio.
 */
export function parsePastedMatrix(
  texto: string,
  maxRows = 6,
  maxCols = 6,
  allowPipe = false,
): PasteMatrixResult {
  return parsePastedInternal(texto, { maxRows, maxCols, allowPipe })
}

/**
 * Parsea una matriz aumentada [A | b]: la última columna es el vector b y las anteriores forman A.
 * Acepta opcionalmente el separador `|` entre A y b.
 */
export function parsePastedAugmented(
  texto: string,
  maxRows = 6,
  maxACols = 6,
): PasteAugmentedResult {
  const parsed = parsePastedInternal(texto, {
    maxRows,
    maxCols: maxACols + 1,
    allowPipe: true,
    maxColsError: (cols) =>
      `La fila 1 tiene ${cols - 1} columnas para A; A admite como máximo ${maxACols} columnas (más la columna b).`,
  })
  if (!parsed.ok) return parsed

  if (parsed.cols < 2) {
    return {
      ok: false,
      error: 'Una matriz aumentada [A | b] necesita al menos 2 columnas (coeficientes de A y vector b).',
    }
  }

  const a: Matrix = parsed.matriz.map((row) => row.slice(0, parsed.cols - 1))
  const vector: Rat[] = parsed.matriz.map((row) => row[parsed.cols - 1])

  return {
    ok: true,
    a,
    vector,
    filas: parsed.filas,
    cols: parsed.cols - 1,
  }
}
