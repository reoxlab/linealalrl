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

import { Rat, parseRat, ratToLatex, ratToString } from './rational.ts'

export type Matrix = Rat[][]

export interface Step {
  texto: string
  matriz: Matrix
}

export interface DetResult {
  valor: Rat
  pasos: Step[]
}

export interface RrefResult {
  resultado: Matrix
  pasos: Step[]
}

export type InverseResult =
  | { inversa: Matrix; pasos: Step[] }
  | { inversa: null; pasos: Step[]; atasco: string }

export interface SolveResult {
  tipo: 'unica' | 'infinitas' | 'ninguna'
  solucion?: Rat[]
  libres?: number[]
  pasos: Step[]
}

function assertValidDims(filas: number, cols: number): void {
  if (!Number.isInteger(filas) || !Number.isInteger(cols) || filas < 0 || cols < 0) {
    throw new Error(`Dimensiones de matriz inválidas: ${filas}×${cols}`)
  }
}

function assertRectangular(m: Matrix): [number, number] {
  const filas = m.length
  if (filas === 0) return [0, 0]
  const cols = m[0].length
  for (let i = 1; i < filas; i++) {
    if (m[i].length !== cols) {
      throw new Error(`La matriz no es rectangular: la fila ${i + 1} tiene ${m[i].length} columnas en lugar de ${cols}`)
    }
  }
  return [filas, cols]
}

export function zeros(filas: number, cols: number): Matrix {
  assertValidDims(filas, cols)
  return Array.from({ length: filas }, () =>
    Array.from({ length: cols }, () => Rat.ZERO),
  )
}

export function identity(n: number): Matrix {
  assertValidDims(n, n)
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? Rat.ONE : Rat.ZERO)),
  )
}

export function clone(m: Matrix): Matrix {
  return m.map((row) => row.slice())
}

export function dims(m: Matrix): [number, number] {
  return [m.length, m.length > 0 ? m[0].length : 0]
}

export function isSquare(m: Matrix): boolean {
  if (m.length === 0) return false
  const cols = m[0].length
  if (cols === 0 || m.length !== cols) return false
  for (let i = 1; i < m.length; i++) {
    if (m[i].length !== cols) return false
  }
  return true
}

export function fromStrings(celdas: string[][]): Matrix | null {
  if (celdas.length === 0) return null
  const cols = celdas[0].length
  if (cols === 0) return null

  const out: Matrix = []
  for (let i = 0; i < celdas.length; i++) {
    const fila = celdas[i]
    if (fila.length !== cols) return null
    const parsed: Rat[] = []
    for (let j = 0; j < cols; j++) {
      const val = parseRat(fila[j])
      if (val === null) return null
      parsed.push(val)
    }
    out.push(parsed)
  }
  return out
}

function minorMatrix(m: Matrix, skipRow: number, skipCol: number): Matrix {
  const res: Matrix = []
  for (let i = 0; i < m.length; i++) {
    if (i === skipRow) continue
    const row: Rat[] = []
    for (let j = 0; j < m[i].length; j++) {
      if (j === skipCol) continue
      row.push(m[i][j])
    }
    res.push(row)
  }
  return res
}

function chooseBestLine(m: Matrix): { tipo: 'fila' | 'columna'; indice: number; ceros: number } {
  const n = m.length
  let best: { tipo: 'fila' | 'columna'; indice: number; ceros: number } = {
    tipo: 'fila',
    indice: 0,
    ceros: -1,
  }

  for (let i = 0; i < n; i++) {
    let ceros = 0
    for (let j = 0; j < n; j++) {
      if (m[i][j].isZero()) ceros++
    }
    if (ceros > best.ceros) {
      best = { tipo: 'fila', indice: i, ceros }
    }
  }

  for (let j = 0; j < n; j++) {
    let ceros = 0
    for (let i = 0; i < n; i++) {
      if (m[i][j].isZero()) ceros++
    }
    if (ceros > best.ceros) {
      best = { tipo: 'columna', indice: j, ceros }
    }
  }

  return best
}

function detRecursive(m: Matrix, pasos: Step[]): Rat {
  const n = m.length
  if (n === 1) {
    const val = m[0][0]
    pasos.push({
      texto: `Determinante 1×1: det = ${ratToString(val)}`,
      matriz: clone(m),
    })
    return val
  }

  if (n === 2) {
    const val = m[0][0].mul(m[1][1]).sub(m[0][1].mul(m[1][0]))
    pasos.push({
      texto: `Determinante 2×2: (${ratToString(m[0][0])})(${ratToString(m[1][1])}) − (${ratToString(m[0][1])})(${ratToString(m[1][0])}) = ${ratToString(val)}`,
      matriz: clone(m),
    })
    return val
  }

  const linea = chooseBestLine(m)
  const etiquetaLinea =
    linea.tipo === 'fila' ? `fila ${linea.indice + 1}` : `columna ${linea.indice + 1}`

  if (linea.ceros === n) {
    pasos.push({
      texto: `La ${etiquetaLinea} es toda de ceros ⇒ det = 0`,
      matriz: clone(m),
    })
    return Rat.ZERO
  }

  let total = Rat.ZERO
  for (let k = 0; k < n; k++) {
    const i = linea.tipo === 'fila' ? linea.indice : k
    const j = linea.tipo === 'fila' ? k : linea.indice
    const elem = m[i][j]
    if (elem.isZero()) continue

    const esPositivo = (i + j) % 2 === 0
    const signoStr = esPositivo ? '+' : '−'
    const menor = minorMatrix(m, i, j)
    const subPasos: Step[] = []
    const detMenor = detRecursive(menor, subPasos)
    const cofactor = esPositivo ? detMenor : detMenor.neg()
    const termino = elem.mul(cofactor)
    total = total.add(termino)

    if (n > 3) {
      pasos.push(...subPasos)
    }

    pasos.push({
      texto: `Cofactor (${i + 1},${j + 1}) por ${etiquetaLinea}: ${signoStr}(${ratToString(elem)}) · det(M_${i + 1},${j + 1}) = ${signoStr}(${ratToString(elem)}) · (${ratToString(detMenor)}) = ${ratToString(termino)}`,
      matriz: menor,
    })
  }

  pasos.push({
    texto: `Suma de cofactores por ${etiquetaLinea}: det = ${ratToString(total)}`,
    matriz: clone(m),
  })

  return total
}

/**
 * Calcula el determinante exacto por expansión de cofactores eligiendo en cada nivel
 * la fila o columna con mayor cantidad de ceros.
 */
export function det(m: Matrix): DetResult {
  if (!isSquare(m)) {
    throw new Error('El determinante solo está definido para matrices cuadradas no vacías')
  }
  const pasos: Step[] = []
  const valor = detRecursive(m, pasos)
  return { valor, pasos }
}

function formatScaleStep(filaIdx: number, pivote: Rat): string {
  const f = `F${filaIdx + 1}`
  if (pivote.d === 1n) {
    if (pivote.n === -1n) return `${f} → −${f}`
    if (pivote.n > 1n) return `${f} → ${f}/${pivote.n.toString()}`
    return `${f} → −${f}/${(-pivote.n).toString()}`
  }
  const inv = pivote.inv()
  if (inv.d === 1n) {
    if (inv.n === -1n) return `${f} → −${f}`
    return `${f} → ${inv.n.toString()}${f}`
  }
  return `${f} → (${ratToString(inv)})${f}`
}

function formatElimStep(targetIdx: number, pivotIdx: number, factor: Rat): string {
  const ft = `F${targetIdx + 1}`
  const fp = `F${pivotIdx + 1}`
  const positivo = factor.n > 0n
  const mag = factor.abs()
  const op = positivo ? '−' : '+'

  if (mag.isOne()) {
    return `${ft} → ${ft} ${op} ${fp}`
  }
  if (mag.d === 1n) {
    return `${ft} → ${ft} ${op} ${mag.n.toString()}${fp}`
  }
  return `${ft} → ${ft} ${op} (${ratToString(mag)})${fp}`
}

/**
 * Ejecuta Gauss-Jordan en las primeras `maxPivotCols` columnas, devolviendo la matriz
 * reducida, los pasos anotados y la lista de columnas con pivote.
 */
function gaussJordan(
  m: Matrix,
  maxPivotCols: number,
): { resultado: Matrix; pasos: Step[]; pivotCols: number[] } {
  const [filas, cols] = assertRectangular(m)
  const A = clone(m)
  const pasos: Step[] = []
  const pivotCols: number[] = []
  const colLimit = Math.min(cols, maxPivotCols)

  let pivotRow = 0
  for (let col = 0; col < colLimit && pivotRow < filas; col++) {
    let found = -1
    for (let r = pivotRow; r < filas; r++) {
      if (!A[r][col].isZero()) {
        found = r
        break
      }
    }
    if (found === -1) continue

    if (found !== pivotRow) {
      const tmp = A[pivotRow]
      A[pivotRow] = A[found]
      A[found] = tmp
      pasos.push({
        texto: `F${pivotRow + 1} ↔ F${found + 1}`,
        matriz: clone(A),
      })
    }

    const pivote = A[pivotRow][col]
    if (!pivote.isOne()) {
      for (let j = 0; j < cols; j++) {
        A[pivotRow][j] = A[pivotRow][j].div(pivote)
      }
      pasos.push({
        texto: formatScaleStep(pivotRow, pivote),
        matriz: clone(A),
      })
    }

    for (let r = 0; r < filas; r++) {
      if (r === pivotRow) continue
      const factor = A[r][col]
      if (factor.isZero()) continue
      for (let j = 0; j < cols; j++) {
        A[r][j] = A[r][j].sub(factor.mul(A[pivotRow][j]))
      }
      pasos.push({
        texto: formatElimStep(r, pivotRow, factor),
        matriz: clone(A),
      })
    }

    pivotCols.push(col)
    pivotRow++
  }

  return { resultado: A, pasos, pivotCols }
}

/** Forma escalonada reducida por filas (Gauss-Jordan) con registro de pasos. */
export function rref(m: Matrix): RrefResult {
  const [, cols] = assertRectangular(m)
  const { resultado, pasos } = gaussJordan(m, cols)
  return { resultado, pasos }
}

/**
 * Calcula la inversa mediante Gauss-Jordan sobre la matriz aumentada [A | I].
 * Si A es singular, devuelve `inversa: null` junto con `atasco` explicando en qué fila se atoró.
 */
export function inverse(m: Matrix): InverseResult {
  if (!isSquare(m)) {
    throw new Error('La inversa solo está definida para matrices cuadradas no vacías')
  }
  const n = m.length
  const I = identity(n)
  const aumentada: Matrix = m.map((row, i) => [...row, ...I[i]])

  const { resultado, pasos, pivotCols } = gaussJordan(aumentada, n)

  if (pivotCols.length < n) {
    let filaAtasco = pivotCols.length
    for (let i = 0; i < n; i++) {
      let todaCeros = true
      for (let j = 0; j < n; j++) {
        if (!resultado[i][j].isZero()) {
          todaCeros = false
          break
        }
      }
      if (todaCeros) {
        filaAtasco = i
        break
      }
    }
    return {
      inversa: null,
      pasos,
      atasco: `Atasco en la fila ${filaAtasco + 1}: el lado izquierdo quedó con todos ceros, por lo que la matriz es singular y no tiene inversa.`,
    }
  }

  const inversa: Matrix = resultado.map((row) => row.slice(n))
  return { inversa, pasos }
}

/** Producto matricial exacto A · B. */
export function multiply(a: Matrix, b: Matrix): Matrix {
  const [fa, ca] = assertRectangular(a)
  const [fb, cb] = assertRectangular(b)
  if (fa === 0 || ca === 0 || fb === 0 || cb === 0) {
    throw new Error('No se pueden multiplicar matrices vacías')
  }
  if (ca !== fb) {
    throw new Error(
      `Dimensiones incompatibles para multiplicar: ${fa}×${ca} no puede multiplicarse por ${fb}×${cb}`,
    )
  }

  const res = zeros(fa, cb)
  for (let i = 0; i < fa; i++) {
    for (let j = 0; j < cb; j++) {
      let suma = Rat.ZERO
      for (let k = 0; k < ca; k++) {
        suma = suma.add(a[i][k].mul(b[k][j]))
      }
      res[i][j] = suma
    }
  }
  return res
}

/** Transpuesta de una matriz. */
export function transpose(m: Matrix): Matrix {
  const [filas, cols] = assertRectangular(m)
  if (filas === 0 || cols === 0) return []
  return Array.from({ length: cols }, (_, j) =>
    Array.from({ length: filas }, (_, i) => m[i][j]),
  )
}

/** Rango de una matriz: número de filas no nulas en su RREF. */
export function rank(m: Matrix): number {
  const [filas, cols] = assertRectangular(m)
  if (filas === 0 || cols === 0) return 0
  const { resultado } = rref(m)
  let r = 0
  for (let i = 0; i < filas; i++) {
    if (resultado[i].some((v) => !v.isZero())) {
      r++
    }
  }
  return r
}

/**
 * Resuelve el sistema lineal Ax = b mediante Gauss-Jordan sobre [A | b].
 * Clasifica el sistema en solución única, infinitas soluciones (con índices de variables libres)
 * o sin solución.
 */
export function solveRref(a: Matrix, b: Rat[]): SolveResult {
  const [filas, cols] = assertRectangular(a)
  if (filas === 0 || cols === 0) {
    throw new Error('La matriz de coeficientes no puede estar vacía')
  }
  if (b.length !== filas) {
    throw new Error(
      `Dimensiones incompatibles: A tiene ${filas} filas pero el vector b tiene ${b.length} entradas`,
    )
  }

  const aumentada: Matrix = a.map((row, i) => [...row, b[i]])
  const { resultado, pasos, pivotCols } = gaussJordan(aumentada, cols)

  for (let i = 0; i < filas; i++) {
    let coefCero = true
    for (let j = 0; j < cols; j++) {
      if (!resultado[i][j].isZero()) {
        coefCero = false
        break
      }
    }
    if (coefCero && !resultado[i][cols].isZero()) {
      return { tipo: 'ninguna', pasos }
    }
  }

  const pivotSet = new Set(pivotCols)
  const libres: number[] = []
  for (let j = 0; j < cols; j++) {
    if (!pivotSet.has(j)) libres.push(j)
  }

  if (libres.length > 0) {
    return { tipo: 'infinitas', libres, pasos }
  }

  const solucion: Rat[] = Array.from({ length: cols }, () => Rat.ZERO)
  for (let i = 0; i < pivotCols.length; i++) {
    solucion[pivotCols[i]] = resultado[i][cols]
  }
  return { tipo: 'unica', solucion, pasos }
}

/** Formatea una matriz en entorno LaTeX `bmatrix`. */
export function matrixToLatex(m: Matrix): string {
  const filas = m.map((row) => row.map(ratToLatex).join(' & ')).join(' \\\\ ')
  return `\\begin{bmatrix} ${filas} \\end{bmatrix}`
}

/** Formatea una matriz en texto plano con fracciones exactas. */
export function matrixToString(m: Matrix): string {
  return m.map((row) => row.map(ratToString).join(' ')).join('\n')
}
