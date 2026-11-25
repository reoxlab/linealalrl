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

// Módulo 1: solucionadores directos de sistemas lineales Ax = b.
// Gauss y LU operan con fracciones exactas (Rat); Cholesky y QR operan en punto flotante (number).

import {
  type Matrix,
  type Step,
  clone,
  identity,
  isSquare,
} from './matrix.ts'
import { Rat, ratToString } from './rational.ts'

export interface GaussResult {
  x: Rat[]
  pasos: Step[]
}

export interface LuFactorResult {
  L: Matrix
  U: Matrix
  P: Matrix
  pasos: Step[]
}

export interface LuSolveResult {
  L: Matrix
  U: Matrix
  P: Matrix
  pb: Rat[]
  y: Rat[]
  x: Rat[]
  pasos: Step[]
}

export type CholeskyFactorResult =
  | { L: number[][]; residuo: number }
  | { error: string }

export type CholeskySolveResult =
  | { L: number[][]; y: number[]; x: number[]; residuo: number; residuoFactor: number }
  | { error: string }

export interface QrFactorResult {
  Q: number[][]
  R: number[][]
  residuoQR: number
}

export interface QrSolveResult {
  Q: number[][]
  R: number[][]
  qtb: number[]
  x: number[]
  residuo: number
  residuoQR: number
}

export type DirectMethodKey = 'gauss' | 'lu' | 'cholesky' | 'qr'

export interface DirectCompareRow {
  metodo: string
  clave: DirectMethodKey
  aplicable: boolean
  motivo?: string
  x: Rat[] | number[]
  exacto: boolean
  residuo: number
  pasos?: Step[]
  operaciones: string
  cuandoUsar: string
}

function assertSquareNonEmpty(a: Matrix, contexto: string): number {
  if (!isSquare(a)) {
    const filas = a.length
    const cols = filas > 0 ? a[0].length : 0
    throw new Error(
      `${contexto} requiere una matriz cuadrada no vacía (recibida ${filas}×${cols}).`,
    )
  }
  return a.length
}

function assertSystemDims(a: Matrix, b: Rat[], contexto: string): number {
  const n = assertSquareNonEmpty(a, contexto)
  if (b.length !== n) {
    throw new Error(
      `Dimensiones incompatibles en ${contexto}: A es ${n}×${n} pero el vector b tiene ${b.length} entradas.`,
    )
  }
  return n
}

function formatElimOp(targetIdx: number, pivotIdx: number, factor: Rat): string {
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

function findPivotRow(m: Matrix, col: number, startRow: number): number {
  let bestRow = startRow
  let bestAbs = m[startRow][col].abs()
  for (let r = startRow + 1; r < m.length; r++) {
    const cand = m[r][col].abs()
    if (cand.cmp(bestAbs) > 0) {
      bestAbs = cand
      bestRow = r
    }
  }
  return bestRow
}

function matrixToNumbers(a: Matrix): number[][] {
  return a.map((row) => row.map((v) => v.toNumber()))
}

function vectorToNumbers(v: Rat[]): number[] {
  return v.map((r) => r.toNumber())
}

function identityNum(n: number): number[][] {
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)),
  )
}

function zerosNum(rows: number, cols: number): number[][] {
  return Array.from({ length: rows }, () => Array.from({ length: cols }, () => 0))
}

function matMulNum(a: number[][], b: number[][]): number[][] {
  const m = a.length
  const k = a[0].length
  const n = b[0].length
  const out = zerosNum(m, n)
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      let s = 0
      for (let t = 0; t < k; t++) {
        s += a[i][t] * b[t][j]
      }
      out[i][j] = s
    }
  }
  return out
}

function transposeNum(a: number[][]): number[][] {
  const m = a.length
  const n = a[0].length
  return Array.from({ length: n }, (_, j) =>
    Array.from({ length: m }, (_, i) => a[i][j]),
  )
}

function frobeniusDiff(a: number[][], b: number[][]): number {
  let sumSq = 0
  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < a[i].length; j++) {
      const diff = a[i][j] - b[i][j]
      sumSq += diff * diff
    }
  }
  return Math.sqrt(sumSq)
}

/**
 * Calcula la inversa numérica de una matriz cuadrada real `a` mediante eliminación
 * gaussiana con pivoteo parcial sobre la matriz aumentada `[A | I]` (aritmética float64).
 * No muta la matriz de entrada. Devuelve `null` si la matriz es singular o numéricamente singular.
 */
export function gaussInverseNum(a: number[][]): number[][] | null {
  const n = a.length
  if (n === 0 || a.some((row) => row.length !== n)) {
    const cols = n > 0 ? a[0].length : 0
    throw new Error(
      `La inversión numérica por Gauss requiere una matriz cuadrada no vacía (recibida ${n}×${cols}).`,
    )
  }

  let maxAbs = 0
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const v = a[i][j]
      if (!Number.isFinite(v)) {
        throw new Error(
          `La matriz A contiene un valor no finito en (${i + 1}, ${j + 1}).`,
        )
      }
      const av = Math.abs(v)
      if (av > maxAbs) maxAbs = av
    }
  }
  if (maxAbs === 0) return null

  // Escalamos por maxAbs para evitar desbordamiento/subdesbordamiento y comparar pivotes de forma relativa.
  const aug: number[][] = Array.from({ length: n }, (_, i) => {
    const row = new Array<number>(2 * n).fill(0)
    for (let j = 0; j < n; j++) {
      row[j] = a[i][j] / maxAbs
    }
    row[n + i] = 1
    return row
  })

  const pivotTol = Number.EPSILON * Math.max(16, n * 16)

  for (let k = 0; k < n; k++) {
    let pivotRow = k
    let bestAbs = Math.abs(aug[k][k])
    for (let r = k + 1; r < n; r++) {
      const cand = Math.abs(aug[r][k])
      if (cand > bestAbs) {
        bestAbs = cand
        pivotRow = r
      }
    }

    if (!Number.isFinite(bestAbs) || bestAbs <= pivotTol) {
      return null
    }

    if (pivotRow !== k) {
      const tmp = aug[k]
      aug[k] = aug[pivotRow]
      aug[pivotRow] = tmp
    }

    const pivote = aug[k][k]
    for (let j = k; j < 2 * n; j++) {
      aug[k][j] /= pivote
    }

    for (let r = 0; r < n; r++) {
      if (r === k) continue
      const factor = aug[r][k]
      if (factor === 0) continue
      aug[r][k] = 0
      for (let j = k + 1; j < 2 * n; j++) {
        aug[r][j] -= factor * aug[k][j]
      }
    }
  }

  // Deshacer el escalado: (A / maxAbs)^(-1) = maxAbs * A^(-1) => A^(-1) = invScaled / maxAbs
  const inv: number[][] = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => {
      const val = aug[i][n + j] / maxAbs
      return val
    }),
  )

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (!Number.isFinite(inv[i][j])) return null
    }
  }

  return inv
}

/** Norma euclidiana del residuo ‖Ax − b‖₂ en aritmética flotante. */
export function residualNorm(a: Matrix, x: number[], b: Rat[]): number {
  let sumSq = 0
  for (let i = 0; i < a.length; i++) {
    let ax = 0
    for (let j = 0; j < a[i].length; j++) {
      ax += a[i][j].toNumber() * x[j]
    }
    const diff = ax - b[i].toNumber()
    sumSq += diff * diff
  }
  return Math.sqrt(sumSq)
}

function formatBackSubExpr(
  varName: string,
  idx: number,
  rhs: Rat,
  terms: { coef: Rat; val: Rat }[],
  diag: Rat,
  result: Rat,
): string {
  const rhsStr = ratToString(rhs)
  const diagStr = ratToString(diag)
  const resStr = ratToString(result)

  if (terms.length === 0) {
    return `Sustitución hacia atrás (${varName}${idx + 1}): ${varName}${idx + 1} = (${rhsStr}) / (${diagStr}) = ${resStr}`
  }

  const subPart = terms
    .map((t) => `(${ratToString(t.coef)})(${ratToString(t.val)})`)
    .join(' − ')
  return `Sustitución hacia atrás (${varName}${idx + 1}): ${varName}${idx + 1} = [${rhsStr} − ${subPart}] / (${diagStr}) = ${resStr}`
}

function formatForwardSubExpr(
  idx: number,
  rhs: Rat,
  terms: { coef: Rat; val: Rat }[],
  result: Rat,
): string {
  const rhsStr = ratToString(rhs)
  const resStr = ratToString(result)
  if (terms.length === 0) {
    return `Sustitución hacia adelante (y${idx + 1}): y${idx + 1} = ${resStr}`
  }
  const subPart = terms
    .map((t) => `(${ratToString(t.coef)})(${ratToString(t.val)})`)
    .join(' − ')
  return `Sustitución hacia adelante (y${idx + 1}): y${idx + 1} = ${rhsStr} − ${subPart} = ${resStr}`
}

/**
 * Eliminación gaussiana exacta con pivoteo parcial y sustitución hacia atrás.
 * Cada intercambio de filas, eliminación y despeje hacia atrás queda anotado en `pasos`.
 * Lanza `Error` en español si la matriz no es cuadrada, las dimensiones no coinciden o es singular.
 */
export function gaussSolve(a: Matrix, b: Rat[]): GaussResult {
  const n = assertSystemDims(a, b, 'La eliminación de Gauss')
  const aug: Matrix = a.map((row, i) => [...row, b[i]])
  const pasos: Step[] = []

  for (let k = 0; k < n; k++) {
    const pivotRow = findPivotRow(aug, k, k)
    if (aug[pivotRow][k].isZero()) {
      throw new Error(
        `La matriz A es singular (pivote cero en la columna ${k + 1}): el sistema no tiene solución única.`,
      )
    }

    if (pivotRow !== k) {
      const tmp = aug[k]
      aug[k] = aug[pivotRow]
      aug[pivotRow] = tmp
      pasos.push({
        texto: `Pivoteo parcial: F${k + 1} ↔ F${pivotRow + 1}`,
        matriz: clone(aug),
      })
    }

    const pivote = aug[k][k]
    for (let i = k + 1; i < n; i++) {
      const elem = aug[i][k]
      if (elem.isZero()) continue
      const factor = elem.div(pivote)
      aug[i][k] = Rat.ZERO
      for (let j = k + 1; j <= n; j++) {
        aug[i][j] = aug[i][j].sub(factor.mul(aug[k][j]))
      }
      pasos.push({
        texto: formatElimOp(i, k, factor),
        matriz: clone(aug),
      })
    }
  }

  pasos.push({
    texto: 'Matriz aumentada en forma escalonada superior [U | c]',
    matriz: clone(aug),
  })

  const x: Rat[] = Array.from({ length: n }, () => Rat.ZERO)
  for (let i = n - 1; i >= 0; i--) {
    const diag = aug[i][i]
    if (diag.isZero()) {
      throw new Error(
        `La matriz A es singular (pivote cero en la fila ${i + 1}): el sistema no tiene solución única.`,
      )
    }
    let suma = Rat.ZERO
    const terms: { coef: Rat; val: Rat }[] = []
    for (let j = i + 1; j < n; j++) {
      if (!aug[i][j].isZero()) {
        suma = suma.add(aug[i][j].mul(x[j]))
        terms.push({ coef: aug[i][j], val: x[j] })
      }
    }
    const numerador = aug[i][n].sub(suma)
    x[i] = numerador.div(diag)
    pasos.push({
      texto: formatBackSubExpr('x', i, aug[i][n], terms, diag, x[i]),
      matriz: x.slice(i).map((v) => [v]),
    })
  }

  return { x, pasos }
}

/**
 * Factorización LU de Doolittle exacta con pivoteo parcial: P · A = L · U,
 * donde L es triangular inferior unitaria, U es triangular superior y P es la matriz de permutación.
 */
export function luFactor(a: Matrix): LuFactorResult {
  const n = assertSquareNonEmpty(a, 'La factorización LU')
  const U = clone(a)
  const L = identity(n)
  const P = identity(n)
  const pasos: Step[] = []

  for (let k = 0; k < n; k++) {
    const pivotRow = findPivotRow(U, k, k)
    if (U[pivotRow][k].isZero()) {
      pasos.push({
        texto: `Columna ${k + 1}: todos los candidatos a pivote son cero (U quedará singular).`,
        matriz: clone(U),
      })
      continue
    }

    if (pivotRow !== k) {
      const tmpU = U[k]
      U[k] = U[pivotRow]
      U[pivotRow] = tmpU

      const tmpP = P[k]
      P[k] = P[pivotRow]
      P[pivotRow] = tmpP

      for (let j = 0; j < k; j++) {
        const tmpL = L[k][j]
        L[k][j] = L[pivotRow][j]
        L[pivotRow][j] = tmpL
      }

      pasos.push({
        texto: `Pivoteo parcial: F${k + 1} ↔ F${pivotRow + 1} (actualiza P, U y columnas previas de L)`,
        matriz: clone(U),
      })
    }

    const pivote = U[k][k]
    for (let i = k + 1; i < n; i++) {
      const elem = U[i][k]
      if (elem.isZero()) continue
      const factor = elem.div(pivote)
      L[i][k] = factor
      U[i][k] = Rat.ZERO
      for (let j = k + 1; j < n; j++) {
        U[i][j] = U[i][j].sub(factor.mul(U[k][j]))
      }
      pasos.push({
        texto: `Multiplicador l_${i + 1},${k + 1} = ${ratToString(factor)} · ${formatElimOp(i, k, factor)}`,
        matriz: clone(U),
      })
    }
  }

  return { L, U, P, pasos }
}

/**
 * Resuelve Ax = b mediante factorización LU exacta (PA = LU) en dos pasos:
 * 1) Ly = Pb (sustitución hacia adelante)
 * 2) Ux = y  (sustitución hacia atrás)
 */
export function luSolve(a: Matrix, b: Rat[]): LuSolveResult {
  const n = assertSystemDims(a, b, 'La solución por LU')
  const { L, U, P, pasos: factorPasos } = luFactor(a)

  for (let i = 0; i < n; i++) {
    if (U[i][i].isZero()) {
      throw new Error(
        `La matriz A es singular (pivote u_${i + 1},${i + 1} = 0 en U): el sistema no tiene solución única.`,
      )
    }
  }

  const pasos: Step[] = [...factorPasos]

  // Pb
  const pb: Rat[] = Array.from({ length: n }, (_, i) => {
    let suma = Rat.ZERO
    for (let j = 0; j < n; j++) {
      if (!P[i][j].isZero()) {
        suma = suma.add(P[i][j].mul(b[j]))
      }
    }
    return suma
  })

  pasos.push({
    texto: 'Vector permutado P·b para resolver L·y = P·b',
    matriz: pb.map((v) => [v]),
  })

  // Sustitución hacia adelante: L y = P b (L tiene 1 en la diagonal)
  const y: Rat[] = Array.from({ length: n }, () => Rat.ZERO)
  for (let i = 0; i < n; i++) {
    let suma = Rat.ZERO
    const terms: { coef: Rat; val: Rat }[] = []
    for (let j = 0; j < i; j++) {
      if (!L[i][j].isZero()) {
        suma = suma.add(L[i][j].mul(y[j]))
        terms.push({ coef: L[i][j], val: y[j] })
      }
    }
    y[i] = pb[i].sub(suma)
    pasos.push({
      texto: formatForwardSubExpr(i, pb[i], terms, y[i]),
      matriz: y.slice(0, i + 1).map((v) => [v]),
    })
  }

  // Sustitución hacia atrás: U x = y
  const x: Rat[] = Array.from({ length: n }, () => Rat.ZERO)
  for (let i = n - 1; i >= 0; i--) {
    let suma = Rat.ZERO
    const terms: { coef: Rat; val: Rat }[] = []
    for (let j = i + 1; j < n; j++) {
      if (!U[i][j].isZero()) {
        suma = suma.add(U[i][j].mul(x[j]))
        terms.push({ coef: U[i][j], val: x[j] })
      }
    }
    x[i] = y[i].sub(suma).div(U[i][i])
    pasos.push({
      texto: formatBackSubExpr('x', i, y[i], terms, U[i][i], x[i]),
      matriz: x.slice(i).map((v) => [v]),
    })
  }

  return { L, U, P, pb, y, x, pasos }
}

/**
 * Verifica exactamente (en aritmética Rat) si una matriz simétrica es definida positiva
 * comprobando que todos los pivotes de eliminación sin intercambio (LDLᵀ) sean estrictamente > 0.
 */
function checkPositiveDefiniteExact(a: Matrix): { ok: true } | { ok: false; pivoteIdx: number; valor: Rat } {
  const n = a.length
  const work = clone(a)
  for (let k = 0; k < n; k++) {
    const pivote = work[k][k]
    if (pivote.cmp(Rat.ZERO) <= 0) {
      return { ok: false, pivoteIdx: k + 1, valor: pivote }
    }
    for (let i = k + 1; i < n; i++) {
      const elem = work[i][k]
      if (elem.isZero()) continue
      const factor = elem.div(pivote)
      for (let j = k + 1; j < n; j++) {
        work[i][j] = work[i][j].sub(factor.mul(work[k][j]))
      }
    }
  }
  return { ok: true }
}

/**
 * Calcula el factor numérico L de Cholesky (A = L·Lᵀ) verificando que todos los pivotes sean > tol.
 * Compartido entre `choleskyFactor` y `verifySpdCholesky` (usado por Gradiente Conjugado).
 */
function computeCholeskyFactorNum(
  aNum: number[][],
  pivTol = 0,
): { ok: true; L: number[][] } | { ok: false; pivoteIdx: number; valor: number } {
  const n = aNum.length
  const L = zerosNum(n, n)

  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let suma = 0
      for (let k = 0; k < j; k++) {
        suma += L[i][k] * L[j][k]
      }
      if (i === j) {
        const diag = aNum[i][i] - suma
        if (!Number.isFinite(diag) || !(diag > pivTol)) {
          return { ok: false, pivoteIdx: i + 1, valor: diag }
        }
        L[i][i] = Math.sqrt(diag)
      } else {
        L[i][j] = (aNum[i][j] - suma) / L[j][j]
      }
    }
  }

  return { ok: true, L }
}

/**
 * Verifica si una matriz numérica `a` es cuadrada, simétrica y definida positiva (SPD)
 * comprobando simetría y que todos los pivotes de Cholesky sean estrictamente positivos.
 */
export function verifySpdCholesky(
  a: number[][],
  metodo = 'Cholesky',
): { ok: true; L: number[][] } | { ok: false; error: string } {
  const filas = a.length
  const cols = filas > 0 ? a[0].length : 0
  if (filas === 0 || cols === 0 || filas !== cols || a.some((row) => row.length !== cols)) {
    return {
      ok: false,
      error: `La matriz A no es cuadrada (${filas}×${cols}). ${metodo} solo aplica a matrices cuadradas simétricas definidas positivas (SPD).`,
    }
  }

  const n = filas
  let maxAbs = 0
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const v = a[i][j]
      if (!Number.isFinite(v)) {
        return {
          ok: false,
          error: `La matriz A contiene un valor no finito en (${i + 1}, ${j + 1}).`,
        }
      }
      const av = Math.abs(v)
      if (av > maxAbs) maxAbs = av
    }
  }

  const symTol = Math.max(1, maxAbs) * 1e-12
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (Math.abs(a[i][j] - a[j][i]) > symTol) {
        return {
          ok: false,
          error: `La matriz A no es simétrica (a_${i + 1},${j + 1} = ${a[i][j]} ≠ a_${j + 1},${i + 1} = ${a[j][i]}). ${metodo} solo aplica a matrices simétricas definidas positivas (SPD); prueba Gauss-Seidel, SOR o factorización LU.`,
        }
      }
    }
  }

  const pivTol = maxAbs * 1e-14
  const res = computeCholeskyFactorNum(a, pivTol)
  if (!res.ok) {
    const valStr = Number.isFinite(res.valor) ? Number(res.valor.toPrecision(4)) : String(res.valor)
    return {
      ok: false,
      error: `El pivote de Cholesky ${res.pivoteIdx} es ${valStr} ≤ 0, por lo que la matriz no es definida positiva. ${metodo} solo aplica a matrices simétricas definidas positivas (SPD); prueba Gauss-Seidel, SOR o factorización LU.`,
    }
  }

  return res
}

/**
 * Factorización de Cholesky numérica A = L · Lᵀ (L triangular inferior con diagonal positiva).
 * Verifica primero la simetría y la definitud positiva en aritmética exacta;
 * si no aplica, devuelve `{ error }` explicando el motivo y sugiriendo usar LU.
 */
export function choleskyFactor(a: Matrix): CholeskyFactorResult {
  if (!isSquare(a)) {
    const filas = a.length
    const cols = filas > 0 ? a[0].length : 0
    return {
      error: `La matriz A no es cuadrada (${filas}×${cols}). Cholesky solo aplica a matrices cuadradas simétricas definidas positivas; usa factorización LU o RREF.`,
    }
  }

  const n = a.length

  // 1. Verificar simetría exacta A = Aᵀ antes de calcular
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (!a[i][j].eq(a[j][i])) {
        return {
          error: `La matriz A no es simétrica (a_${i + 1},${j + 1} = ${ratToString(a[i][j])} ≠ a_${j + 1},${i + 1} = ${ratToString(a[j][i])}). Cholesky solo aplica a matrices simétricas definidas positivas; usa factorización LU en su lugar.`,
        }
      }
    }
  }

  // 2. Verificar definitud positiva exacta (todos los pivotes > 0) antes de calcular raíces
  const pdCheck = checkPositiveDefiniteExact(a)
  if (!pdCheck.ok) {
    return {
      error: `El pivote ${pdCheck.pivoteIdx} es ${ratToString(pdCheck.valor)} ≤ 0, por lo que la matriz no es definida positiva. Cholesky solo aplica a matrices simétricas definidas positivas; usa factorización LU en su lugar.`,
    }
  }

  // 3. Calcular factor numérico L (punto flotante)
  const aNum = matrixToNumbers(a)
  const facNum = computeCholeskyFactorNum(aNum, 0)
  if (!facNum.ok) {
    return {
      error: `El pivote ${facNum.pivoteIdx} resultó ≤ 0 en aritmética flotante (matriz muy mal condicionada). Usa factorización LU o QR en su lugar.`,
    }
  }

  const L = facNum.L
  const llt = matMulNum(L, transposeNum(L))
  const residuo = frobeniusDiff(aNum, llt)
  return { L, residuo }
}

/**
 * Resuelve Ax = b por factorización de Cholesky numérica (L y = b, Lᵀ x = y).
 * Devuelve la solución aproximada y el residuo ‖Ax − b‖₂, o `{ error }` si Cholesky no aplica.
 */
export function choleskySolve(a: Matrix, b: Rat[]): CholeskySolveResult {
  const fac = choleskyFactor(a)
  if ('error' in fac) {
    return fac
  }
  const n = a.length
  if (b.length !== n) {
    return {
      error: `Dimensiones incompatibles en Cholesky: A es ${n}×${n} pero el vector b tiene ${b.length} entradas.`,
    }
  }

  const { L } = fac
  const bNum = vectorToNumbers(b)

  // L y = b
  const y = new Array<number>(n).fill(0)
  for (let i = 0; i < n; i++) {
    let suma = 0
    for (let j = 0; j < i; j++) {
      suma += L[i][j] * y[j]
    }
    y[i] = (bNum[i] - suma) / L[i][i]
  }

  // Lᵀ x = y
  const x = new Array<number>(n).fill(0)
  for (let i = n - 1; i >= 0; i--) {
    let suma = 0
    for (let j = i + 1; j < n; j++) {
      suma += L[j][i] * x[j]
    }
    x[i] = (y[i] - suma) / L[i][i]
  }

  const residuo = residualNorm(a, x, b)
  return { L, y, x, residuo, residuoFactor: fac.residuo }
}

/**
 * Factorización QR numérica mediante reflexiones de Householder sobre matriz flotante:
 * A = Q · R, con Q ortogonal (QᵀQ = I) y R triangular superior.
 * Reutilizada tanto por `qrHouseholder` (módulo 1) como por la iteración QR (módulo 3).
 */
export function qrHouseholderNum(aNum: number[][]): QrFactorResult {
  const m = aNum.length
  const n = m > 0 ? aNum[0].length : 0
  if (m === 0 || n === 0) {
    throw new Error('La factorización QR requiere una matriz no vacía.')
  }
  for (let i = 1; i < m; i++) {
    if (aNum[i].length !== n) {
      throw new Error('La matriz de entrada no es rectangular.')
    }
  }
  if (m < n) {
    throw new Error(
      `La factorización QR requiere al menos tantas filas como columnas (m ≥ n); recibida ${m}×${n}.`,
    )
  }

  const R = aNum.map((row) => row.slice())
  const Q = identityNum(m)

  const limit = Math.min(m - 1, n)
  for (let k = 0; k < limit; k++) {
    const len = m - k
    const x = new Array<number>(len)
    let normSq = 0
    for (let i = 0; i < len; i++) {
      x[i] = R[k + i][k]
      normSq += x[i] * x[i]
    }
    const normX = Math.sqrt(normSq)
    if (normX === 0) continue

    const alpha = x[0] >= 0 ? -normX : normX
    const v = x.slice()
    v[0] = x[0] - alpha

    let normV2 = 0
    for (let i = 0; i < len; i++) {
      normV2 += v[i] * v[i]
    }
    if (normV2 === 0) continue

    // Aplicar H_k a R por la izquierda: R <- H_k R
    for (let j = k; j < n; j++) {
      let dot = 0
      for (let i = 0; i < len; i++) {
        dot += v[i] * R[k + i][j]
      }
      const scale = (2 * dot) / normV2
      for (let i = 0; i < len; i++) {
        R[k + i][j] -= scale * v[i]
      }
    }

    // Limpiar ruido numérico debajo de la diagonal en la columna k
    for (let i = k + 1; i < m; i++) {
      R[i][k] = 0
    }

    // Acumular Q <- Q H_k por la derecha
    for (let i = 0; i < m; i++) {
      let dot = 0
      for (let j = 0; j < len; j++) {
        dot += Q[i][k + j] * v[j]
      }
      const scale = (2 * dot) / normV2
      for (let j = 0; j < len; j++) {
        Q[i][k + j] -= scale * v[j]
      }
    }
  }

  // Normalizar signo de la diagonal de R (R[k][k] >= 0) manteniendo Q ortogonal y A = Q R
  const diagLen = Math.min(m, n)
  for (let k = 0; k < diagLen; k++) {
    if (R[k][k] < 0) {
      for (let j = k; j < n; j++) {
        R[k][j] = R[k][j] === 0 ? 0 : -R[k][j]
      }
      for (let i = 0; i < m; i++) {
        Q[i][k] = Q[i][k] === 0 ? 0 : -Q[i][k]
      }
    }
  }

  const qrProd = matMulNum(Q, R)
  const residuoQR = frobeniusDiff(aNum, qrProd)
  return { Q, R, residuoQR }
}

/**
 * Factorización QR numérica mediante reflexiones de Householder: A = Q · R,
 * con Q ortogonal (QᵀQ = I) y R triangular superior.
 */
export function qrHouseholder(a: Matrix): QrFactorResult {
  const m = a.length
  const n = m > 0 ? a[0].length : 0
  if (m === 0 || n === 0) {
    throw new Error('La factorización QR requiere una matriz no vacía.')
  }
  for (let i = 1; i < m; i++) {
    if (a[i].length !== n) {
      throw new Error('La matriz de entrada no es rectangular.')
    }
  }
  if (m < n) {
    throw new Error(
      `La factorización QR requiere al menos tantas filas como columnas (m ≥ n); recibida ${m}×${n}.`,
    )
  }
  return qrHouseholderNum(matrixToNumbers(a))
}

/**
 * Resuelve el sistema cuadrado Ax = b por factorización QR (Householder):
 * calcula Qᵀb y resuelve Rx = Qᵀb por sustitución hacia atrás.
 */
export function qrSolve(a: Matrix, b: Rat[]): QrSolveResult {
  const n = assertSystemDims(a, b, 'La solución por QR')
  const { Q, R, residuoQR } = qrHouseholder(a)

  let maxDiag = 0
  for (let i = 0; i < n; i++) {
    const d = Math.abs(R[i][i])
    if (d > maxDiag) maxDiag = d
  }
  const singularTol = maxDiag > 0 ? maxDiag * 1e-12 : 1e-14

  for (let i = 0; i < n; i++) {
    if (Math.abs(R[i][i]) <= singularTol) {
      throw new Error(
        `La matriz A es singular (o numéricamente singular: |r_${i + 1},${i + 1}| ≈ ${Math.abs(R[i][i]).toExponential(2)}): el sistema no tiene solución única.`,
      )
    }
  }

  const bNum = vectorToNumbers(b)
  const qtb = new Array<number>(n).fill(0)
  for (let i = 0; i < n; i++) {
    let suma = 0
    for (let j = 0; j < n; j++) {
      suma += Q[j][i] * bNum[j]
    }
    qtb[i] = suma
  }

  const x = new Array<number>(n).fill(0)
  for (let i = n - 1; i >= 0; i--) {
    let suma = 0
    for (let j = i + 1; j < n; j++) {
      suma += R[i][j] * x[j]
    }
    x[i] = (qtb[i] - suma) / R[i][i]
  }

  const residuo = residualNorm(a, x, b)
  return { Q, R, qtb, x, residuo, residuoQR }
}

function approxOpsLabel(n: number, formula: string, mults: number): string {
  const rounded = Math.max(1, Math.round(mults))
  return `${formula} (≈ ${rounded} mult./div. para n = ${n})`
}

/**
 * Resuelve el mismo sistema Ax = b con Gauss, LU, Cholesky (si aplica) y QR,
 * devolviendo una tabla comparativa con solución, residuo, costo aproximado y recomendación de uso.
 * Si Cholesky no aplica, incluye el motivo en su fila en lugar de fallar.
 */
export function directCompare(a: Matrix, b: Rat[]): DirectCompareRow[] {
  const n = assertSystemDims(a, b, 'El comparador de solucionadores directos')
  const n3 = n * n * n

  const gauss = gaussSolve(a, b)
  const lu = luSolve(a, b)
  const chol = choleskySolve(a, b)
  const qr = qrSolve(a, b)

  const rows: DirectCompareRow[] = [
    {
      metodo: 'Eliminación de Gauss',
      clave: 'gauss',
      aplicable: true,
      x: gauss.x,
      exacto: true,
      residuo: 0,
      pasos: gauss.pasos,
      operaciones: approxOpsLabel(n, '~n³/3', n3 / 3),
      cuandoUsar:
        'Un solo vector b en sistemas generales pequeños o medianos; con fracciones exactas no acumula error de redondeo.',
    },
    {
      metodo: 'Factorización LU (PA = LU)',
      clave: 'lu',
      aplicable: true,
      x: lu.x,
      exacto: true,
      residuo: 0,
      pasos: lu.pasos,
      operaciones: approxOpsLabel(n, '~n³/3 (+ n² por cada b)', n3 / 3),
      cuandoUsar:
        'Varios sistemas con la misma matriz A y distintos vectores b: factoriza una sola vez y reutiliza L y U.',
    },
    'error' in chol
      ? {
          metodo: 'Cholesky (A = LLᵀ)',
          clave: 'cholesky',
          aplicable: false,
          motivo: chol.error,
          x: [],
          exacto: false,
          residuo: Number.NaN,
          operaciones: approxOpsLabel(n, '~n³/6', n3 / 6),
          cuandoUsar:
            'Solo para matrices simétricas definidas positivas (SPD): cuesta la mitad que LU y no requiere pivoteo.',
        }
      : {
          metodo: 'Cholesky (A = LLᵀ)',
          clave: 'cholesky',
          aplicable: true,
          x: chol.x,
          exacto: false,
          residuo: chol.residuo,
          operaciones: approxOpsLabel(n, '~n³/6', n3 / 6),
          cuandoUsar:
            'Matrices simétricas definidas positivas (SPD): cuesta la mitad que LU y es estable sin pivoteo.',
        },
    {
      metodo: 'QR (Householder)',
      clave: 'qr',
      aplicable: true,
      x: qr.x,
      exacto: false,
      residuo: qr.residuo,
      operaciones: approxOpsLabel(n, '~2n³/3', (2 * n3) / 3),
      cuandoUsar:
        'Sistemas mal condicionados o mínimos cuadrados: las reflexiones ortogonales preservan la norma y mejoran la estabilidad.',
    },
  ]

  return rows
}
