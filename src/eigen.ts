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

// Módulo 3: valores y vectores propios.
// Polinomio característico (Faddeev–LeVerrier), raíces racionales, multiplicidades exactas y
// detección de matrices defectivas en aritmética exacta Rat; potencia, QR y Jacobi en float64.

import { gaussSolve, qrHouseholderNum } from './direct.ts'
import {
  type Matrix,
  clone,
  isSquare,
  multiply,
  rank,
  rref,
  zeros,
} from './matrix.ts'
import { Rat, parseRat, ratToLatex, ratToString } from './rational.ts'

export interface PowerIterStep {
  k: number
  lambda: number
}

export interface PowerMethodOptions {
  inverse?: boolean
}

export interface PowerMethodResult {
  lambda: number
  vector: number[]
  iteraciones: PowerIterStep[]
  convergio: boolean
  motivo?: string
}

export interface QrIterationResult {
  valores: number[]
  iteraciones: number
  convergio: boolean
  motivo?: string
}

export interface JacobiEigenResult {
  valores: number[]
  /** Matriz ortogonal Q cuyas columnas son los vectores propios ortonormales (A ≈ Q · D · Qᵀ). */
  vectores: number[][]
  iteraciones: number
  reconstruccion: number[][]
  residuo: number
  residuoOrtogonalidad: number
}

export interface MultiplicidadExacta {
  algebraica: number
  geometrica: number
}

export interface MultiplicidadResult {
  algebraica: number | null
  geometrica: number
  aviso?: string
}

function assertSquareExact(a: Matrix, contexto: string): number {
  if (!isSquare(a)) {
    const filas = a.length
    const cols = filas > 0 ? a[0].length : 0
    throw new Error(
      `${contexto} requiere una matriz cuadrada no vacía (recibida ${filas}×${cols}).`,
    )
  }
  return a.length
}

function assertSquareNum(a: number[][], contexto: string): number {
  const n = a.length
  if (n === 0 || a.some((row) => row.length !== n)) {
    const cols = n > 0 ? a[0].length : 0
    throw new Error(
      `${contexto} requiere una matriz cuadrada no vacía (recibida ${n}×${cols}).`,
    )
  }
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (!Number.isFinite(a[i][j])) {
        throw new Error(
          `La matriz A contiene un valor no finito en (${i + 1}, ${j + 1}) en ${contexto}.`,
        )
      }
    }
  }
  return n
}

function gcdBig(a: bigint, b: bigint): bigint {
  let x = a < 0n ? -a : a
  let y = b < 0n ? -b : b
  while (y !== 0n) {
    const t = x % y
    x = y
    y = t
  }
  return x
}

function lcmBig(a: bigint, b: bigint): bigint {
  if (a === 0n || b === 0n) return 0n
  const xa = a < 0n ? -a : a
  const xb = b < 0n ? -b : b
  return (xa / gcdBig(xa, xb)) * xb
}

function traceExact(m: Matrix): Rat {
  let tr = Rat.ZERO
  for (let i = 0; i < m.length; i++) {
    tr = tr.add(m[i][i])
  }
  return tr
}

function addScalarIdentity(m: Matrix, c: Rat): Matrix {
  const n = m.length
  const out = clone(m)
  if (c.isZero()) return out
  for (let i = 0; i < n; i++) {
    out[i][i] = out[i][i].add(c)
  }
  return out
}

/**
 * Comprueba si una matriz exacta `a` es simétrica (A = Aᵀ).
 * Si no lo es, devuelve el primer par `(i, j)` donde `a[i][j] ≠ a[j][i]`.
 */
export function isSymmetricExact(
  a: Matrix,
): { ok: true } | { ok: false; i: number; j: number } {
  const n = assertSquareExact(a, 'La comprobación de simetría')
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (!a[i][j].eq(a[j][i])) {
        return { ok: false, i, j }
      }
    }
  }
  return { ok: true }
}

/**
 * Calcula los coeficientes exactos del polinomio característico
 *   p(λ) = det(λI − A) = λⁿ + c₁λⁿ⁻¹ + … + cₙ
 * mediante el algoritmo de Faddeev–LeVerrier en aritmética exacta `Rat`.
 * Devuelve `[1, c₁, …, cₙ]` (coeficiente líder 1 primero).
 */
export function charPoly(a: Matrix): Rat[] {
  const n = assertSquareExact(a, 'El polinomio característico')
  const coefs: Rat[] = [Rat.ONE]

  let B = clone(a)
  let c = traceExact(B).neg()
  coefs.push(c)

  for (let k = 2; k <= n; k++) {
    const shifted = addScalarIdentity(B, c)
    B = multiply(a, shifted)
    c = traceExact(B).neg().div(Rat.fromInt(k))
    coefs.push(c)
  }

  return coefs
}

function trimLeadingZeros(coefs: Rat[]): Rat[] {
  let idx = 0
  while (idx < coefs.length - 1 && coefs[idx].isZero()) {
    idx++
  }
  return coefs.slice(idx)
}

/** Formatea un polinomio característico en LaTeX para KaTeX: `p(\lambda) = \lambda^2 - 4\lambda + 3`. */
export function charPolyLatex(coefs: Rat[]): string {
  if (coefs.length === 0) return 'p(\\lambda) = 0'
  const poly = trimLeadingZeros(coefs)
  const deg = poly.length - 1
  const parts: string[] = []

  for (let i = 0; i < poly.length; i++) {
    const c = poly[i]
    if (c.isZero()) continue

    const exp = deg - i
    const neg = c.n < 0n
    const mag = c.abs()
    const isFirst = parts.length === 0

    let term = ''
    if (exp === 0) {
      term = ratToLatex(mag)
    } else {
      const varPart = exp === 1 ? '\\lambda' : exp <= 9 ? `\\lambda^${exp}` : `\\lambda^{${exp}}`
      term = mag.isOne() ? varPart : `${ratToLatex(mag)}${varPart}`
    }

    if (isFirst) {
      parts.push(neg ? `-${term}` : term)
    } else {
      parts.push(neg ? ` - ${term}` : ` + ${term}`)
    }
  }

  return `p(\\lambda) = ${parts.length > 0 ? parts.join('') : '0'}`
}

/** Representación en texto plano del polinomio característico (respaldo sin KaTeX). */
export function charPolyPlain(coefs: Rat[]): string {
  if (coefs.length === 0) return 'p(λ) = 0'
  const poly = trimLeadingZeros(coefs)
  const deg = poly.length - 1
  const parts: string[] = []

  for (let i = 0; i < poly.length; i++) {
    const c = poly[i]
    if (c.isZero()) continue

    const exp = deg - i
    const neg = c.n < 0n
    const mag = c.abs()
    const isFirst = parts.length === 0

    let term = ''
    if (exp === 0) {
      term = ratToString(mag)
    } else {
      const varPart = exp === 1 ? 'λ' : `λ^${exp}`
      const coefStr = mag.d === 1n ? ratToString(mag) : `(${ratToString(mag)})`
      term = mag.isOne() ? varPart : `${coefStr}${varPart}`
    }

    if (isFirst) {
      parts.push(neg ? `-${term}` : term)
    } else {
      parts.push(neg ? ` - ${term}` : ` + ${term}`)
    }
  }

  return `p(λ) = ${parts.length > 0 ? parts.join('') : '0'}`
}

/** Evalúa p(x) de forma exacta mediante el esquema de Horner en `Rat`. */
export function evalPoly(coefs: Rat[], x: Rat): Rat {
  let acc = Rat.ZERO
  for (let i = 0; i < coefs.length; i++) {
    acc = acc.mul(x).add(coefs[i])
  }
  return acc
}

function toPrimitiveIntPoly(coefs: Rat[]): bigint[] {
  const poly = trimLeadingZeros(coefs)
  if (poly.length === 0) return [0n]

  let denLcm = 1n
  for (const c of poly) {
    denLcm = lcmBig(denLcm, c.d)
  }

  let ints = poly.map((c) => c.n * (denLcm / c.d))
  let content = 0n
  for (const v of ints) {
    content = gcdBig(content, v)
  }
  if (content > 1n) {
    ints = ints.map((v) => v / content)
  }
  if (ints[0] < 0n) {
    ints = ints.map((v) => -v)
  }
  return ints
}

const MAX_TRIAL_DIVISOR = 1_000_000n
const MAX_DIVISORS = 8192

function positiveDivisors(value: bigint): bigint[] {
  const n = value < 0n ? -value : value
  if (n === 0n) return []
  if (n === 1n) return [1n]

  const factors: { prime: bigint; exp: number }[] = []
  let rem = n

  let exp2 = 0
  while ((rem & 1n) === 0n) {
    rem >>= 1n
    exp2++
  }
  if (exp2 > 0) factors.push({ prime: 2n, exp: exp2 })

  let d = 3n
  while (d * d <= rem && d <= MAX_TRIAL_DIVISOR) {
    if (rem % d === 0n) {
      let e = 0
      while (rem % d === 0n) {
        rem /= d
        e++
      }
      factors.push({ prime: d, exp: e })
    }
    d += 2n
  }
  if (rem > 1n) {
    factors.push({ prime: rem, exp: 1 })
  }

  let divs: bigint[] = [1n]
  for (const { prime, exp } of factors) {
    const next: bigint[] = []
    let pPow = 1n
    for (let e = 0; e <= exp; e++) {
      for (const base of divs) {
        next.push(base * pPow)
        if (next.length >= MAX_DIVISORS) break
      }
      if (next.length >= MAX_DIVISORS) break
      pPow *= prime
    }
    divs = next
  }

  return divs
}

/**
 * División sintética exacta de un polinomio entero `ints` por `(q·λ − p)` con `gcd(|p|, q) = 1`.
 * Devuelve el cociente entero si `p/q` es raíz exacta, o `null` en caso contrario.
 */
function syntheticDivideInt(ints: bigint[], p: bigint, q: bigint): bigint[] | null {
  const deg = ints.length - 1
  if (deg <= 0) return null
  const out = new Array<bigint>(deg)
  let rem = ints[0]
  for (let i = 0; i < deg; i++) {
    if (rem % q !== 0n) return null
    const coeff = rem / q
    out[i] = coeff
    rem = ints[i + 1] + coeff * p
  }
  if (rem !== 0n) return null
  return out
}

/**
 * Encuentra todas las raíces racionales distintas de un polinomio con coeficientes `Rat`
 * aplicando el teorema de la raíz racional (divisores del término independiente entre divisores
 * del coeficiente líder) y verificando exactamente `p(r) = 0`.
 * Devuelve las raíces ordenadas de menor a mayor.
 */
export function rationalRoots(coefs: Rat[]): Rat[] {
  const poly = trimLeadingZeros(coefs)
  if (poly.length <= 1) return []

  let ints = toPrimitiveIntPoly(poly)
  const roots: Rat[] = []

  // Extraer raíz λ = 0 si el término independiente es nulo
  if (ints[ints.length - 1] === 0n) {
    roots.push(Rat.ZERO)
    while (ints.length > 1 && ints[ints.length - 1] === 0n) {
      ints = ints.slice(0, -1)
    }
  }

  if (ints.length <= 1) {
    return roots
  }

  let foundInPass = true
  while (foundInPass && ints.length > 1) {
    foundInPass = false
    const lead = ints[0] < 0n ? -ints[0] : ints[0]
    const tail = ints[ints.length - 1] < 0n ? -ints[ints.length - 1] : ints[ints.length - 1]
    const pDivs = positiveDivisors(tail)
    const qDivs = positiveDivisors(lead)

    outer: for (const pAbs of pDivs) {
      for (const q of qDivs) {
        if (gcdBig(pAbs, q) !== 1n) continue
        for (const sign of [1n, -1n] as const) {
          const p = sign * pAbs
          const deflated = syntheticDivideInt(ints, p, q)
          if (deflated !== null) {
            const root = new Rat(p, q)
            if (evalPoly(poly, root).isZero()) {
              if (!roots.some((r) => r.eq(root))) {
                roots.push(root)
              }
              let cur = deflated
              while (cur.length > 1) {
                const again = syntheticDivideInt(cur, p, q)
                if (again === null) break
                cur = again
              }
              ints = cur
              foundInPass = true
              break outer
            }
          }
        }
      }
    }
  }

  roots.sort((a, b) => a.cmp(b))
  return roots
}

// ---------------------------------------------------------------------------
// Álgebra de polinomios exacta en Q[λ] (para multiplicidad y esDefectiva)
// ---------------------------------------------------------------------------

function polyNormalizeMonic(p: Rat[]): Rat[] {
  const trimmed = trimLeadingZeros(p)
  if (trimmed.length === 0 || (trimmed.length === 1 && trimmed[0].isZero())) {
    return [Rat.ZERO]
  }
  const lead = trimmed[0]
  if (lead.isOne()) return trimmed
  return trimmed.map((c) => c.div(lead))
}

function polyIsZero(p: Rat[]): boolean {
  return p.every((c) => c.isZero())
}

function polyDerivative(p: Rat[]): Rat[] {
  const poly = trimLeadingZeros(p)
  const deg = poly.length - 1
  if (deg <= 0) return [Rat.ZERO]
  const out: Rat[] = []
  for (let i = 0; i < deg; i++) {
    out.push(poly[i].mul(Rat.fromInt(deg - i)))
  }
  return trimLeadingZeros(out)
}

function polyDivMod(dividend: Rat[], divisor: Rat[]): { q: Rat[]; r: Rat[] } {
  const a = trimLeadingZeros(dividend)
  const b = trimLeadingZeros(divisor)
  if (polyIsZero(b)) {
    throw new Error('División por el polinomio cero')
  }
  const degA = a.length - 1
  const degB = b.length - 1
  if (degA < degB || polyIsZero(a)) {
    return { q: [Rat.ZERO], r: a }
  }

  const rem = a.slice()
  const qLen = degA - degB + 1
  const q: Rat[] = Array.from({ length: qLen }, () => Rat.ZERO)
  const leadB = b[0]

  for (let i = 0; i < qLen; i++) {
    const factor = rem[i].div(leadB)
    q[i] = factor
    if (!factor.isZero()) {
      for (let j = 0; j <= degB; j++) {
        rem[i + j] = rem[i + j].sub(factor.mul(b[j]))
      }
    }
  }

  return { q: trimLeadingZeros(q), r: trimLeadingZeros(rem) }
}

function polyGcdMonic(a: Rat[], b: Rat[]): Rat[] {
  let x = polyNormalizeMonic(a)
  let y = polyNormalizeMonic(b)
  while (!polyIsZero(y)) {
    const { r } = polyDivMod(x, y)
    x = y
    y = polyIsZero(r) ? [Rat.ZERO] : polyNormalizeMonic(r)
  }
  return polyNormalizeMonic(x)
}

/** Evalúa un polinomio p(λ) en una matriz cuadrada A: p(A) = c₀Aᵈ + … + c_d I (exacto). */
function evalMatrixPoly(coefs: Rat[], a: Matrix): Matrix {
  const n = a.length
  const poly = trimLeadingZeros(coefs)
  if (poly.length === 0 || polyIsZero(poly)) {
    return zeros(n, n)
  }

  let acc = addScalarIdentity(zeros(n, n), poly[0])
  for (let i = 1; i < poly.length; i++) {
    acc = multiply(acc, a)
    if (!poly[i].isZero()) {
      acc = addScalarIdentity(acc, poly[i])
    }
  }
  return acc
}

function isZeroMatrix(m: Matrix): boolean {
  for (let i = 0; i < m.length; i++) {
    for (let j = 0; j < m[i].length; j++) {
      if (!m[i][j].isZero()) return false
    }
  }
  return true
}

function exactRootMultiplicity(coefs: Rat[], lambda: Rat): number {
  let cur = trimLeadingZeros(coefs)
  let mult = 0
  while (cur.length > 1) {
    const deg = cur.length - 1
    const next = new Array<Rat>(deg)
    let rem = cur[0]
    for (let i = 0; i < deg; i++) {
      next[i] = rem
      rem = cur[i + 1].add(rem.mul(lambda))
    }
    if (!rem.isZero()) break
    mult++
    cur = next
  }
  return mult
}

/** Calcula el rango numérico de una matriz flotante mediante Gauss con pivoteo total. */
function numericalRank(m: number[][], tol: number): number {
  const rows = m.length
  const cols = rows > 0 ? m[0].length : 0
  if (rows === 0 || cols === 0) return 0

  const work = m.map((row) => row.slice())
  let r = 0

  for (let step = 0; step < Math.min(rows, cols); step++) {
    let bestVal = 0
    let bestI = -1
    let bestJ = -1
    for (let i = r; i < rows; i++) {
      for (let j = r; j < cols; j++) {
        const av = Math.abs(work[i][j])
        if (av > bestVal) {
          bestVal = av
          bestI = i
          bestJ = j
        }
      }
    }

    if (bestVal <= tol || bestI === -1 || bestJ === -1) {
      break
    }

    if (bestI !== r) {
      const tmp = work[r]
      work[r] = work[bestI]
      work[bestI] = tmp
    }
    if (bestJ !== r) {
      for (let i = 0; i < rows; i++) {
        const tmp = work[i][r]
        work[i][r] = work[i][bestJ]
        work[i][bestJ] = tmp
      }
    }

    const piv = work[r][r]
    for (let i = r + 1; i < rows; i++) {
      const factor = work[i][r] / piv
      if (factor === 0) continue
      work[i][r] = 0
      for (let j = r + 1; j < cols; j++) {
        work[i][j] -= factor * work[r][j]
      }
    }

    r++
  }

  return r
}

/**
 * Calcula las multiplicidades algebraica y geométrica de un valor propio `lambda`.
 * - Si `lambda` es `Rat` (o un `number` que coincide con una raíz racional exacta):
 *   algebraica por multiplicidad exacta en el polinomio característico;
 *   geométrica por `n − rank(A − λI)` exacto con `rank` del núcleo.
 * - Si `lambda` es un `number` que no corresponde a una raíz racional:
 *   `algebraica: null` con aviso explicativo y `geometrica` estimada numéricamente.
 */
export function multiplicidades(a: Matrix, lambda: Rat): MultiplicidadExacta
export function multiplicidades(a: Matrix, lambda: number): MultiplicidadResult
export function multiplicidades(a: Matrix, lambda: Rat | number): MultiplicidadResult
export function multiplicidades(a: Matrix, lambda: Rat | number): MultiplicidadResult {
  const n = assertSquareExact(a, 'El cálculo de multiplicidades')
  const coefs = charPoly(a)

  if (lambda instanceof Rat) {
    const algebraica = exactRootMultiplicity(coefs, lambda)
    const shifted = addScalarIdentity(a, lambda.neg())
    const geometrica = n - rank(shifted)
    return { algebraica, geometrica }
  }

  if (!Number.isFinite(lambda)) {
    throw new Error(`El valor propio λ debe ser un número finito (recibido ${String(lambda)}).`)
  }

  const roots = rationalRoots(coefs)
  const matchTol = 1e-9 * Math.max(1, Math.abs(lambda))
  const exactMatch = roots.find((r) => Math.abs(r.toNumber() - lambda) <= matchTol)
  if (exactMatch) {
    const algebraica = exactRootMultiplicity(coefs, exactMatch)
    const shifted = addScalarIdentity(a, exactMatch.neg())
    const geometrica = n - rank(shifted)
    return { algebraica, geometrica }
  }

  // Raíz no racional: rango numérico de (A − λI)
  let maxAbs = 0
  const shiftedNum: number[][] = a.map((row, i) =>
    row.map((val, j) => {
      const v = val.toNumber() - (i === j ? lambda : 0)
      const av = Math.abs(v)
      if (av > maxAbs) maxAbs = av
      return v
    }),
  )
  const rankTol = Math.max(1e-7, Math.max(maxAbs, Math.abs(lambda), 1) * 1e-7)
  const numRank = numericalRank(shiftedNum, rankTol)

  return {
    algebraica: null,
    geometrica: Math.max(0, n - numRank),
    aviso:
      'La raíz no es racional; la multiplicidad algebraica exacta no puede determinarse con aritmética racional.',
  }
}

/**
 * Determina de forma exacta si una matriz cuadrada es defectiva (no diagonalizable sobre ℂ),
 * es decir, si existe algún valor propio cuya multiplicidad geométrica es estrictamente menor
 * que su multiplicidad algebraica.
 *
 * Usa la parte libre de cuadrados exacta s(λ) = p(λ) / gcd(p(λ), p'(λ)):
 * A es diagonalizable sobre ℂ si y solo si s(A) = 0, por lo que A es defectiva iff s(A) ≠ 0.
 */
export function esDefectiva(a: Matrix): boolean {
  assertSquareExact(a, 'La comprobación de matriz defectiva')
  const p = charPoly(a)
  const dp = polyDerivative(p)
  const g = polyGcdMonic(p, dp)
  // Si gcd(p, p') = 1, todas las raíces son simples => geom = alg = 1 => nunca defectiva.
  if (g.length <= 1) return false

  const { q: squareFree } = polyDivMod(p, g)
  const evalMat = evalMatrixPoly(squareFree, a)
  return !isZeroMatrix(evalMat)
}

/**
 * Calcula una base exacta del espacio propio ker(A − λI) para un valor propio racional `lambda`
 * mediante RREF del núcleo.
 */
export function exactEigenvectors(a: Matrix, lambda: Rat): Rat[][] {
  const n = assertSquareExact(a, 'El cálculo de vectores propios exactos')
  const shifted = addScalarIdentity(a, lambda.neg())
  const { resultado } = rref(shifted)

  const pivotColByRow: number[] = []
  const pivotColSet = new Set<number>()
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (!resultado[i][j].isZero()) {
        pivotColByRow.push(j)
        pivotColSet.add(j)
        break
      }
    }
  }

  const freeCols: number[] = []
  for (let j = 0; j < n; j++) {
    if (!pivotColSet.has(j)) freeCols.push(j)
  }

  return freeCols.map((freeCol) => {
    const vec: Rat[] = Array.from({ length: n }, () => Rat.ZERO)
    vec[freeCol] = Rat.ONE
    for (let rowIdx = 0; rowIdx < pivotColByRow.length; rowIdx++) {
      const pCol = pivotColByRow[rowIdx]
      vec[pCol] = resultado[rowIdx][freeCol].neg()
    }
    return vec
  })
}

// ---------------------------------------------------------------------------
// Utilidades numéricas compartidas (punto flotante)
// ---------------------------------------------------------------------------

function dotNum(u: number[], v: number[]): number {
  let s = 0
  for (let i = 0; i < u.length; i++) s += u[i] * v[i]
  return s
}

function norm2Num(v: number[]): number {
  return Math.sqrt(dotNum(v, v))
}

function matVecNum(a: number[][], v: number[]): number[] {
  const n = a.length
  const out = new Array<number>(n)
  for (let i = 0; i < n; i++) {
    let s = 0
    const row = a[i]
    for (let j = 0; j < n; j++) s += row[j] * v[j]
    out[i] = s
  }
  return out
}

function matMulNum(a: number[][], b: number[][]): number[][] {
  const n = a.length
  const out = Array.from({ length: n }, () => new Array<number>(n).fill(0))
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      let s = 0
      for (let k = 0; k < n; k++) {
        s += a[i][k] * b[k][j]
      }
      out[i][j] = s
    }
  }
  return out
}

function frobeniusDiffNum(a: number[][], b: number[][]): number {
  let sumSq = 0
  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < a[i].length; j++) {
      const diff = a[i][j] - b[i][j]
      sumSq += diff * diff
    }
  }
  return Math.sqrt(sumSq)
}

function canonicalizeUnitVector(v: number[]): number[] {
  const norm = norm2Num(v)
  if (norm === 0 || !Number.isFinite(norm)) return v.slice()
  let maxIdx = 0
  let maxAbs = Math.abs(v[0])
  for (let i = 1; i < v.length; i++) {
    const av = Math.abs(v[i])
    if (av > maxAbs + 1e-12) {
      maxAbs = av
      maxIdx = i
    }
  }
  const sign = v[maxIdx] < 0 ? -1 : 1
  return v.map((val) => {
    const scaled = (sign * val) / norm
    return Math.abs(scaled) < 1e-15 ? 0 : scaled
  })
}

/** Residuo propio ‖A v − λ v‖₂. */
export function eigenResidual(a: number[][], vector: number[], lambda: number): number {
  const av = matVecNum(a, vector)
  let sumSq = 0
  for (let i = 0; i < av.length; i++) {
    const diff = av[i] - lambda * vector[i]
    sumSq += diff * diff
  }
  return Math.sqrt(sumSq)
}

function numberToRat(v: number): Rat {
  if (!Number.isFinite(v)) {
    throw new Error(`No se puede convertir un valor no finito a racional: ${String(v)}`)
  }
  if (v === 0) return Rat.ZERO
  const parsed = parseRat(v.toPrecision(12))
  if (parsed === null) {
    throw new Error(`No se pudo convertir ${String(v)} a fracción exacta.`)
  }
  return parsed
}

/**
 * Método de la potencia con cociente de Rayleigh.
 * - Sin `shift` (o `shift = 0`) y sin `inverse`: converge al valor propio dominante (mayor |λ|).
 * - Con `shift` y sin `inverse`: potencia con desplazamiento sobre `(A − shift·I)`.
 * - Con `options.inverse = true`: potencia inversa resolviendo `(A − shift·I)y = x`
 *   en cada paso con `gaussSolve` del módulo 1 (converge al valor propio más cercano a `shift`).
 */
export function powerMethod(
  a: number[][],
  x0: number[],
  tol: number,
  maxIter: number,
  shift?: number,
  options?: PowerMethodOptions,
): PowerMethodResult {
  const n = assertSquareNum(a, 'El método de la potencia')
  if (x0.length !== n) {
    throw new Error(
      `Dimensiones incompatibles en el método de la potencia: A es ${n}×${n} pero x₀ tiene ${x0.length} entradas.`,
    )
  }
  if (x0.some((v) => !Number.isFinite(v))) {
    throw new Error('El vector inicial x₀ contiene valores no finitos.')
  }
  const initNorm = norm2Num(x0)
  if (initNorm === 0) {
    throw new Error('El vector inicial x₀ no puede ser el vector nulo.')
  }
  if (!Number.isFinite(tol) || tol <= 0) {
    throw new Error(`La tolerancia tol debe ser positiva (recibida ${String(tol)}).`)
  }
  if (!Number.isInteger(maxIter) || maxIter < 1) {
    throw new Error(
      `El máximo de iteraciones debe ser un entero mayor o igual a 1 (recibido ${String(maxIter)}).`,
    )
  }
  const shiftVal = shift ?? 0
  if (!Number.isFinite(shiftVal)) {
    throw new Error(`El desplazamiento shift debe ser finito (recibido ${String(shift)}).`)
  }

  const useInverse = options?.inverse === true

  let x = canonicalizeUnitVector(x0)
  let prevLambda = dotNum(x, matVecNum(a, x))
  const iteraciones: PowerIterStep[] = []

  // Para potencia inversa preparamos (A − shift·I) en Rat una sola vez para reutilizar gaussSolve
  let shiftedRat: Matrix | null = null
  if (useInverse) {
    shiftedRat = a.map((row, i) =>
      row.map((val, j) => numberToRat(val - (i === j ? shiftVal : 0))),
    )
  }

  for (let k = 1; k <= maxIter; k++) {
    let y: number[]
    if (useInverse && shiftedRat) {
      const bRat = x.map(numberToRat)
      try {
        const solved = gaussSolve(shiftedRat, bRat)
        y = solved.x.map((r) => r.toNumber())
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err)
        return {
          lambda: shiftVal,
          vector: x.slice(),
          iteraciones,
          convergio: false,
          motivo: `Potencia inversa detenida: (A − ${shiftVal}·I) es singular (${msg}). Si ${shiftVal} coincide con un valor propio exacto, ajusta ligeramente el desplazamiento.`,
        }
      }
    } else {
      y = new Array<number>(n)
      for (let i = 0; i < n; i++) {
        let s = 0
        const row = a[i]
        for (let j = 0; j < n; j++) {
          const entry = i === j ? row[j] - shiftVal : row[j]
          s += entry * x[j]
        }
        y[i] = s
      }
    }

    const normY = norm2Num(y)
    if (!Number.isFinite(normY)) {
      return {
        lambda: prevLambda,
        vector: x.slice(),
        iteraciones,
        convergio: false,
        motivo: `El método de la potencia desbordó numéricamente en la iteración ${k}.`,
      }
    }
    if (normY === 0) {
      // (A − shift·I)x = 0 => x es vector propio exacto con λ = shiftVal
      const lambdaExact = shiftVal
      iteraciones.push({ k, lambda: lambdaExact })
      return {
        lambda: lambdaExact,
        vector: x.slice(),
        iteraciones,
        convergio: true,
      }
    }

    const xNext = canonicalizeUnitVector(y)
    const lambdaK = dotNum(xNext, matVecNum(a, xNext))
    if (!Number.isFinite(lambdaK)) {
      return {
        lambda: prevLambda,
        vector: x.slice(),
        iteraciones,
        convergio: false,
        motivo: `Cociente de Rayleigh no finito en la iteración ${k}.`,
      }
    }

    iteraciones.push({ k, lambda: lambdaK })
    const res = eigenResidual(a, xNext, lambdaK)
    const delta = Math.abs(lambdaK - prevLambda)
    const effTol = Math.max(tol, 1e-14 * Math.max(1, Math.abs(lambdaK)))

    x = xNext
    prevLambda = lambdaK

    if (res < effTol && delta < effTol) {
      return {
        lambda: lambdaK,
        vector: x.slice(),
        iteraciones,
        convergio: true,
      }
    }
  }

  const finalRes = eigenResidual(a, x, prevLambda)
  return {
    lambda: prevLambda,
    vector: x.slice(),
    iteraciones,
    convergio: false,
    motivo: `El método de la potencia no convergió en ${maxIter} iteraciones (residuo ‖Av − λv‖₂ = ${finalRes.toExponential(2)} ≥ ${tol.toExponential(2)}). Puede deberse a valores propios de igual módulo, complejos, matriz defectiva o vector inicial ortogonal al espacio propio dominante.`,
  }
}

/**
 * Variante explícita del método de la potencia inversa con desplazamiento `shift`.
 * Resuelve `(A − shift·I)y = x` en cada paso reutilizando `gaussSolve` del módulo 1.
 */
export function inversePowerMethod(
  a: number[][],
  x0: number[],
  tol: number,
  maxIter: number,
  shift = 0,
): PowerMethodResult {
  return powerMethod(a, x0, tol, maxIter, shift, { inverse: true })
}

function maxStrictLowerAbs(m: number[][]): number {
  let maxVal = 0
  for (let i = 1; i < m.length; i++) {
    for (let j = 0; j < i; j++) {
      const av = Math.abs(m[i][j])
      if (av > maxVal) maxVal = av
    }
  }
  return maxVal
}

/**
 * Iteración QR sin desplazamiento reutilizando la factorización QR de Householder del módulo 1.
 * Itera A_{k+1} = R_k · Q_k (donde A_k = Q_k · R_k) hasta que todos los elementos bajo la
 * diagonal sean < tol. Si no converge en `maxIter`, devuelve la diagonal actual con `convergio: false`.
 */
export function qrIteration(
  a: number[][],
  tol: number,
  maxIter: number,
): QrIterationResult {
  const n = assertSquareNum(a, 'La iteración QR')
  if (!Number.isFinite(tol) || tol <= 0) {
    throw new Error(`La tolerancia tol debe ser positiva (recibida ${String(tol)}).`)
  }
  if (!Number.isInteger(maxIter) || maxIter < 1) {
    throw new Error(
      `El máximo de iteraciones debe ser un entero mayor o igual a 1 (recibido ${String(maxIter)}).`,
    )
  }

  let ak = a.map((row) => row.slice())

  if (n === 1 || maxStrictLowerAbs(ak) < tol) {
    return {
      valores: ak.map((row, i) => row[i]),
      iteraciones: 0,
      convergio: true,
    }
  }

  for (let k = 1; k <= maxIter; k++) {
    const { Q, R } = qrHouseholderNum(ak)
    ak = matMulNum(R, Q)

    let finite = true
    for (let i = 0; i < n && finite; i++) {
      for (let j = 0; j < n; j++) {
        if (!Number.isFinite(ak[i][j])) {
          finite = false
          break
        }
      }
    }
    if (!finite) {
      return {
        valores: ak.map((row, i) => row[i]),
        iteraciones: k,
        convergio: false,
        motivo: `La iteración QR produjo valores no finitos en la iteración ${k}.`,
      }
    }

    if (maxStrictLowerAbs(ak) < tol) {
      return {
        valores: ak.map((row, i) => row[i]),
        iteraciones: k,
        convergio: true,
      }
    }
  }

  const subMax = maxStrictLowerAbs(ak)
  return {
    valores: ak.map((row, i) => row[i]),
    iteraciones: maxIter,
    convergio: false,
    motivo: `La iteración QR sin desplazamiento no convergió en ${maxIter} iteraciones (máximo elemento bajo la diagonal = ${subMax.toExponential(2)} ≥ ${tol.toExponential(2)}). Esto suele ocurrir si la matriz tiene valores propios complejos o de igual módulo.`,
  }
}

/**
 * Método de Jacobi para matrices reales simétricas: aplica rotaciones ortogonales planas
 * que anulan en cada paso el mayor elemento fuera de la diagonal hasta diagonalizar A.
 * Devuelve valores propios (ordenados de mayor a menor), matriz ortogonal Q (columnas = vectores
 * propios ortonormales), reconstrucción `Q·D·Qᵀ` y residuos de reconstrucción y ortogonalidad.
 */
export function jacobiEigen(a: number[][]): JacobiEigenResult {
  const n = assertSquareNum(a, 'El método de Jacobi para matrices simétricas')

  let maxAbs = 0
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const av = Math.abs(a[i][j])
      if (av > maxAbs) maxAbs = av
    }
  }

  const symTol = Math.max(1, maxAbs) * 1e-10
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (Math.abs(a[i][j] - a[j][i]) > symTol) {
        throw new Error(
          `El método de Jacobi requiere una matriz simétrica (a_${i + 1},${j + 1} = ${a[i][j]} ≠ a_${j + 1},${i + 1} = ${a[j][i]}).`,
        )
      }
    }
  }

  // Trabajar sobre copia exactamente simetrizada sin mutar `a`
  const S: number[][] = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? a[i][i] : 0.5 * (a[i][j] + a[j][i]))),
  )
  const Q: number[][] = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)),
  )

  const tol = Math.max(1e-14, maxAbs * 1e-13)
  const maxIter = Math.max(100, 60 * n * n)
  let iteraciones = 0

  if (n > 1) {
    for (let it = 0; it < maxIter; it++) {
      let p = 0
      let q = 1
      let maxOff = Math.abs(S[0][1])
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          const av = Math.abs(S[i][j])
          if (av > maxOff) {
            maxOff = av
            p = i
            q = j
          }
        }
      }

      if (maxOff <= tol) break

      const app = S[p][p]
      const aqq = S[q][q]
      const apq = S[p][q]

      const tau = (aqq - app) / (2 * apq)
      const t =
        tau === 0
          ? 1
          : (tau > 0 ? 1 : -1) / (Math.abs(tau) + Math.sqrt(1 + tau * tau))
      const c = 1 / Math.sqrt(1 + t * t)
      const s = t * c

      for (let k = 0; k < n; k++) {
        if (k === p || k === q) continue
        const skp = S[k][p]
        const skq = S[k][q]
        const newKp = c * skp - s * skq
        const newKq = s * skp + c * skq
        S[k][p] = newKp
        S[p][k] = newKp
        S[k][q] = newKq
        S[q][k] = newKq
      }

      S[p][p] = app - t * apq
      S[q][q] = aqq + t * apq
      S[p][q] = 0
      S[q][p] = 0

      for (let i = 0; i < n; i++) {
        const qip = Q[i][p]
        const qiq = Q[i][q]
        Q[i][p] = c * qip - s * qiq
        Q[i][q] = s * qip + c * qiq
      }

      iteraciones++
    }
  }

  // Ordenar valores propios de mayor a menor y reordenar columnas de Q con signo canónico
  const order = Array.from({ length: n }, (_, idx) => idx).sort((i, j) => S[j][j] - S[i][i])
  const valores = order.map((idx) => (Math.abs(S[idx][idx]) < 1e-15 ? 0 : S[idx][idx]))

  const sortedQ: number[][] = Array.from({ length: n }, () => new Array<number>(n).fill(0))
  for (let col = 0; col < n; col++) {
    const srcCol = order[col]
    const vec = Array.from({ length: n }, (_, row) => Q[row][srcCol])
    const canon = canonicalizeUnitVector(vec)
    for (let row = 0; row < n; row++) {
      sortedQ[row][col] = canon[row]
    }
  }

  // Reconstrucción Q · D · Qᵀ
  const reconstruccion: number[][] = Array.from({ length: n }, () => new Array<number>(n).fill(0))
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      let s = 0
      for (let k = 0; k < n; k++) {
        s += sortedQ[i][k] * valores[k] * sortedQ[j][k]
      }
      reconstruccion[i][j] = s
    }
  }

  // Ortogonalidad Qᵀ · Q frente a I
  const qtq: number[][] = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => {
      let s = 0
      for (let k = 0; k < n; k++) s += sortedQ[k][i] * sortedQ[k][j]
      return s
    }),
  )
  const ident: number[][] = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)),
  )

  const residuo = frobeniusDiffNum(a, reconstruccion)
  const residuoOrtogonalidad = frobeniusDiffNum(qtq, ident)

  return {
    valores,
    vectores: sortedQ,
    iteraciones,
    reconstruccion,
    residuo,
    residuoOrtogonalidad,
  }
}
