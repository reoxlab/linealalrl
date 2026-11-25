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

// Módulo 4: normas vectoriales y matriciales, número de condición κ(A),
// veredicto didáctico, cota de error de la solución, equilibrado por filas y matrices de Hilbert.

import { gaussInverseNum } from './direct.ts'
import { jacobiEigen } from './eigen.ts'

export type VectorNorm = 1 | 2 | 'inf'
export type MatrixNorm = 1 | 2 | 'inf' | 'fro'
export type CondVerdictLevel = 'bien' | 'moderada' | 'mal'

export interface CondNumberResult {
  kappa: number
  inversaNorma: number
}

export interface CondVerdict {
  nivel: CondVerdictLevel
  texto: string
}

export interface EquilibrateResult {
  matriz: number[][]
  escalas: number[]
}

const POWER_NORM2_MAX_ITER = 100
const POWER_NORM2_REL_TOL = 1e-10
const POWER_NORM2_RES_TOL = 1e-9

function assertRectangularFinite(a: number[][], contexto: string): [number, number] {
  const m = a.length
  const n = m > 0 ? a[0].length : 0
  if (m === 0 || n === 0) {
    throw new Error(`${contexto} requiere una matriz no vacía (recibida ${m}×${n}).`)
  }
  for (let i = 0; i < m; i++) {
    if (a[i].length !== n) {
      throw new Error(
        `La matriz no es rectangular en ${contexto}: la fila ${i + 1} tiene ${a[i].length} columnas en lugar de ${n}.`,
      )
    }
    for (let j = 0; j < n; j++) {
      if (!Number.isFinite(a[i][j])) {
        throw new Error(
          `La matriz contiene un valor no finito en (${i + 1}, ${j + 1}) en ${contexto}.`,
        )
      }
    }
  }
  return [m, n]
}

function assertSquareFinite(a: number[][], contexto: string): number {
  const [m, n] = assertRectangularFinite(a, contexto)
  if (m !== n) {
    throw new Error(
      `${contexto} requiere una matriz cuadrada no vacía (recibida ${m}×${n}).`,
    )
  }
  return n
}

function dot(u: number[], v: number[]): number {
  let s = 0
  for (let i = 0; i < u.length; i++) s += u[i] * v[i]
  return s
}

function norm2Scaled(v: number[]): number {
  let maxAbs = 0
  for (let i = 0; i < v.length; i++) {
    const av = Math.abs(v[i])
    if (av > maxAbs) maxAbs = av
  }
  if (maxAbs === 0) return 0
  let sumSq = 0
  for (let i = 0; i < v.length; i++) {
    const r = v[i] / maxAbs
    sumSq += r * r
  }
  return maxAbs * Math.sqrt(sumSq)
}

function matVec(m: number[][], v: number[]): number[] {
  const n = m.length
  const out = new Array<number>(n)
  for (let i = 0; i < n; i++) {
    const row = m[i]
    let s = 0
    for (let j = 0; j < row.length; j++) {
      s += row[j] * v[j]
    }
    out[i] = s
  }
  return out
}

function assertValidVectorNorm(p: VectorNorm): void {
  if (p !== 1 && p !== 2 && p !== 'inf') {
    throw new Error(
      `Norma vectorial no válida (recibida ${String(p)}). Se admiten 1, 2 o 'inf'.`,
    )
  }
}

function assertValidMatrixNorm(p: MatrixNorm): void {
  if (p !== 1 && p !== 2 && p !== 'inf' && p !== 'fro') {
    throw new Error(
      `Norma matricial no válida (recibida ${String(p)}). Se admiten 1, 2, 'inf' o 'fro'.`,
    )
  }
}

/**
 * Calcula la norma vectorial p ∈ {1, 2, 'inf'} de un vector numérico `x`:
 * - p = 1:   ‖x‖₁ = Σ |xᵢ|
 * - p = 2:   ‖x‖₂ = √(Σ xᵢ²)
 * - p = 'inf': ‖x‖∞ = máx |xᵢ|
 */
export function normVector(x: number[], p: VectorNorm): number {
  assertValidVectorNorm(p)
  for (let i = 0; i < x.length; i++) {
    if (!Number.isFinite(x[i])) {
      throw new Error(`El vector contiene un valor no finito en la posición ${i + 1}.`)
    }
  }
  if (x.length === 0) return 0

  if (p === 1) {
    let sum = 0
    for (let i = 0; i < x.length; i++) {
      sum += Math.abs(x[i])
    }
    return sum
  }

  if (p === 'inf') {
    let maxVal = 0
    for (let i = 0; i < x.length; i++) {
      const av = Math.abs(x[i])
      if (av > maxVal) maxVal = av
    }
    return maxVal
  }

  return norm2Scaled(x)
}

function frobeniusNormInternal(a: number[][]): number {
  let maxAbs = 0
  for (let i = 0; i < a.length; i++) {
    const row = a[i]
    for (let j = 0; j < row.length; j++) {
      const av = Math.abs(row[j])
      if (av > maxAbs) maxAbs = av
    }
  }
  if (maxAbs === 0) return 0

  let sumSq = 0
  for (let i = 0; i < a.length; i++) {
    const row = a[i]
    for (let j = 0; j < row.length; j++) {
      const r = row[j] / maxAbs
      sumSq += r * r
    }
  }
  return maxAbs * Math.sqrt(sumSq)
}

/**
 * Calcula la norma 2 matricial (mayor valor singular σ_max(A) = √λ_max(AᵀA))
 * mediante el método de la potencia sobre la matriz simétrica semidefinida positiva AᵀA
 * con un máximo de 100 iteraciones y semillas deterministas para evitar ortogonalidad accidental.
 * Si el método de la potencia no converge dentro de las 100 iteraciones (o una semilla sin
 * convergencia sugiere un autovalor mayor), recurre al respaldo determinista exacto con
 * `jacobiEigen` sobre AᵀA.
 */
function spectralNormPower(a: number[][]): number {
  const m = a.length
  const n = a[0].length

  let maxAbs = 0
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      const av = Math.abs(a[i][j])
      if (av > maxAbs) maxAbs = av
    }
  }
  if (maxAbs === 0) return 0
  if (m === 1 && n === 1) return maxAbs

  // Formamos M = (A / maxAbs)ᵀ (A / maxAbs) (n×n simétrica PSD) para evitar desbordamiento
  const ata: number[][] = Array.from({ length: n }, () => new Array<number>(n).fill(0))
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      let s = 0
      for (let r = 0; r < m; r++) {
        s += (a[r][i] / maxAbs) * (a[r][j] / maxAbs)
      }
      ata[i][j] = s
      ata[j][i] = s
    }
  }

  // Semillas deterministas: vector de unos, rampa, alternante y base canónica
  // (garantiza que al menos una semilla tenga proyección no nula sobre el subespacio dominante).
  const seeds: number[][] = [
    new Array<number>(n).fill(1),
    Array.from({ length: n }, (_, idx) => idx + 1),
    Array.from({ length: n }, (_, idx) => (idx % 2 === 0 ? 1 : -1)),
  ]
  const canonicalLimit = Math.min(n, 12)
  for (let idx = 0; idx < canonicalLimit; idx++) {
    const e = new Array<number>(n).fill(0)
    e[idx] = 1
    seeds.push(e)
  }

  let bestLambda = -1
  let maxUnconvergedLambda = -1
  let anyConverged = false

  for (const seed of seeds) {
    const sNorm = norm2Scaled(seed)
    if (sNorm === 0) continue
    let v = seed.map((val) => val / sNorm)
    let prevLambda = 0
    let converged = false
    let lambda = 0

    for (let k = 1; k <= POWER_NORM2_MAX_ITER; k++) {
      const w = matVec(ata, v)
      lambda = dot(v, w)
      const wNorm = norm2Scaled(w)

      if (!Number.isFinite(wNorm) || !Number.isFinite(lambda)) {
        converged = false
        break
      }
      if (wNorm === 0) {
        lambda = 0
        converged = true
        break
      }

      // Residuo ‖M v − λ v‖₂
      let resSq = 0
      for (let i = 0; i < n; i++) {
        const diff = w[i] - lambda * v[i]
        resSq += diff * diff
      }
      const res = Math.sqrt(resSq)

      const scaleRef = Math.max(1, Math.abs(lambda))
      const relChange = Math.abs(lambda - prevLambda) / scaleRef
      v = w.map((val) => val / wNorm)
      prevLambda = lambda

      if (res <= POWER_NORM2_RES_TOL * scaleRef || (k >= 2 && relChange <= POWER_NORM2_REL_TOL)) {
        converged = true
        break
      }
    }

    if (converged && lambda >= 0 && Number.isFinite(lambda)) {
      anyConverged = true
      if (lambda > bestLambda) {
        bestLambda = lambda
      }
    } else if (Number.isFinite(lambda) && lambda > maxUnconvergedLambda) {
      maxUnconvergedLambda = lambda
    }
  }

  if (anyConverged && bestLambda >= 0 && maxUnconvergedLambda <= bestLambda - 1e-9) {
    const sigma = maxAbs * Math.sqrt(bestLambda)
    if (Number.isFinite(sigma)) return sigma
  }

  // Respaldo determinista exacto: AᵀA es simétrica semidefinida positiva;
  // diagonalizamos con Jacobi para obtener el mayor autovalor σ²_max / maxAbs².
  const jac = jacobiEigen(ata)
  const maxVal = Math.max(0, ...jac.valores)
  const sigma = maxAbs * Math.sqrt(maxVal)
  if (!Number.isFinite(sigma)) {
    throw new Error('No se pudo calcular la norma 2 matricial: el resultado no es finito.')
  }
  return sigma
}

/**
 * Calcula la norma matricial de `a`:
 * - `1`:     máxima suma absoluta por columna:  ‖A‖₁ = máx_j Σ_i |aᵢⱼ|
 * - `'inf'`: máxima suma absoluta por fila:     ‖A‖∞ = máx_i Σ_j |aᵢⱼ|
 * - `'fro'`: norma de Frobenius:                ‖A‖_F = √(Σ_i Σ_j aᵢⱼ²)
 * - `2`:     norma espectral (mayor valor singular σ_max vía potencia sobre AᵀA,
 *            con respaldo determinista exacto mediante `jacobiEigen` sobre AᵀA).
 */
export function normMatrix(a: number[][], p: MatrixNorm): number {
  assertValidMatrixNorm(p)
  const [m, n] = assertRectangularFinite(a, 'El cálculo de la norma matricial')

  if (p === 1) {
    let maxColSum = 0
    for (let j = 0; j < n; j++) {
      let colSum = 0
      for (let i = 0; i < m; i++) {
        colSum += Math.abs(a[i][j])
      }
      if (colSum > maxColSum) maxColSum = colSum
    }
    return maxColSum
  }

  if (p === 'inf') {
    let maxRowSum = 0
    for (let i = 0; i < m; i++) {
      let rowSum = 0
      const row = a[i]
      for (let j = 0; j < n; j++) {
        rowSum += Math.abs(row[j])
      }
      if (rowSum > maxRowSum) maxRowSum = rowSum
    }
    return maxRowSum
  }

  if (p === 'fro') {
    return frobeniusNormInternal(a)
  }

  return spectralNormPower(a)
}

/**
 * Calcula el número de condición κ_p(A) = ‖A‖_p · ‖A⁻¹‖_p en la norma indicada `p`.
 * La inversa A⁻¹ se calcula numéricamente mediante eliminación gaussiana con pivoteo parcial
 * reutilizando `gaussInverseNum` de `direct.ts`.
 * Si A es singular (o numéricamente singular), devuelve `{ kappa: Infinity, inversaNorma: Infinity }`.
 */
export function condNumber(a: number[][], p: MatrixNorm): CondNumberResult {
  assertValidMatrixNorm(p)
  assertSquareFinite(a, 'El cálculo del número de condición')

  const inv = gaussInverseNum(a)
  if (inv === null) {
    return { kappa: Number.POSITIVE_INFINITY, inversaNorma: Number.POSITIVE_INFINITY }
  }

  const normA = normMatrix(a, p)
  const inversaNorma = normMatrix(inv, p)
  const kappa = normA * inversaNorma

  if (!Number.isFinite(kappa)) {
    return {
      kappa: Number.POSITIVE_INFINITY,
      inversaNorma: Number.isFinite(inversaNorma) ? inversaNorma : Number.POSITIVE_INFINITY,
    }
  }

  return { kappa, inversaNorma }
}

function formatKappaCompact(kappa: number): string {
  if (!Number.isFinite(kappa)) return '∞'
  if (kappa >= 1e5 || (kappa > 0 && kappa < 1e-3)) {
    return kappa.toExponential(2)
  }
  return Number(kappa.toPrecision(4)).toString()
}

/**
 * Clasifica el número de condición κ(A) en tres niveles didácticos:
 * - `'bien'`     si κ < 100
 * - `'moderada'` si 100 ≤ κ < 10⁴
 * - `'mal'`      si κ ≥ 10⁴ (o ∞)
 * Incluye una explicación en palabras llanas de la regla práctica κ ~ 10ᵏ → pérdida de hasta k dígitos decimales.
 */
export function veredictoCondicion(kappa: number): CondVerdict {
  if (!Number.isFinite(kappa) || kappa < 0) {
    return {
      nivel: 'mal',
      texto:
        'Matriz singular (κ = ∞): no existe A⁻¹ y el sistema Ax = b no tiene solución única. Como regla práctica, κ ~ 10ᵏ indica que se pueden perder k dígitos decimales; con κ = ∞ se pierde toda la precisión y cualquier perturbación mínima hace que la solución no exista o cambie sin límite.',
    }
  }

  const kDigits = kappa > 1 ? Math.log10(kappa) : 0
  const kLabel = kDigits < 0.05 ? '0' : kDigits.toFixed(1)
  const kappaStr = formatKappaCompact(kappa)

  if (kappa < 100) {
    return {
      nivel: 'bien',
      texto: `Bien condicionada (κ ≈ ${kappaStr}): la solución es estable frente a pequeños errores en los datos. Como regla práctica, κ ~ 10ᵏ significa que se pueden perder hasta k dígitos decimales al resolver el sistema; aquí se perderían a lo sumo ~${kLabel} dígitos.`,
    }
  }

  if (kappa < 1e4) {
    return {
      nivel: 'moderada',
      texto: `Condicionamiento moderado (κ ≈ ${kappaStr}): el error de los datos puede amplificarse de forma apreciable. Recuerda que κ ~ 10ᵏ significa que se pueden perder alrededor de k dígitos decimales al resolver el sistema (aquí ~${kLabel} dígitos); conviene revisar el escalado o usar mayor precisión.`,
    }
  }

  return {
    nivel: 'mal',
    texto: `Mal condicionada (κ ≈ ${kappaStr}): pequeñas perturbaciones en A o b provocan grandes errores en x. Como κ ~ 10ᵏ significa que se pueden perder alrededor de k dígitos decimales al resolver el sistema, aquí se pueden perder ~${kLabel} dígitos de precisión (de los ~16 dígitos de punto flotante).`,
  }
}

/**
 * Evalúa la cota clásica de sensibilidad del sistema lineal Ax = b:
 *   ‖δx‖ / ‖x‖ ≤ κ(A) · (‖δb‖ / ‖b‖)
 */
export function estimaErrorSolucion(kappa: number, errorRelativoB: number): number {
  if (Number.isNaN(kappa) || kappa < 0) {
    throw new Error(`El número de condición κ debe ser no negativo (recibido ${String(kappa)}).`)
  }
  if (!Number.isFinite(errorRelativoB) || errorRelativoB < 0) {
    throw new Error(
      `El error relativo de b debe ser un número finito no negativo (recibido ${String(errorRelativoB)}).`,
    )
  }
  if (!Number.isFinite(kappa)) {
    return Number.POSITIVE_INFINITY
  }
  return kappa * errorRelativoB
}

/**
 * Construye la matriz de Hilbert n×n con H[i][j] = 1 / (i + j + 1) (índices desde 0).
 * Es el ejemplo clásico de matriz simétrica definida positiva cuyo número de condición crece exponencialmente con n.
 */
export function hilbert(n: number): number[][] {
  if (!Number.isInteger(n) || n < 1) {
    throw new Error(`El orden n de la matriz de Hilbert debe ser un entero ≥ 1 (recibido ${String(n)}).`)
  }
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => 1 / (i + j + 1)),
  )
}

/**
 * Equilibrado de filas: divide cada fila de `a` por su mayor valor absoluto
 * (norma ∞ de la fila). Si una fila es idénticamente nula, su factor de escala se mantiene en 1.
 * No muta la matriz de entrada.
 */
export function equilibrarFilas(a: number[][]): EquilibrateResult {
  const [m, n] = assertRectangularFinite(a, 'El equilibrado por filas')
  const escalas = new Array<number>(m)
  const matriz: number[][] = new Array<number[]>(m)

  for (let i = 0; i < m; i++) {
    let maxRow = 0
    for (let j = 0; j < n; j++) {
      const av = Math.abs(a[i][j])
      if (av > maxRow) maxRow = av
    }
    const scale = maxRow > 0 ? maxRow : 1
    escalas[i] = scale
    matriz[i] = a[i].map((v) => (maxRow > 0 ? v / scale : 0))
  }

  return { matriz, escalas }
}
