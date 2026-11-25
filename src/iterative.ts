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

// Módulo 2: solucionadores iterativos numéricos (punto flotante) para sistemas Ax = b.
// Error reportado en cada iteración: norma euclidiana del residuo ‖Ax_k − b‖₂.

import { verifySpdCholesky } from './direct.ts'

export interface IterStep {
  k: number
  x: number[]
  error: number
}

export interface IterResult {
  x: number[]
  iteraciones: IterStep[]
  convergio: boolean
  motivo?: string
}

function isSquareNum(a: number[][]): boolean {
  const n = a.length
  if (n === 0) return false
  for (let i = 0; i < n; i++) {
    if (a[i].length !== n) return false
  }
  return true
}

function assertIterativeInputs(
  a: number[][],
  b: number[],
  x0: number[],
  tol: number,
  maxIter: number,
  contexto: string,
): number {
  if (!isSquareNum(a)) {
    const filas = a.length
    const cols = filas > 0 ? a[0].length : 0
    throw new Error(
      `${contexto} requiere una matriz cuadrada no vacía (recibida ${filas}×${cols}).`,
    )
  }
  const n = a.length
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (!Number.isFinite(a[i][j])) {
        throw new Error(`La matriz A contiene un valor no finito en (${i + 1}, ${j + 1}).`)
      }
    }
  }
  if (b.length !== n) {
    throw new Error(
      `Dimensiones incompatibles en ${contexto}: A es ${n}×${n} pero el vector b tiene ${b.length} entradas.`,
    )
  }
  if (b.some((v) => !Number.isFinite(v))) {
    throw new Error(`El vector b contiene valores no finitos en ${contexto}.`)
  }
  if (x0.length !== n) {
    throw new Error(
      `Dimensiones incompatibles en ${contexto}: el vector inicial x₀ debe tener ${n} entradas (recibidas ${x0.length}).`,
    )
  }
  if (x0.some((v) => !Number.isFinite(v))) {
    throw new Error(`El vector inicial x₀ contiene valores no finitos en ${contexto}.`)
  }
  if (!Number.isFinite(tol) || tol <= 0) {
    throw new Error(`La tolerancia tol debe ser un número positivo (recibida ${String(tol)}).`)
  }
  if (!Number.isInteger(maxIter) || maxIter < 1) {
    throw new Error(
      `El máximo de iteraciones debe ser un entero mayor o igual a 1 (recibido ${String(maxIter)}).`,
    )
  }
  return n
}

/** Norma euclidiana del residuo ‖Ax − b‖₂. */
export function residualNorm2(a: number[][], x: number[], b: number[]): number {
  let sumSq = 0
  for (let i = 0; i < a.length; i++) {
    let ax = 0
    const row = a[i]
    for (let j = 0; j < row.length; j++) {
      ax += row[j] * x[j]
    }
    const diff = ax - b[i]
    sumSq += diff * diff
  }
  return Math.sqrt(sumSq)
}

function dot(u: number[], v: number[]): number {
  let s = 0
  for (let i = 0; i < u.length; i++) {
    s += u[i] * v[i]
  }
  return s
}

function findZeroDiagonal(a: number[][]): number {
  for (let i = 0; i < a.length; i++) {
    if (a[i][i] === 0) return i
  }
  return -1
}

/**
 * Método de Jacobi.
 * Fórmula de actualización (componente a componente):
 *   x_i^{(k+1)} = (1 / a_{ii}) * (b_i - sum_{j != i} a_{ij} * x_j^{(k)})
 * En forma matricial: x^{(k+1)} = D^{-1} (b - (L + U) x^{(k)}).
 * Se detiene cuando el residuo ‖Ax_k − b‖₂ < tol o al alcanzar maxIter.
 */
export function jacobi(
  a: number[][],
  b: number[],
  x0: number[],
  tol: number,
  maxIter: number,
): IterResult {
  const n = assertIterativeInputs(a, b, x0, tol, maxIter, 'El método de Jacobi')
  const zeroIdx = findZeroDiagonal(a)
  if (zeroIdx !== -1) {
    return {
      x: x0.slice(),
      iteraciones: [],
      convergio: false,
      motivo: `El elemento diagonal a_${zeroIdx + 1},${zeroIdx + 1} es 0: Jacobi requiere diagonal no nula. Reordena filas o prueba un método directo.`,
    }
  }

  let x = x0.slice()
  const err0 = residualNorm2(a, x, b)
  if (err0 < tol) {
    return {
      x: x.slice(),
      iteraciones: [{ k: 0, x: x.slice(), error: err0 }],
      convergio: true,
    }
  }

  const iteraciones: IterStep[] = []
  for (let k = 1; k <= maxIter; k++) {
    const xNext = new Array<number>(n)
    for (let i = 0; i < n; i++) {
      let suma = 0
      const row = a[i]
      for (let j = 0; j < n; j++) {
        if (j !== i) suma += row[j] * x[j]
      }
      xNext[i] = (b[i] - suma) / row[i]
    }

    const error = residualNorm2(a, xNext, b)
    if (!Number.isFinite(error) || xNext.some((v) => !Number.isFinite(v))) {
      return {
        x: x.slice(),
        iteraciones,
        convergio: false,
        motivo: `Jacobi divergió (desbordamiento numérico en la iteración ${k}). Prueba Gauss-Seidel, SOR con ω < 1 o un método directo.`,
      }
    }

    x = xNext
    iteraciones.push({ k, x: x.slice(), error })
    if (error < tol) {
      return { x: x.slice(), iteraciones, convergio: true }
    }
  }

  const ultimoErr = iteraciones[iteraciones.length - 1]?.error ?? err0
  return {
    x: x.slice(),
    iteraciones,
    convergio: false,
    motivo: `No convergió tras ${maxIter} iteraciones (residuo final ‖Ax − b‖₂ = ${ultimoErr.toExponential(2)} ≥ ${tol.toExponential(2)}). Prueba otro método, aumenta el máximo de iteraciones o relaja la tolerancia.`,
  }
}

/**
 * Método de sobrerrelajación sucesiva (SOR) con parámetro 0 < ω < 2.
 * Fórmula de actualización (en el lugar, usando x_j^{(k+1)} para j < i):
 *   x_i^{(k+1)} = (1 - ω) x_i^{(k)} + (ω / a_{ii}) * (b_i - sum_{j < i} a_{ij} x_j^{(k+1)} - sum_{j > i} a_{ij} x_j^{(k)})
 * Con ω = 1 coincide exactamente con Gauss-Seidel.
 * Lanza `Error` en español si ω no está en (0, 2).
 */
export function sor(
  a: number[][],
  b: number[],
  x0: number[],
  omega: number,
  tol: number,
  maxIter: number,
): IterResult {
  if (!Number.isFinite(omega) || omega <= 0 || omega >= 2) {
    throw new Error(
      `El factor de relajación ω debe cumplir 0 < ω < 2 (recibido ${String(omega)}).`,
    )
  }
  const n = assertIterativeInputs(a, b, x0, tol, maxIter, 'El método SOR')
  const zeroIdx = findZeroDiagonal(a)
  if (zeroIdx !== -1) {
    return {
      x: x0.slice(),
      iteraciones: [],
      convergio: false,
      motivo: `El elemento diagonal a_${zeroIdx + 1},${zeroIdx + 1} es 0: SOR requiere diagonal no nula. Reordena filas o prueba un método directo.`,
    }
  }

  const x = x0.slice()
  const err0 = residualNorm2(a, x, b)
  if (err0 < tol) {
    return {
      x: x.slice(),
      iteraciones: [{ k: 0, x: x.slice(), error: err0 }],
      convergio: true,
    }
  }

  const iteraciones: IterStep[] = []
  for (let k = 1; k <= maxIter; k++) {
    const prev = x.slice()
    for (let i = 0; i < n; i++) {
      let suma = 0
      const row = a[i]
      for (let j = 0; j < n; j++) {
        if (j !== i) suma += row[j] * x[j]
      }
      const gs = (b[i] - suma) / row[i]
      x[i] = omega === 1 ? gs : (1 - omega) * x[i] + omega * gs
    }

    const error = residualNorm2(a, x, b)
    if (!Number.isFinite(error) || x.some((v) => !Number.isFinite(v))) {
      return {
        x: prev,
        iteraciones,
        convergio: false,
        motivo: `SOR (ω = ${omega}) divergió (desbordamiento numérico en la iteración ${k}). Prueba reducir ω, usar otro método o un solucionador directo.`,
      }
    }

    iteraciones.push({ k, x: x.slice(), error })
    if (error < tol) {
      return { x: x.slice(), iteraciones, convergio: true }
    }
  }

  const ultimoErr = iteraciones[iteraciones.length - 1]?.error ?? err0
  return {
    x: x.slice(),
    iteraciones,
    convergio: false,
    motivo: `No convergió tras ${maxIter} iteraciones (residuo final ‖Ax − b‖₂ = ${ultimoErr.toExponential(2)} ≥ ${tol.toExponential(2)}). Prueba ajustar ω, cambiar de método o relajar la tolerancia.`,
  }
}

/**
 * Método de Gauss-Seidel.
 * Fórmula de actualización (en el lugar, usando x_j^{(k+1)} para j < i):
 *   x_i^{(k+1)} = (1 / a_{ii}) * (b_i - sum_{j < i} a_{ij} x_j^{(k+1)} - sum_{j > i} a_{ij} x_j^{(k)})
 * Equivale a SOR con ω = 1.
 */
export function gaussSeidel(
  a: number[][],
  b: number[],
  x0: number[],
  tol: number,
  maxIter: number,
): IterResult {
  const n = assertIterativeInputs(a, b, x0, tol, maxIter, 'El método de Gauss-Seidel')
  const zeroIdx = findZeroDiagonal(a)
  if (zeroIdx !== -1) {
    return {
      x: x0.slice(),
      iteraciones: [],
      convergio: false,
      motivo: `El elemento diagonal a_${zeroIdx + 1},${zeroIdx + 1} es 0: Gauss-Seidel requiere diagonal no nula. Reordena filas o prueba un método directo.`,
    }
  }

  const res = sor(a, b, x0, 1, tol, maxIter)
  if (!res.convergio && res.motivo) {
    return {
      ...res,
      motivo: res.motivo
        .replace(/^SOR \(ω = 1\)/, 'Gauss-Seidel')
        .replace('ajustar ω, cambiar de método', 'probar otro método (como SOR o un método directo)'),
    }
  }
  // `n` validado arriba
  void n
  return res
}

/**
 * Método del gradiente conjugado (CG) para matrices simétricas definidas positivas (SPD).
 * Verifica simetría y definitud positiva (pivotes de Cholesky > 0) ANTES de iterar;
 * si la matriz no es SPD, devuelve `{ error }` explicando el motivo.
 * Fórmula de actualización:
 *   α_k = (r_kᵀ r_k) / (p_kᵀ A p_k)
 *   x_{k+1} = x_k + α_k p_k
 *   r_{k+1} = r_k − α_k A p_k
 *   β_k = (r_{k+1}ᵀ r_{k+1}) / (r_kᵀ r_k)
 *   p_{k+1} = r_{k+1} + β_k p_k
 */
export function conjugateGradient(
  a: number[][],
  b: number[],
  x0: number[],
  tol: number,
  maxIter: number,
): IterResult | { error: string } {
  const spd = verifySpdCholesky(a, 'El gradiente conjugado')
  if (!spd.ok) {
    return { error: spd.error }
  }

  const n = assertIterativeInputs(a, b, x0, tol, maxIter, 'El gradiente conjugado')
  const x = x0.slice()

  // r_0 = b - A x_0
  const r = new Array<number>(n)
  for (let i = 0; i < n; i++) {
    let ax = 0
    for (let j = 0; j < n; j++) ax += a[i][j] * x[j]
    r[i] = b[i] - ax
  }

  let rsOld = dot(r, r)
  const err0 = Math.sqrt(rsOld)
  if (err0 < tol) {
    return {
      x: x.slice(),
      iteraciones: [{ k: 0, x: x.slice(), error: err0 }],
      convergio: true,
    }
  }

  const p = r.slice()
  const iteraciones: IterStep[] = []

  for (let k = 1; k <= maxIter; k++) {
    const ap = new Array<number>(n)
    for (let i = 0; i < n; i++) {
      let s = 0
      const row = a[i]
      for (let j = 0; j < n; j++) s += row[j] * p[j]
      ap[i] = s
    }

    const pAp = dot(p, ap)
    if (!Number.isFinite(pAp) || pAp <= 0) {
      return {
        x: x.slice(),
        iteraciones,
        convergio: false,
        motivo: `La curvatura pᵀAp resultó ≤ 0 en la iteración ${k} (pérdida de definitud positiva por redondeo). Prueba un método directo.`,
      }
    }

    const alpha = rsOld / pAp
    for (let i = 0; i < n; i++) {
      x[i] += alpha * p[i]
      r[i] -= alpha * ap[i]
    }

    const rsNew = dot(r, r)
    const error = residualNorm2(a, x, b)
    if (!Number.isFinite(error) || x.some((v) => !Number.isFinite(v))) {
      return {
        x: x.slice(),
        iteraciones,
        convergio: false,
        motivo: `El gradiente conjugado perdió estabilidad numérica en la iteración ${k}. Prueba un método directo.`,
      }
    }

    iteraciones.push({ k, x: x.slice(), error })
    if (error < tol || rsNew === 0) {
      return { x: x.slice(), iteraciones, convergio: true }
    }

    const beta = rsNew / rsOld
    for (let i = 0; i < n; i++) {
      p[i] = r[i] + beta * p[i]
    }
    rsOld = rsNew
  }

  const ultimoErr = iteraciones[iteraciones.length - 1]?.error ?? err0
  return {
    x: x.slice(),
    iteraciones,
    convergio: false,
    motivo: `No convergió tras ${maxIter} iteraciones (residuo final ‖Ax − b‖₂ = ${ultimoErr.toExponential(2)} ≥ ${tol.toExponential(2)}). Prueba aumentar el máximo de iteraciones, relajar la tolerancia o usar otro método.`,
  }
}

/**
 * Comprueba si `a` es estrictamente diagonalmente dominante por filas:
 * para toda fila i, |a_{ii}| > sum_{j != i} |a_{ij}|.
 */
export function esDiagonalDominante(a: number[][]): boolean {
  if (!isSquareNum(a)) return false
  const n = a.length
  for (let i = 0; i < n; i++) {
    const diag = Math.abs(a[i][i])
    if (!Number.isFinite(diag)) return false
    let offSum = 0
    for (let j = 0; j < n; j++) {
      if (j === i) continue
      const v = Math.abs(a[i][j])
      if (!Number.isFinite(v)) return false
      offSum += v
    }
    if (!(diag > offSum)) return false
  }
  return true
}

/**
 * Reduce una matriz real cuadrada `h` (in-place) a forma de Hessenberg superior
 * mediante transformaciones de similitud ortogonales de Householder.
 * Preserva todos los valores propios.
 */
function reduceToUpperHessenberg(h: number[][]): void {
  const n = h.length
  for (let k = 0; k < n - 2; k++) {
    let normSq = 0
    for (let i = k + 1; i < n; i++) {
      normSq += h[i][k] * h[i][k]
    }
    if (normSq === 0) continue

    const norm = Math.sqrt(normSq)
    const alpha = h[k + 1][k] >= 0 ? -norm : norm
    const v0 = h[k + 1][k] - alpha
    const m = n - (k + 1)
    const u = new Array<number>(m)
    u[0] = 1
    for (let i = 1; i < m; i++) {
      u[i] = h[k + 1 + i][k] / v0
    }
    const beta = -v0 / alpha

    h[k + 1][k] = alpha
    for (let i = k + 2; i < n; i++) {
      h[i][k] = 0
    }

    // Multiplicación por la izquierda: (I - β u uᵀ) H
    for (let j = k + 1; j < n; j++) {
      let s = 0
      for (let i = 0; i < m; i++) {
        s += u[i] * h[k + 1 + i][j]
      }
      const factor = beta * s
      for (let i = 0; i < m; i++) {
        h[k + 1 + i][j] -= factor * u[i]
      }
    }

    // Multiplicación por la derecha: H (I - β u uᵀ)
    for (let i = 0; i < n; i++) {
      let s = 0
      for (let j = 0; j < m; j++) {
        s += h[i][k + 1 + j] * u[j]
      }
      const factor = beta * s
      for (let j = 0; j < m; j++) {
        h[i][k + 1 + j] -= factor * u[j]
      }
    }
  }
}

/**
 * Calcula el radio espectral (máximo módulo de los valores propios reales o complejos)
 * de una matriz en forma de Hessenberg superior mediante el algoritmo QR real
 * con doble desplazamiento implícito de Francis y desplazamientos excepcionales de Wilkinson.
 * Devuelve `null` si algún bloque no converge dentro de `maxIterPerEig`.
 */
function spectralRadiusHessenbergQR(h: number[][], maxIterPerEig: number): number | null {
  const n = h.length
  const eps = Number.EPSILON

  let normH = 0
  for (let i = 0; i < n; i++) {
    for (let j = Math.max(0, i - 1); j < n; j++) {
      normH += Math.abs(h[i][j])
    }
  }
  if (normH === 0) return 0

  let maxRadius = 0
  let nn = n - 1

  while (nn >= 0) {
    let its = 0
    while (true) {
      let l = nn
      while (l > 0) {
        let s = Math.abs(h[l - 1][l - 1]) + Math.abs(h[l][l])
        if (s === 0) s = normH
        if (Math.abs(h[l][l - 1]) <= eps * s) {
          h[l][l - 1] = 0
          break
        }
        l--
      }

      if (l === nn) {
        // Bloque 1×1: valor propio real
        const mod = Math.abs(h[nn][nn])
        if (!Number.isFinite(mod)) return Number.POSITIVE_INFINITY
        if (mod > maxRadius) maxRadius = mod
        nn--
        break
      }

      if (l === nn - 1) {
        // Bloque 2×2: dos valores propios reales o un par complejo conjugado
        const a00 = h[nn - 1][nn - 1]
        const a01 = h[nn - 1][nn]
        const a10 = h[nn][nn - 1]
        const a11 = h[nn][nn]
        const mid = 0.5 * (a00 + a11)
        const halfDiff = 0.5 * (a00 - a11)
        const disc = halfDiff * halfDiff + a01 * a10
        const mod =
          disc >= 0
            ? Math.abs(mid) + Math.sqrt(disc)
            : Math.hypot(mid, Math.sqrt(-disc))
        if (!Number.isFinite(mod)) return Number.POSITIVE_INFINITY
        if (mod > maxRadius) maxRadius = mod
        nn -= 2
        break
      }

      if (its >= maxIterPerEig) {
        return null
      }

      let x: number
      let y: number
      let w: number
      if (its > 0 && its % 10 === 0) {
        // Desplazamiento excepcional de Wilkinson para romper simetrías cíclicas
        const s = Math.abs(h[nn][nn - 1]) + Math.abs(h[nn - 1][nn - 2])
        x = 0.75 * s
        y = 0.75 * s
        w = -0.4375 * s * s
      } else {
        x = h[nn][nn]
        y = h[nn - 1][nn - 1]
        w = h[nn][nn - 1] * h[nn - 1][nn]
      }
      its++

      let p = (h[l][l] - x) * (h[l][l] - y) - w + h[l][l + 1] * h[l + 1][l]
      let q = h[l + 1][l] * (h[l][l] + h[l + 1][l + 1] - x - y)
      let r = h[l + 2][l + 1] * h[l + 1][l]
      let scale = Math.abs(p) + Math.abs(q) + Math.abs(r)
      if (scale === 0) continue
      p /= scale
      q /= scale
      r /= scale

      for (let k = l; k <= nn - 1; k++) {
        const notLast = k !== nn - 1
        if (k !== l) {
          p = h[k][k - 1]
          q = h[k + 1][k - 1]
          r = notLast ? h[k + 2][k - 1] : 0
          scale = Math.abs(p) + Math.abs(q) + Math.abs(r)
          if (scale === 0) continue
          p /= scale
          q /= scale
          r /= scale
        }

        let s = Math.sqrt(p * p + q * q + r * r)
        if (p < 0) s = -s
        if (s === 0) continue

        if (k !== l) {
          h[k][k - 1] = -s * scale
          h[k + 1][k - 1] = 0
          if (notLast) h[k + 2][k - 1] = 0
        }

        p += s
        const x1 = p / s
        const y1 = q / s
        const z1 = r / s
        q /= p
        r /= p

        for (let j = k; j <= nn; j++) {
          let val = h[k][j] + q * h[k + 1][j]
          if (notLast) {
            val += r * h[k + 2][j]
            h[k + 2][j] -= val * z1
          }
          h[k + 1][j] -= val * y1
          h[k][j] -= val * x1
        }

        const iMax = Math.min(nn, k + 3)
        for (let i = l; i <= iMax; i++) {
          let val = x1 * h[i][k] + y1 * h[i][k + 1]
          if (notLast) {
            val += z1 * h[i][k + 2]
            h[i][k + 2] -= val * r
          }
          h[i][k + 1] -= val * q
          h[i][k] -= val
        }
      }
    }
  }

  return maxRadius
}

/**
 * Respaldo por método de la potencia con múltiples vectores iniciales deterministas
 * (unos, bases canónicas, rampa y alternante), tomando la mayor estimación.
 */
function estimaRadioPotenciaMultiInicio(tj: number[][], steps: number): number {
  const n = tj.length
  const seeds: number[][] = [
    new Array<number>(n).fill(1),
    Array.from({ length: n }, (_, i) => i + 1),
    Array.from({ length: n }, (_, i) => (i % 2 === 0 ? 1 : -1)),
  ]
  for (let idx = 0; idx < n; idx++) {
    const e = new Array<number>(n).fill(0)
    e[idx] = 1
    seeds.push(e)
  }

  let best = 0
  for (const seed of seeds) {
    const seedNorm = Math.sqrt(dot(seed, seed))
    if (seedNorm === 0) continue
    let v = seed.map((val) => val / seedNorm)
    let prevNorm = 0
    let lastNorm = 0
    let annihilated = false

    for (let k = 0; k < steps; k++) {
      const w = new Array<number>(n)
      let normSq = 0
      for (let i = 0; i < n; i++) {
        let s = 0
        const row = tj[i]
        for (let j = 0; j < n; j++) s += row[j] * v[j]
        w[i] = s
        normSq += s * s
      }
      const norm = Math.sqrt(normSq)
      if (!Number.isFinite(norm)) return Number.POSITIVE_INFINITY
      if (norm === 0) {
        annihilated = true
        break
      }
      prevNorm = lastNorm
      lastNorm = norm
      v = w.map((val) => val / norm)
    }

    if (!annihilated) {
      const est = prevNorm > 0 ? Math.sqrt(prevNorm * lastNorm) : lastNorm
      if (est > best) best = est
    }
  }

  return best
}

/**
 * Calcula el radio espectral ρ(T_J) de la matriz de iteración de Jacobi
 * T_J = -D^{-1}(L + U) mediante reducción ortogonal a Hessenberg superior y
 * algoritmo QR real con doble desplazamiento implícito de Francis (con respaldo
 * de potencia multi-inicio si algún bloque no converge).
 * Devuelve `Infinity` si algún elemento diagonal es cero, hay valores no finitos
 * o la matriz no es cuadrada.
 */
export function estimaRadioEspectralJacobi(a: number[][], iteraciones = 50): number {
  if (!isSquareNum(a)) return Number.POSITIVE_INFINITY
  const n = a.length
  for (let i = 0; i < n; i++) {
    if (a[i][i] === 0 || !Number.isFinite(a[i][i])) return Number.POSITIVE_INFINITY
    for (let j = 0; j < n; j++) {
      if (!Number.isFinite(a[i][j])) return Number.POSITIVE_INFINITY
    }
  }
  if (n === 1) return 0

  // Construir T_J = -D^{-1}(L + U) y comprobar si es idénticamente cero
  let maxAbs = 0
  const tj: number[][] = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => {
      if (i === j) return 0
      const v = -a[i][j] / a[i][i]
      const absV = Math.abs(v)
      if (absV > maxAbs) maxAbs = absV
      return v
    }),
  )
  if (!Number.isFinite(maxAbs)) return Number.POSITIVE_INFINITY
  if (maxAbs === 0) return 0

  // Caso 2×2 exacto: T_J = [[0, c], [d, 0]] tiene valores propios ±√(cd)
  if (n === 2) {
    return Math.sqrt(Math.abs(tj[0][1] * tj[1][0]))
  }

  // Escalar para evitar desbordamiento/subdesbordamiento numérico en QR
  const scaled: number[][] = tj.map((row) => row.map((v) => v / maxAbs))
  reduceToUpperHessenberg(scaled)

  const maxIterPerEig = Math.max(60, Math.floor(iteraciones))
  const rhoScaled = spectralRadiusHessenbergQR(scaled, maxIterPerEig)
  if (rhoScaled !== null) {
    const rho = rhoScaled * maxAbs
    return Number.isFinite(rho) ? rho : Number.POSITIVE_INFINITY
  }

  // Respaldo determinista multi-inicio si QR no convergiera en algún bloque
  const steps = Math.max(2, Math.floor(iteraciones))
  return estimaRadioPotenciaMultiInicio(tj, steps)
}

const RHO_UNIT_TOL = 1e-12

/**
 * Estima el factor de relajación óptimo de SOR:
 *   ω_opt = 2 / (1 + sqrt(1 − ρ²))
 * a partir del radio espectral estimado ρ de Jacobi.
 * Devuelve `null` si ρ ≥ 1 (dentro de tolerancia numérica) o no es finito (no existe óptimo útil).
 */
export function omegaOptimo(a: number[][]): number | null {
  const rho = estimaRadioEspectralJacobi(a)
  if (!Number.isFinite(rho) || rho < 0 || rho >= 1 - RHO_UNIT_TOL) {
    return null
  }
  const rad = Math.max(0, 1 - rho * rho)
  return 2 / (1 + Math.sqrt(rad))
}
