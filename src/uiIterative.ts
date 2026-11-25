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

// Interfaz interactiva del módulo 2: solucionadores iterativos
// (Jacobi, Gauss-Seidel, SOR, Gradiente conjugado, comparador y gráfica de convergencia).

import {
  type IterResult,
  type IterStep,
  conjugateGradient,
  esDiagonalDominante,
  estimaRadioEspectralJacobi,
  gaussSeidel,
  jacobi,
  omegaOptimo,
  sor,
} from './iterative.ts'
import { renderLatex } from './katexRender.ts'
import { type Matrix, isSquare, solveRref } from './matrix.ts'
import { parsePastedAugmented } from './paste.ts'
import { type Rat, parseRat, ratToDecimal, ratToLatex, ratToString } from './rational.ts'
import { MAX_GRID_DIM } from './share.ts'
import {
  type TrackFn,
  createMatrixGrid,
  createResultFrame,
  setResultError,
} from './ui.ts'

export type IterativeSelectorValue = 'jacobi' | 'gauss-seidel' | 'sor' | 'cg' | 'compare'

export interface IterativePreset {
  id: string
  nombre: string
  descripcion: string
  a: string[][]
  b: string[]
  x0?: string[]
}

export const ITERATIVE_PRESETS: readonly IterativePreset[] = [
  {
    id: 'dom-3x3',
    nombre: 'Diagonal dominante 3×3 (SPD)',
    descripcion:
      'Sistema estrictamente diagonalmente dominante y simétrico definido positivo con solución exacta x = (1, 1, 1). Los 4 métodos convergen.',
    a: [
      ['4', '1', '1'],
      ['1', '5', '1'],
      ['1', '1', '6'],
    ],
    b: ['6', '7', '8'],
  },
  {
    id: 'poisson-4x4',
    nombre: 'Tridiagonal Poisson 4×4 (SPD)',
    descripcion:
      'Matriz simétrica definida positiva clásica (ρ ≈ 0.81). Mueve el deslizador de ω en SOR cerca de ω óptimo (≈ 1.26) para acelerar la convergencia.',
    a: [
      ['2', '-1', '0', '0'],
      ['-1', '2', '-1', '0'],
      ['0', '-1', '2', '-1'],
      ['0', '0', '-1', '2'],
    ],
    b: ['1', '0', '0', '1'],
  },
  {
    id: 'no-dom-2x2',
    nombre: 'No diagonal dominante 2×2 (diverge)',
    descripcion:
      'Sistema con solución x = (1, 1) donde A no es diagonal dominante (ρ ≈ 1.22 ≥ 1): Jacobi diverge y se muestra el aviso correspondiente.',
    a: [
      ['1', '2'],
      ['3', '4'],
    ],
    b: ['3', '7'],
  },
  {
    id: 'nosym-dom-3x3',
    nombre: 'No simétrica diagonal dominante 3×3',
    descripcion:
      'Solución exacta x = (1, 1, 1). Jacobi, Gauss-Seidel y SOR convergen, mientras que Gradiente conjugado explica por qué no aplica.',
    a: [
      ['5', '1', '1'],
      ['0', '4', '1'],
      ['1', '0', '3'],
    ],
    b: ['7', '5', '4'],
  },
]

const METHOD_FORMULAS: Record<
  Exclude<IterativeSelectorValue, 'compare'>,
  { titulo: string; tex: string; fallback: string }
> = {
  jacobi: {
    titulo: 'Fórmula de actualización de Jacobi',
    tex: 'x_i^{(k+1)} = \\frac{1}{a_{ii}} \\left( b_i - \\sum_{j \\neq i} a_{ij}\\, x_j^{(k)} \\right)',
    fallback: 'x_i^(k+1) = (1 / a_ii) * (b_i - sum_{j != i} a_ij * x_j^(k))',
  },
  'gauss-seidel': {
    titulo: 'Fórmula de actualización de Gauss-Seidel',
    tex: 'x_i^{(k+1)} = \\frac{1}{a_{ii}} \\left( b_i - \\sum_{j < i} a_{ij}\\, x_j^{(k+1)} - \\sum_{j > i} a_{ij}\\, x_j^{(k)} \\right)',
    fallback:
      'x_i^(k+1) = (1 / a_ii) * (b_i - sum_{j < i} a_ij * x_j^(k+1) - sum_{j > i} a_ij * x_j^(k))',
  },
  sor: {
    titulo: 'Fórmula de actualización de SOR (0 < ω < 2)',
    tex: 'x_i^{(k+1)} = (1 - \\omega)\\, x_i^{(k)} + \\frac{\\omega}{a_{ii}} \\left( b_i - \\sum_{j < i} a_{ij}\\, x_j^{(k+1)} - \\sum_{j > i} a_{ij}\\, x_j^{(k)} \\right)',
    fallback:
      'x_i^(k+1) = (1 - ω) x_i^(k) + (ω / a_ii) * (b_i - sum_{j < i} a_ij * x_j^(k+1) - sum_{j > i} a_ij * x_j^(k))',
  },
  cg: {
    titulo: 'Fórmula de actualización del Gradiente conjugado (SPD)',
    tex: '\\alpha_k = \\frac{\\mathbf{r}_k^{\\mathsf{T}}\\mathbf{r}_k}{\\mathbf{p}_k^{\\mathsf{T}} A \\mathbf{p}_k}, \\quad \\mathbf{x}_{k+1} = \\mathbf{x}_k + \\alpha_k \\mathbf{p}_k, \\quad \\mathbf{r}_{k+1} = \\mathbf{r}_k - \\alpha_k A\\mathbf{p}_k',
    fallback: 'alpha_k = (r_k^T r_k) / (p_k^T A p_k), x_{k+1} = x_k + alpha_k p_k',
  },
}

const INITIAL_TABLE_ROWS = 100

interface PlotPalette {
  bg: string
  plotBg: string
  border: string
  grid: string
  tolLine: string
  text: string
  textMuted: string
  legendBg: string
  legendBorder: string
}

const LIGHT_PALETTE: PlotPalette = {
  bg: '#ffffff',
  plotBg: '#ffffff',
  border: '#b8a4a8',
  grid: '#e6dce0',
  tolLine: '#cca757',
  text: '#131315',
  textMuted: '#5b5560',
  legendBg: 'rgba(255, 255, 255, 0.93)',
  legendBorder: '#d3adb4',
}

const DARK_PALETTE: PlotPalette = {
  bg: '#1d1d21',
  plotBg: '#161619',
  border: '#574044',
  grid: '#2d2629',
  tolLine: '#d6b365',
  text: '#f0ebe6',
  textMuted: '#b3a8ab',
  legendBg: 'rgba(29, 29, 33, 0.93)',
  legendBorder: '#574044',
}

function activePlotPalette(): PlotPalette {
  if (typeof document !== 'undefined' && document.documentElement.dataset.theme === 'dark') {
    return DARK_PALETTE
  }
  return LIGHT_PALETTE
}

function getMethodColors(): Record<'jacobi' | 'gauss-seidel' | 'sor' | 'cg', string> {
  const dark = typeof document !== 'undefined' && document.documentElement.dataset.theme === 'dark'
  return {
    jacobi: dark ? '#f08096' : '#80001e',
    'gauss-seidel': dark ? '#60a5fa' : '#1f77b4',
    sor: dark ? '#fb923c' : '#d97706',
    cg: dark ? '#4ade80' : '#15803d',
  }
}

export interface ConvergenceSeries {
  label: string
  color: string
  steps: IterStep[]
}

export interface ConvergenceChartOptions {
  tol: number
  title?: string
}

const SUPERSCRIPTS: Record<string, string> = {
  '-': '⁻',
  '0': '⁰',
  '1': '¹',
  '2': '²',
  '3': '³',
  '4': '⁴',
  '5': '⁵',
  '6': '⁶',
  '7': '⁷',
  '8': '⁸',
  '9': '⁹',
}

function formatPow10(exp: number): string {
  if (exp === 0) return '1'
  if (exp === 1) return '10'
  const sup = String(exp)
    .split('')
    .map((ch) => SUPERSCRIPTS[ch] ?? ch)
    .join('')
  return `10${sup}`
}

function niceIntStep(span: number, targetTicks = 7): number {
  if (span <= targetTicks) return 1
  const raw = span / targetTicks
  const pow10 = Math.pow(10, Math.floor(Math.log10(raw)))
  const frac = raw / pow10
  let mult = 1
  if (frac <= 1) mult = 1
  else if (frac <= 2) mult = 2
  else if (frac <= 5) mult = 5
  else mult = 10
  return Math.max(1, Math.round(mult * pow10))
}

/**
 * Dibuja la gráfica de convergencia (error ‖Ax_k − b‖₂ contra iteración k) en Canvas HiDPI
 * con eje Y en escala logarítmica, línea de tolerancia y leyenda para múltiples curvas.
 */
export function drawConvergenceChart(
  canvas: HTMLCanvasElement,
  seriesList: ConvergenceSeries[],
  opts: ConvergenceChartOptions,
): void {
  const ctx = canvas.getContext('2d')
  if (!ctx) return

  const width = Math.max(300, Math.round(canvas.clientWidth || 720))
  const height = Math.max(240, Math.round(canvas.clientHeight || 310))
  const dpr = Math.min(window.devicePixelRatio || 1, 3)

  if (
    canvas.width !== Math.round(width * dpr) ||
    canvas.height !== Math.round(height * dpr)
  ) {
    canvas.width = Math.round(width * dpr)
    canvas.height = Math.round(height * dpr)
  }

  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  const pal = activePlotPalette()

  const margin = { top: 38, right: 24, bottom: 48, left: 72 }
  const plotX = margin.left
  const plotY = margin.top
  const plotW = Math.max(40, width - margin.left - margin.right)
  const plotH = Math.max(40, height - margin.top - margin.bottom)

  ctx.fillStyle = pal.bg
  ctx.fillRect(0, 0, width, height)
  ctx.fillStyle = pal.plotBg
  ctx.fillRect(plotX, plotY, plotW, plotH)

  const nonEmpty = seriesList.filter((s) => s.steps.length > 0)
  if (nonEmpty.length === 0) {
    ctx.strokeStyle = pal.border
    ctx.lineWidth = 1.2
    ctx.strokeRect(plotX + 0.5, plotY + 0.5, plotW - 1, plotH - 1)
    ctx.fillStyle = pal.textMuted
    ctx.font = '600 12.5px Montserrat, system-ui, sans-serif'
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'
    ctx.fillText('Sin iteraciones para graficar', plotX + plotW / 2, plotY + plotH / 2)
    return
  }

  let minK = Infinity
  let maxK = -Infinity
  let minPosErr = Infinity
  let maxPosErr = -Infinity

  for (const s of nonEmpty) {
    for (const pt of s.steps) {
      if (pt.k < minK) minK = pt.k
      if (pt.k > maxK) maxK = pt.k
      if (Number.isFinite(pt.error) && pt.error > 0) {
        if (pt.error < minPosErr) minPosErr = pt.error
        if (pt.error > maxPosErr) maxPosErr = pt.error
      }
    }
  }

  const tol = Number.isFinite(opts.tol) && opts.tol > 0 ? opts.tol : 1e-6
  const zeroFloor = Number.isFinite(minPosErr)
    ? Math.max(1e-18, Math.min(minPosErr, tol) * 0.1)
    : Math.max(1e-18, tol * 0.1)

  if (!Number.isFinite(minPosErr)) minPosErr = zeroFloor
  if (!Number.isFinite(maxPosErr)) maxPosErr = Math.max(tol * 10, 1)

  const hasZero = nonEmpty.some((s) => s.steps.some((pt) => pt.error === 0))
  const logMinRaw = Math.log10(hasZero ? zeroFloor : Math.min(minPosErr, tol))
  const logMaxRaw = Math.log10(Math.max(maxPosErr, tol * 10))

  let minExp = Math.floor(logMinRaw)
  let maxExp = Math.ceil(logMaxRaw)
  if (maxExp <= minExp) {
    minExp -= 1
    maxExp += 1
  }

  const xMin = minK === maxK ? Math.max(0, minK - 1) : minK
  const xMax = minK === maxK ? maxK + 1 : maxK
  const xSpan = Math.max(1, xMax - xMin)
  const ySpan = Math.max(1, maxExp - minExp)

  const toPxX = (k: number): number => plotX + ((k - xMin) / xSpan) * plotW
  const toPxY = (err: number): number => {
    const safe = err > 0 ? err : zeroFloor
    const logv = Math.log10(safe)
    const norm = (logv - minExp) / ySpan
    return plotY + (1 - norm) * plotH
  }

  // Ticks X (enteros)
  const xStep = niceIntStep(xSpan, 7)
  const xTicks: number[] = []
  const firstTickX = xMin === 0 ? 0 : Math.ceil(xMin / xStep) * xStep
  if (xMin < firstTickX) xTicks.push(xMin)
  for (let kx = firstTickX; kx <= xMax; kx += xStep) {
    if (xTicks[xTicks.length - 1] !== kx) xTicks.push(kx)
  }
  if (xTicks[xTicks.length - 1] !== xMax && xMax - (xTicks[xTicks.length - 1] ?? 0) >= xStep * 0.4) {
    xTicks.push(xMax)
  }

  // Ticks Y (potencias de 10)
  const expStep = ySpan <= 8 ? 1 : ySpan <= 18 ? 2 : ySpan <= 36 ? 4 : Math.ceil(ySpan / 8)
  const yExps: number[] = []
  for (let e = minExp; e <= maxExp; e += expStep) {
    yExps.push(e)
  }
  if (yExps[yExps.length - 1] !== maxExp) yExps.push(maxExp)

  // Rejilla y curvas con clipping
  ctx.save()
  ctx.beginPath()
  ctx.rect(plotX, plotY, plotW, plotH)
  ctx.clip()

  ctx.strokeStyle = pal.grid
  ctx.lineWidth = 1
  ctx.beginPath()
  for (const kx of xTicks) {
    const px = Math.round(toPxX(kx)) + 0.5
    ctx.moveTo(px, plotY)
    ctx.lineTo(px, plotY + plotH)
  }
  for (const e of yExps) {
    const py = Math.round(plotY + (1 - (e - minExp) / ySpan) * plotH) + 0.5
    ctx.moveTo(plotX, py)
    ctx.lineTo(plotX + plotW, py)
  }
  ctx.stroke()

  // Línea horizontal de tolerancia
  const logTol = Math.log10(tol)
  if (logTol >= minExp && logTol <= maxExp) {
    const pyTol = Math.round(plotY + (1 - (logTol - minExp) / ySpan) * plotH) + 0.5
    ctx.save()
    ctx.strokeStyle = pal.tolLine
    ctx.lineWidth = 1.6
    ctx.setLineDash([6, 4])
    ctx.beginPath()
    ctx.moveTo(plotX, pyTol)
    ctx.lineTo(plotX + plotW, pyTol)
    ctx.stroke()
    ctx.restore()

    ctx.fillStyle = pal.tolLine
    ctx.font = '700 10.5px Montserrat, system-ui, sans-serif'
    ctx.textAlign = 'right'
    ctx.textBaseline = 'bottom'
    ctx.fillText(`tol = ${tol.toExponential(0)}`, plotX + plotW - 6, pyTol - 4)
  }

  // Curvas de error
  for (const s of nonEmpty) {
    ctx.strokeStyle = s.color
    ctx.fillStyle = s.color
    ctx.lineWidth = 2.2
    ctx.lineJoin = 'round'
    ctx.lineCap = 'round'

    if (s.steps.length > 1) {
      ctx.beginPath()
      let started = false
      for (const pt of s.steps) {
        if (!Number.isFinite(pt.error)) {
          started = false
          continue
        }
        const px = toPxX(pt.k)
        const py = toPxY(pt.error)
        if (!started) {
          ctx.moveTo(px, py)
          started = true
        } else {
          ctx.lineTo(px, py)
        }
      }
      ctx.stroke()
    }

    if (s.steps.length <= 45) {
      for (const pt of s.steps) {
        if (!Number.isFinite(pt.error)) continue
        const px = toPxX(pt.k)
        const py = toPxY(pt.error)
        ctx.beginPath()
        ctx.arc(px, py, s.steps.length === 1 ? 4.2 : 2.8, 0, Math.PI * 2)
        ctx.fill()
      }
    }
  }

  ctx.restore()

  // Marco del gráfico
  ctx.strokeStyle = pal.border
  ctx.lineWidth = 1.2
  ctx.strokeRect(plotX + 0.5, plotY + 0.5, plotW - 1, plotH - 1)

  // Marcas y números de ejes
  ctx.fillStyle = pal.textMuted
  ctx.strokeStyle = pal.border
  ctx.lineWidth = 1
  ctx.font = '11px Montserrat, system-ui, sans-serif'

  ctx.textAlign = 'center'
  ctx.textBaseline = 'top'
  for (const kx of xTicks) {
    const px = toPxX(kx)
    if (px < plotX - 1 || px > plotX + plotW + 1) continue
    ctx.beginPath()
    ctx.moveTo(Math.round(px) + 0.5, plotY + plotH)
    ctx.lineTo(Math.round(px) + 0.5, plotY + plotH + 5)
    ctx.stroke()
    ctx.fillText(String(kx), px, plotY + plotH + 8)
  }

  ctx.textAlign = 'right'
  ctx.textBaseline = 'middle'
  for (const e of yExps) {
    const py = plotY + (1 - (e - minExp) / ySpan) * plotH
    if (py < plotY - 1 || py > plotY + plotH + 1) continue
    ctx.beginPath()
    ctx.moveTo(plotX - 5, Math.round(py) + 0.5)
    ctx.lineTo(plotX, Math.round(py) + 0.5)
    ctx.stroke()
    ctx.fillText(formatPow10(e), plotX - 8, py)
  }

  // Leyenda (cuando hay 2 o más curvas, o para identificar la curva)
  if (nonEmpty.length > 0) {
    ctx.font = '600 11px Montserrat, system-ui, sans-serif'
    const padX = 10
    const padY = 7
    const rowH = 17
    const swatchW = 16
    let maxLabelW = 0
    for (const s of nonEmpty) {
      const w = ctx.measureText(s.label).width
      if (w > maxLabelW) maxLabelW = w
    }
    const boxW = Math.min(plotW - 16, padX * 2 + swatchW + 8 + maxLabelW)
    const boxH = padY * 2 + nonEmpty.length * rowH
    const boxX = plotX + plotW - boxW - 10
    const boxY = plotY + 10

    ctx.fillStyle = pal.legendBg
    ctx.strokeStyle = pal.legendBorder
    ctx.lineWidth = 1
    ctx.beginPath()
    if (typeof ctx.roundRect === 'function') {
      ctx.roundRect(boxX, boxY, boxW, boxH, 6)
    } else {
      ctx.rect(boxX, boxY, boxW, boxH)
    }
    ctx.fill()
    ctx.stroke()

    ctx.textAlign = 'left'
    ctx.textBaseline = 'middle'
    nonEmpty.forEach((s, idx) => {
      const cy = boxY + padY + idx * rowH + rowH / 2
      ctx.strokeStyle = s.color
      ctx.lineWidth = 2.6
      ctx.beginPath()
      ctx.moveTo(boxX + padX, cy)
      ctx.lineTo(boxX + padX + swatchW, cy)
      ctx.stroke()

      ctx.fillStyle = pal.text
      ctx.fillText(s.label, boxX + padX + swatchW + 8, cy, boxW - padX * 2 - swatchW - 8)
    })
  }

  // Título y etiquetas de ejes
  ctx.fillStyle = pal.text
  const title = opts.title ?? 'Convergencia (escala logarítmica en el error)'
  if (title.trim()) {
    ctx.font = '700 13px Montserrat, system-ui, sans-serif'
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'
    ctx.fillText(title, plotX + plotW / 2, margin.top / 2, width - 24)
  }

  ctx.font = '600 12px Montserrat, system-ui, sans-serif'
  ctx.textAlign = 'center'
  ctx.textBaseline = 'bottom'
  ctx.fillText('Iteración k', plotX + plotW / 2, height - 6)

  ctx.save()
  ctx.translate(16, plotY + plotH / 2)
  ctx.rotate(-Math.PI / 2)
  ctx.textAlign = 'center'
  ctx.textBaseline = 'top'
  ctx.fillText('Error ‖Ax_k − b‖₂ (log)', 0, 0)
  ctx.restore()
}

function formatFixed6(v: number): string {
  if (!Number.isFinite(v)) return String(v)
  const clean = Math.abs(v) < 5e-13 ? 0 : v
  if (Math.abs(clean) >= 1e9) return clean.toExponential(4)
  return clean.toFixed(6)
}

function formatErr(e: number): string {
  if (!Number.isFinite(e)) return '∞'
  if (e === 0) return '0.00e+0'
  return e.toExponential(2)
}

function formatVectorFixed6(v: number[]): string {
  return `(${v.map(formatFixed6).join(', ')})`
}

function numVectorToLatex6(v: number[]): string {
  return `\\begin{bmatrix} ${v.map(formatFixed6).join(' \\\\ ')} \\end{bmatrix}`
}

function ratVectorToLatex(v: Rat[]): string {
  return `\\begin{bmatrix} ${v.map(ratToLatex).join(' \\\\ ')} \\end{bmatrix}`
}

function maxAbsDiff(a: number[], b: number[]): number {
  let m = 0
  for (let i = 0; i < Math.min(a.length, b.length); i++) {
    const d = Math.abs(a[i] - b[i])
    if (d > m) m = d
  }
  return m
}

function createMathBox(tex: string, fallbackText: string, compact = false): HTMLDivElement {
  const box = document.createElement('div')
  box.className = compact ? 'la-result-math la-result-math--compact' : 'la-result-math'
  void renderLatex(tex, box, { displayMode: true, fallbackText })
  return box
}

function buildIterationTableElement(steps: IterStep[], caption?: string): HTMLElement {
  const section = document.createElement('div')
  section.className = 'la-iter-table-section'

  if (caption) {
    const heading = document.createElement('h4')
    heading.className = 'la-subheading'
    heading.textContent = caption
    section.appendChild(heading)
  }

  if (steps.length === 0) {
    const note = document.createElement('p')
    note.className = 'la-note'
    note.textContent = 'No se registraron iteraciones.'
    section.appendChild(note)
    return section
  }

  const wrap = document.createElement('div')
  wrap.className = 'la-table-wrap la-iter-table-wrap'

  const table = document.createElement('table')
  table.className = 'la-table'
  table.innerHTML = `
    <thead>
      <tr>
        <th scope="col">k</th>
        <th scope="col">x_k (6 decimales)</th>
        <th scope="col">Error ‖Ax_k − b‖₂</th>
      </tr>
    </thead>
    <tbody></tbody>
  `
  const tbody = table.querySelector('tbody')!

  const appendRows = (from: number, to: number): void => {
    const end = Math.min(to, steps.length)
    for (let i = from; i < end; i++) {
      const st = steps[i]
      const tr = document.createElement('tr')

      const tdK = document.createElement('td')
      tdK.className = 'la-table-mono'
      tdK.textContent = String(st.k)

      const tdX = document.createElement('td')
      tdX.className = 'la-table-mono'
      tdX.textContent = formatVectorFixed6(st.x)

      const tdErr = document.createElement('td')
      tdErr.className = 'la-table-mono'
      tdErr.textContent = formatErr(st.error)

      tr.appendChild(tdK)
      tr.appendChild(tdX)
      tr.appendChild(tdErr)
      tbody.appendChild(tr)
    }
  }

  const initialCount = Math.min(steps.length, INITIAL_TABLE_ROWS)
  appendRows(0, initialCount)
  wrap.appendChild(table)
  section.appendChild(wrap)

  if (steps.length > initialCount) {
    const remaining = steps.length - initialCount
    const moreBar = document.createElement('div')
    moreBar.className = 'la-steps-more'

    const btnMore = document.createElement('button')
    btnMore.type = 'button'
    btnMore.className = 'la-btn la-btn--sm'
    btnMore.textContent =
      remaining === 1
        ? 'Mostrar la iteración restante'
        : `Mostrar las ${remaining} iteraciones restantes`

    const info = document.createElement('span')
    info.className = 'la-note'
    info.textContent = `Mostrando ${initialCount} de ${steps.length} iteraciones.`

    btnMore.addEventListener('click', () => {
      appendRows(initialCount, steps.length)
      moreBar.remove()
    })

    moreBar.appendChild(btnMore)
    moreBar.appendChild(info)
    section.appendChild(moreBar)
  }

  return section
}

export function mountIterativeUI(app: HTMLElement, track: TrackFn): void {
  const defaultPreset = ITERATIVE_PRESETS[0]
  let vectorBValues: string[] = defaultPreset.b.slice()
  let vectorX0Values: string[] = Array.from({ length: defaultPreset.a.length }, () => '0')
  let activePresetId: string | null = defaultPreset.id
  let vectorBInputs: HTMLInputElement[] = []
  let vectorX0Inputs: HTMLInputElement[] = []

  const defaultANum = defaultPreset.a.map((row) => row.map((v) => Number(v)))
  const initialOptOmega = omegaOptimo(defaultANum)
  let currentOmega =
    initialOptOmega !== null
      ? Math.max(0.01, Math.min(1.99, Math.round(initialOptOmega * 100) / 100))
      : 1.05
  let lastOptOmega: number | null = initialOptOmega
  let hasActiveResult = false
  let compareDetailKey: 'jacobi' | 'gauss-seidel' | 'sor' | 'cg' = 'sor'

  let lastChartState: {
    canvas: HTMLCanvasElement
    series: ConvergenceSeries[]
    tol: number
    title: string
  } | null = null

  app.innerHTML = `
    <div class="la-workspace">
      <section class="la-panel">
        <div class="la-panel-head">
          <h2 class="la-panel-title">Sistema lineal Ax = b (solucionadores iterativos)</h2>
          <span class="la-panel-meta" id="iter-dim-meta"></span>
        </div>

        <div class="la-toolbar" id="iter-presets-bar">
          <span class="la-toolbar-label">Presets:</span>
        </div>
        <p class="la-preset-desc" id="iter-preset-desc"></p>

        <div class="la-panel-body la-direct-input-layout">
          <div class="la-direct-system-editors">
            <div class="la-direct-matrix-block">
              <span class="la-field-label">Matriz de coeficientes A (cuadrada, hasta 6×6)</span>
              <div id="iter-grid-a-mount"></div>
            </div>

            <div class="la-direct-vector-block">
              <div class="la-vector-head">
                <span class="la-field-label">Vector b</span>
                <button class="la-btn la-btn--sm" id="btn-iter-clear-b" type="button" aria-label="Limpiar vector b">Limpiar b</button>
              </div>
              <div class="la-matrix-scroll">
                <div class="la-matrix-bracket">
                  <div class="la-vector-grid" id="iter-vector-b-grid"></div>
                </div>
              </div>
            </div>

            <div class="la-direct-vector-block">
              <div class="la-vector-head">
                <span class="la-field-label">Inicial x₀</span>
                <button class="la-btn la-btn--sm" id="btn-iter-zero-x0" type="button" aria-label="Poner vector inicial x0 en ceros">Ceros</button>
              </div>
              <div class="la-matrix-scroll">
                <div class="la-matrix-bracket">
                  <div class="la-vector-grid" id="iter-vector-x0-grid"></div>
                </div>
              </div>
            </div>
          </div>

          <div class="la-paste-box">
            <label class="la-field-label" for="iter-paste-input">Pegar sistema [A | b]</label>
            <p class="la-hint" id="iter-paste-hint">Pega la matriz aumentada (última columna = vector b; separador «|» opcional, hasta 6×6).</p>
            <textarea
              id="iter-paste-input"
              class="la-textarea"
              rows="4"
              spellcheck="false"
              aria-describedby="iter-paste-hint"
              placeholder="4 1 1 | 6&#10;1 5 1 | 7&#10;1 1 6 | 8"
            ></textarea>
            <div class="la-paste-actions">
              <button class="la-btn la-btn--sm" id="btn-iter-paste" type="button">Pegar sistema</button>
              <span class="la-feedback" id="iter-paste-feedback" role="status" aria-live="polite" hidden></span>
            </div>
            <div class="la-error-banner" id="iter-paste-error" role="alert" hidden></div>
          </div>
        </div>

        <div class="la-iter-diag-bar" id="iter-convergence-diag" aria-live="polite"></div>
        <div class="la-error-banner la-panel-error" id="iter-input-error" role="alert" hidden></div>
      </section>

      <section class="la-panel">
        <div class="la-panel-head">
          <h2 class="la-panel-title">Método iterativo y convergencia</h2>
          <span class="la-panel-meta">Numérico (punto flotante) · Error = residuo ‖Ax_k − b‖₂</span>
        </div>

        <div class="la-toolbar la-iter-controls-bar">
          <label class="la-direct-method-label" for="iter-method-select">
            <span>Método:</span>
            <select class="la-select la-direct-method-select" id="iter-method-select">
              <option value="jacobi" selected>Jacobi</option>
              <option value="gauss-seidel">Gauss-Seidel</option>
              <option value="sor">SOR (sobrerrelajación sucesiva)</option>
              <option value="cg">Gradiente conjugado (SPD)</option>
              <option value="compare">Comparar todos</option>
            </select>
          </label>

          <label class="la-iter-param-field" for="iter-tol-input">
            <span>Tolerancia (tol)</span>
            <input
              id="iter-tol-input"
              class="la-input la-input--compact"
              type="text"
              inputmode="decimal"
              spellcheck="false"
              value="1e-6"
            />
          </label>

          <label class="la-iter-param-field" for="iter-maxiter-input">
            <span>Máx. iteraciones</span>
            <input
              id="iter-maxiter-input"
              class="la-input la-input--compact"
              type="number"
              min="1"
              max="5000"
              step="1"
              value="100"
            />
          </label>

          <button class="la-btn la-btn--primary" id="btn-iter-solve" type="button">Resolver</button>
        </div>

        <div class="la-iter-omega-box" id="iter-omega-box" hidden>
          <div class="la-param-head">
            <label class="la-param-name" for="iter-omega-slider">
              Factor de relajación ω (SOR) =
              <span class="la-param-value" id="iter-omega-val">${currentOmega.toFixed(2)}</span>
            </label>
            <div class="la-iter-omega-meta">
              <span class="la-iter-omega-opt" id="iter-omega-opt-label"></span>
              <button class="la-btn la-btn--sm" id="btn-iter-use-opt" type="button">Usar ω óptimo</button>
            </div>
          </div>
          <input
            type="range"
            id="iter-omega-slider"
            class="la-slider"
            min="0.01"
            max="1.99"
            step="0.01"
            value="${currentOmega.toFixed(2)}"
          />
          <div class="la-param-bounds">
            <span>0.01 (subrelajación)</span>
            <span>1.00 (Gauss-Seidel)</span>
            <span>1.99 (sobrerrelajación)</span>
          </div>
        </div>

        <div class="la-iter-formula-box" id="iter-formula-box">
          <div class="la-iter-formula-title" id="iter-formula-title"></div>
          <div class="la-iter-formula-math" id="iter-formula-math"></div>
        </div>

        <div class="la-panel-body la-results-stack">
          <p class="la-note" id="iter-empty-hint">
            Selecciona un método iterativo o «Comparar todos» y pulsa <strong>Resolver</strong> para ver la gráfica de convergencia en escala logarítmica, la tabla de iteraciones y la comparación con la solución exacta.
          </p>
          <article class="la-result-card" id="iter-result-card" hidden></article>
        </div>
      </section>
    </div>
  `

  const presetsBar = app.querySelector<HTMLDivElement>('#iter-presets-bar')!
  const presetDescEl = app.querySelector<HTMLParagraphElement>('#iter-preset-desc')!
  const dimMetaEl = app.querySelector<HTMLSpanElement>('#iter-dim-meta')!
  const gridAMount = app.querySelector<HTMLDivElement>('#iter-grid-a-mount')!
  const vectorBGrid = app.querySelector<HTMLDivElement>('#iter-vector-b-grid')!
  const btnClearB = app.querySelector<HTMLButtonElement>('#btn-iter-clear-b')!
  const vectorX0Grid = app.querySelector<HTMLDivElement>('#iter-vector-x0-grid')!
  const btnZeroX0 = app.querySelector<HTMLButtonElement>('#btn-iter-zero-x0')!
  const pasteInput = app.querySelector<HTMLTextAreaElement>('#iter-paste-input')!
  const btnPaste = app.querySelector<HTMLButtonElement>('#btn-iter-paste')!
  const pasteFeedbackEl = app.querySelector<HTMLSpanElement>('#iter-paste-feedback')!
  const pasteErrorEl = app.querySelector<HTMLDivElement>('#iter-paste-error')!
  const diagBarEl = app.querySelector<HTMLDivElement>('#iter-convergence-diag')!
  const inputErrorEl = app.querySelector<HTMLDivElement>('#iter-input-error')!

  const methodSelect = app.querySelector<HTMLSelectElement>('#iter-method-select')!
  const tolInput = app.querySelector<HTMLInputElement>('#iter-tol-input')!
  const maxIterInput = app.querySelector<HTMLInputElement>('#iter-maxiter-input')!
  const btnSolve = app.querySelector<HTMLButtonElement>('#btn-iter-solve')!

  const omegaBox = app.querySelector<HTMLDivElement>('#iter-omega-box')!
  const omegaSlider = app.querySelector<HTMLInputElement>('#iter-omega-slider')!
  const omegaValEl = app.querySelector<HTMLSpanElement>('#iter-omega-val')!
  const omegaOptLabel = app.querySelector<HTMLSpanElement>('#iter-omega-opt-label')!
  const btnUseOpt = app.querySelector<HTMLButtonElement>('#btn-iter-use-opt')!

  const formulaBox = app.querySelector<HTMLDivElement>('#iter-formula-box')!
  const formulaTitleEl = app.querySelector<HTMLDivElement>('#iter-formula-title')!
  const formulaMathEl = app.querySelector<HTMLDivElement>('#iter-formula-math')!

  const emptyHintEl = app.querySelector<HTMLParagraphElement>('#iter-empty-hint')!
  const resultCardEl = app.querySelector<HTMLElement>('#iter-result-card')!

  const supportsAccent =
    typeof CSS !== 'undefined' && typeof CSS.supports === 'function'
      ? CSS.supports('accent-color', 'auto')
      : true

  function updateSliderProgress(val: number): void {
    if (!supportsAccent) {
      const pct = ((val - 0.01) / (1.99 - 0.01)) * 100
      omegaSlider.style.setProperty('--progress', `${Math.max(0, Math.min(100, pct))}%`)
    }
  }
  updateSliderProgress(currentOmega)

  function showBannerError(el: HTMLElement, msg: string | null): void {
    if (!msg) {
      el.hidden = true
      el.textContent = ''
      return
    }
    el.textContent = msg
    el.hidden = false
  }

  function showTimedFeedback(el: HTMLElement, msg: string): void {
    el.textContent = msg
    el.hidden = false
    const prev = Number(el.dataset.timerId ?? '0')
    if (prev) window.clearTimeout(prev)
    const id = window.setTimeout(() => {
      el.hidden = true
    }, 3000)
    el.dataset.timerId = String(id)
  }

  function markResultStale(): void {
    if (resultCardEl.hidden || resultCardEl.classList.contains('is-stale')) return
    resultCardEl.classList.add('is-stale')
    const head = resultCardEl.querySelector('.la-result-head')
    const notice = document.createElement('div')
    notice.className = 'la-stale-banner'
    notice.textContent = 'Resultado de una versión anterior del sistema'
    if (head && head.nextSibling) {
      resultCardEl.insertBefore(notice, head.nextSibling)
    } else {
      resultCardEl.appendChild(notice)
    }
  }

  function renderVectorEditor(
    container: HTMLElement,
    values: string[],
    prefix: string,
    ariaKind: string,
    rowCount: number,
    onMutate: () => void,
  ): { values: string[]; inputs: HTMLInputElement[] } {
    const nextVals = Array.from({ length: rowCount }, (_, i) => values[i] ?? '0')
    container.innerHTML = ''
    const nextInputs: HTMLInputElement[] = []

    for (let i = 0; i < rowCount; i++) {
      const row = document.createElement('div')
      row.className = 'la-vector-row'

      const tag = document.createElement('span')
      tag.className = 'la-vector-idx'
      tag.textContent = `${prefix}${i + 1}`

      const input = document.createElement('input')
      input.type = 'text'
      input.className = 'la-cell-input'
      input.value = nextVals[i]
      input.spellcheck = false
      input.autocomplete = 'off'
      input.setAttribute('aria-label', `${ariaKind}, fila ${i + 1}`)
      input.setAttribute('aria-invalid', 'false')

      input.addEventListener('input', () => {
        nextVals[i] = input.value
        if (input.classList.contains('is-invalid') && parseRat(input.value) !== null) {
          input.classList.remove('is-invalid')
          input.setAttribute('aria-invalid', 'false')
        }
        onMutate()
      })

      row.appendChild(tag)
      row.appendChild(input)
      container.appendChild(row)
      nextInputs.push(input)
    }

    return { values: nextVals, inputs: nextInputs }
  }

  function syncVectors(rowCount: number): void {
    const resB = renderVectorEditor(
      vectorBGrid,
      vectorBValues,
      'b',
      'Vector b (iterativos)',
      rowCount,
      () => {
        activePresetId = null
        showBannerError(inputErrorEl, null)
        markResultStale()
        updatePresetButtons()
      },
    )
    vectorBValues = resB.values
    vectorBInputs = resB.inputs

    const resX0 = renderVectorEditor(
      vectorX0Grid,
      vectorX0Values,
      'x₀,',
      'Vector inicial x0',
      rowCount,
      () => {
        showBannerError(inputErrorEl, null)
        markResultStale()
      },
    )
    vectorX0Values = resX0.values
    vectorX0Inputs = resX0.inputs
  }

  function readVector(
    inputs: HTMLInputElement[],
    backing: string[],
    label: string,
  ):
    | { ok: true; rat: Rat[]; num: number[] }
    | { ok: false; error: string; firstInvalid: HTMLInputElement } {
    const rat: Rat[] = []
    const num: number[] = []
    let firstInvalid: HTMLInputElement | null = null
    const badRows: number[] = []

    for (let i = 0; i < inputs.length; i++) {
      const input = inputs[i]
      backing[i] = input.value
      const val = parseRat(input.value)
      if (val === null) {
        input.classList.add('is-invalid')
        input.setAttribute('aria-invalid', 'true')
        if (!firstInvalid) firstInvalid = input
        badRows.push(i + 1)
      } else {
        input.classList.remove('is-invalid')
        input.setAttribute('aria-invalid', 'false')
        rat.push(val)
        num.push(val.toNumber())
      }
    }

    if (firstInvalid) {
      return {
        ok: false,
        firstInvalid,
        error: `Revisa el ${label} en ${badRows.length === 1 ? `la fila ${badRows[0]}` : `las filas ${badRows.join(', ')}`}: escribe enteros, decimales o fracciones como 3/4.`,
      }
    }

    return { ok: true, rat, num }
  }

  function updateDimMeta(): void {
    const [r, c] = gridA.getSize()
    dimMetaEl.textContent =
      r === c
        ? `Sistema cuadrado ${r}×${c} (Ax = b)`
        : `A es ${r}×${c} (no cuadrada: los métodos iterativos requieren n×n)`
  }

  function updateOmegaOptDisplay(opt: number | null): void {
    lastOptOmega = opt
    if (opt !== null) {
      omegaOptLabel.textContent = `ω óptimo estimado: ${opt.toFixed(2)}`
      btnUseOpt.disabled = false
    } else {
      omegaOptLabel.textContent = 'ω óptimo estimado: no estimable'
      btnUseOpt.disabled = true
    }
  }

  function updateConvergenceDiagnostics(): void {
    const matRat = gridA.tryReadMatrix()
    diagBarEl.innerHTML = ''

    if (!matRat) {
      const span = document.createElement('span')
      span.className = 'la-note'
      span.textContent =
        'Completa las celdas numéricas de A para evaluar la condición de convergencia.'
      diagBarEl.appendChild(span)
      updateOmegaOptDisplay(null)
      return
    }

    if (!isSquare(matRat)) {
      const badge = document.createElement('span')
      badge.className = 'la-sys-badge la-sys-badge--ninguna'
      badge.textContent = `Matriz no cuadrada (${matRat.length}×${matRat[0].length}): los métodos iterativos requieren una matriz cuadrada n×n`
      diagBarEl.appendChild(badge)
      updateOmegaOptDisplay(null)
      return
    }

    const aNum = matRat.map((row) => row.map((v) => v.toNumber()))
    const hasZeroDiag = aNum.some((row, i) => row[i] === 0)
    if (hasZeroDiag) {
      const badge = document.createElement('span')
      badge.className = 'la-sys-badge la-sys-badge--ninguna'
      badge.textContent =
        'Diagonal con ceros: Jacobi, Gauss-Seidel y SOR requieren aᵢᵢ ≠ 0 (reordena filas o usa un método directo)'
      diagBarEl.appendChild(badge)
      updateOmegaOptDisplay(null)
      return
    }

    const dom = esDiagonalDominante(aNum)
    const rho = estimaRadioEspectralJacobi(aNum)
    const opt = omegaOptimo(aNum)
    updateOmegaOptDisplay(opt)

    const badge = document.createElement('span')
    const detail = document.createElement('span')
    detail.className = 'la-iter-diag-detail'

    if (dom) {
      badge.className = 'la-sys-badge la-sys-badge--unica'
      badge.textContent =
        'Diagonal dominante: convergencia garantizada para Jacobi y Gauss-Seidel'
      detail.textContent = `ρ estimado (Jacobi) ≈ ${rho.toFixed(4)} · ω óptimo estimado: ${opt !== null ? opt.toFixed(2) : 'no estimable'}`
    } else if (Number.isFinite(rho) && opt !== null) {
      badge.className = 'la-sys-badge la-sys-badge--approx'
      badge.textContent = `No es diagonal dominante · ρ estimado (Jacobi) ≈ ${rho.toFixed(4)} < 1 (dentro de tolerancia)`
      detail.textContent = `Jacobi converge (aunque puede ser lento si ρ ≈ 1) · ω óptimo estimado: ${opt.toFixed(2)}`
    } else {
      badge.className = 'la-sys-badge la-sys-badge--ninguna'
      const rhoText = Number.isFinite(rho) ? rho.toFixed(4) : '∞'
      badge.textContent = `No es diagonal dominante · ρ estimado (Jacobi) ≈ ${rhoText} ≥ 1: Jacobi no convergerá`
      detail.textContent =
        'Prueba Gauss-Seidel, SOR, Gradiente conjugado (si es SPD) o un solucionador directo.'
    }

    diagBarEl.appendChild(badge)
    diagBarEl.appendChild(detail)
  }

  function updateMethodUI(): void {
    const method = methodSelect.value as IterativeSelectorValue
    const showOmega = method === 'sor' || method === 'compare'
    omegaBox.hidden = !showOmega

    if (method === 'compare') {
      formulaBox.hidden = true
      return
    }

    formulaBox.hidden = false
    const info = METHOD_FORMULAS[method]
    formulaTitleEl.textContent = info.titulo
    void renderLatex(info.tex, formulaMathEl, {
      displayMode: true,
      fallbackText: info.fallback,
    })
  }

  const presetButtons = new Map<string, HTMLButtonElement>()
  function updatePresetButtons(): void {
    for (const [id, btn] of presetButtons.entries()) {
      const active = id === activePresetId
      btn.classList.toggle('la-btn--active', active)
      btn.setAttribute('aria-pressed', active ? 'true' : 'false')
    }
    const preset = ITERATIVE_PRESETS.find((p) => p.id === activePresetId)
    presetDescEl.textContent = preset
      ? preset.descripcion
      : 'Sistema personalizado (admite enteros, decimales y fracciones como 3/4).'
  }

  const gridA = createMatrixGrid({
    idPrefix: 'iter-grid-a',
    matrixName: 'A',
    initialValues: defaultPreset.a,
    onSizeChange: (rows) => {
      syncVectors(rows)
      updateDimMeta()
      updateConvergenceDiagnostics()
    },
    onInputChange: () => {
      activePresetId = null
      showBannerError(inputErrorEl, null)
      markResultStale()
      updatePresetButtons()
      updateConvergenceDiagnostics()
    },
  })

  gridAMount.appendChild(gridA.element)
  syncVectors(defaultPreset.a.length)
  updateDimMeta()
  updateConvergenceDiagnostics()
  updateMethodUI()

  function applyPreset(preset: IterativePreset): void {
    activePresetId = preset.id
    vectorBValues = preset.b.slice()
    vectorX0Values = preset.x0
      ? preset.x0.slice()
      : Array.from({ length: preset.a.length }, () => '0')
    gridA.setValues(preset.a)
    syncVectors(preset.a.length)
    showBannerError(inputErrorEl, null)
    showBannerError(pasteErrorEl, null)
    markResultStale()
    updatePresetButtons()
    updateDimMeta()
    updateConvergenceDiagnostics()
    if (lastOptOmega !== null) {
      currentOmega = Math.max(0.01, Math.min(1.99, Math.round(lastOptOmega * 100) / 100))
      omegaSlider.value = currentOmega.toFixed(2)
      omegaValEl.textContent = currentOmega.toFixed(2)
      updateSliderProgress(currentOmega)
    }
  }

  for (const preset of ITERATIVE_PRESETS) {
    const btn = document.createElement('button')
    btn.type = 'button'
    btn.className = 'la-btn la-btn--sm'
    btn.textContent = preset.nombre
    btn.addEventListener('click', () => {
      applyPreset(preset)
      track('linealalrl-iter-preset', { preset: preset.id })
    })
    presetsBar.appendChild(btn)
    presetButtons.set(preset.id, btn)
  }
  updatePresetButtons()

  btnClearB.addEventListener('click', () => {
    vectorBValues = vectorBValues.map(() => '0')
    syncVectors(vectorBValues.length)
    activePresetId = null
    markResultStale()
    updatePresetButtons()
  })

  btnZeroX0.addEventListener('click', () => {
    vectorX0Values = vectorX0Values.map(() => '0')
    syncVectors(vectorX0Values.length)
    markResultStale()
  })

  btnPaste.addEventListener('click', () => {
    const parsed = parsePastedAugmented(pasteInput.value, MAX_GRID_DIM, MAX_GRID_DIM)
    if (!parsed.ok) {
      showBannerError(pasteErrorEl, parsed.error)
      pasteInput.focus()
      return
    }
    showBannerError(pasteErrorEl, null)
    showBannerError(inputErrorEl, null)
    activePresetId = null
    vectorBValues = parsed.vector.map(ratToString)
    vectorX0Values = Array.from({ length: parsed.filas }, () => '0')
    gridA.setMatrix(parsed.a)
    syncVectors(parsed.filas)
    markResultStale()
    updatePresetButtons()
    updateDimMeta()
    updateConvergenceDiagnostics()
    showTimedFeedback(
      pasteFeedbackEl,
      `Sistema cargado (${parsed.filas}×${parsed.cols} + vector b)`,
    )
    track('linealalrl-iter-paste', { size: `${parsed.filas}x${parsed.cols}` })
  })

  function readParams():
    | { ok: true; tol: number; maxIter: number }
    | { ok: false; error: string; focusEl: HTMLInputElement } {
    const rawTol = tolInput.value.trim().replace(/,/g, '.')
    const tol = Number(rawTol)
    if (!rawTol || !Number.isFinite(tol) || tol <= 0) {
      return {
        ok: false,
        error: 'La tolerancia (tol) debe ser un número positivo, por ejemplo 1e-6 o 0.0001.',
        focusEl: tolInput,
      }
    }

    const maxIter = Number(maxIterInput.value)
    if (!Number.isInteger(maxIter) || maxIter < 1 || maxIter > 5000) {
      return {
        ok: false,
        error: 'El máximo de iteraciones debe ser un entero entre 1 y 5000.',
        focusEl: maxIterInput,
      }
    }

    return { ok: true, tol, maxIter }
  }

  function renderExactReferenceBlock(
    body: HTMLElement,
    aRat: Matrix,
    bRat: Rat[],
    approxX?: number[],
  ): void {
    const refBox = document.createElement('div')
    refBox.className = 'la-iter-ref-box'

    try {
      const exact = solveRref(aRat, bRat)
      if (exact.tipo === 'unica' && exact.solucion) {
        const solRat = exact.solucion
        const solNum = solRat.map((v) => v.toNumber())
        const title = document.createElement('div')
        title.className = 'la-iter-ref-title'
        title.textContent = 'Referencia exacta (calculada con solveRref del núcleo):'
        refBox.appendChild(title)

        const varsTex = solRat.map((_, i) => `x_${i + 1}`).join(' \\\\ ')
        refBox.appendChild(
          createMathBox(
            `\\mathbf{x}_{\\text{exacta}} = \\begin{bmatrix} ${varsTex} \\end{bmatrix} = ${ratVectorToLatex(solRat)}`,
            `x_exacta = (${solRat.map(ratToString).join(', ')})`,
            true,
          ),
        )

        const metaParts: string[] = []
        if (solRat.some((v) => v.d !== 1n)) {
          metaParts.push(
            `Decimal: (${solRat.map((v) => ratToDecimal(v, 6)).join(', ')})`,
          )
        }
        if (approxX && approxX.length === solNum.length && approxX.every(Number.isFinite)) {
          const dist = maxAbsDiff(approxX, solNum)
          metaParts.push(`‖x_iter − x_exacta‖∞ = ${formatErr(dist)}`)
        }
        if (metaParts.length > 0) {
          const p = document.createElement('p')
          p.className = 'la-note'
          p.textContent = metaParts.join(' · ')
          refBox.appendChild(p)
        }
      } else if (exact.tipo === 'infinitas') {
        const p = document.createElement('p')
        p.className = 'la-note'
        p.textContent =
          'Referencia exacta (solveRref): la matriz A es singular y el sistema tiene infinitas soluciones.'
        refBox.appendChild(p)
      } else {
        const p = document.createElement('p')
        p.className = 'la-note'
        p.textContent =
          'Referencia exacta (solveRref): el sistema es inconsistente (no tiene solución).'
        refBox.appendChild(p)
      }
      body.appendChild(refBox)
    } catch {
      // Si solveRref no aplica, omitir referencia silenciosamente.
    }
  }

  function createChartBlock(
    series: ConvergenceSeries[],
    tol: number,
    title: string,
  ): HTMLElement {
    const wrap = document.createElement('div')
    wrap.className = 'la-canvas-wrap'

    const canvas = document.createElement('canvas')
    canvas.className = 'la-canvas'
    canvas.setAttribute('aria-label', title)
    wrap.appendChild(canvas)

    const hint = document.createElement('p')
    hint.className = 'la-canvas-hint'
    hint.textContent =
      'Eje Y en escala logarítmica: error e_k = ‖Ax_k − b‖₂ por iteración. La línea dorada punteada marca la tolerancia objetivo.'
    wrap.appendChild(hint)

    lastChartState = { canvas, series, tol, title }
    if (typeof window !== 'undefined' && typeof window.requestAnimationFrame === 'function') {
      window.requestAnimationFrame(() => {
        if (lastChartState && lastChartState.canvas === canvas) {
          drawConvergenceChart(canvas, series, { tol, title })
        }
      })
    } else {
      drawConvergenceChart(canvas, series, { tol, title })
    }

    return wrap
  }

  function redrawActiveChart(): void {
    if (!lastChartState || !lastChartState.canvas.isConnected) return
    drawConvergenceChart(lastChartState.canvas, lastChartState.series, {
      tol: lastChartState.tol,
      title: lastChartState.title,
    })
  }

  function renderSingleMethodResult(
    methodKey: 'jacobi' | 'gauss-seidel' | 'sor' | 'cg',
    methodTitle: string,
    res: IterResult,
    aRat: Matrix,
    bRat: Rat[],
    tol: number,
    maxIter: number,
  ): void {
    const n = aRat.length
    const count = res.iteraciones.length
    const meta = res.convergio
      ? `${n}×${n} · Convergió en ${count} ${count === 1 ? 'iteración' : 'iteraciones'}`
      : `${n}×${n} · No convergió (${count}/${maxIter} iteraciones)`

    const body = createResultFrame(resultCardEl, methodTitle, meta)

    const badge = document.createElement('div')
    badge.className = res.convergio
      ? 'la-sys-badge la-sys-badge--unica'
      : 'la-sys-badge la-sys-badge--ninguna'
    badge.textContent = res.convergio
      ? `Convergió en ${count} ${count === 1 ? 'iteración' : 'iteraciones'} (error < ${tol.toExponential(1)})`
      : `No convergió tras ${maxIter} iteraciones`
    body.appendChild(badge)

    const criterionNote = document.createElement('p')
    criterionNote.className = 'la-note'
    criterionNote.textContent =
      'Criterio de error: residuo euclidiano e_k = ‖Ax_k − b‖₂ (se detiene cuando e_k < tol).'
    body.appendChild(criterionNote)

    if (!res.convergio && res.motivo) {
      const warn = document.createElement('div')
      warn.className = 'la-error-banner'
      warn.setAttribute('role', 'alert')
      warn.textContent = res.motivo
      body.appendChild(warn)

      const actions = document.createElement('div')
      actions.className = 'la-inline-actions'

      const cmpBtn = document.createElement('button')
      cmpBtn.type = 'button'
      cmpBtn.className = 'la-btn la-btn--sm la-btn--primary'
      cmpBtn.textContent = 'Comparar todos los métodos'
      cmpBtn.addEventListener('click', () => {
        methodSelect.value = 'compare'
        updateMethodUI()
        runSolve(true)
      })
      actions.appendChild(cmpBtn)

      if (tol < 1e-3) {
        const relaxBtn = document.createElement('button')
        relaxBtn.type = 'button'
        relaxBtn.className = 'la-btn la-btn--sm'
        relaxBtn.textContent = 'Relajar tolerancia (1e-3)'
        relaxBtn.addEventListener('click', () => {
          tolInput.value = '1e-3'
          runSolve(true)
        })
        actions.appendChild(relaxBtn)
      }
      body.appendChild(actions)
    }

    if (res.iteraciones.length > 0) {
      const lastStep = res.iteraciones[res.iteraciones.length - 1]
      const varsTex = res.x.map((_, i) => `x_${i + 1}`).join(' \\\\ ')
      body.appendChild(
        createMathBox(
          `\\mathbf{x}_{\\text{final}} \\approx \\begin{bmatrix} ${varsTex} \\end{bmatrix} = ${numVectorToLatex6(res.x)}`,
          `x_final = ${formatVectorFixed6(res.x)}`,
        ),
      )

      const strip = document.createElement('div')
      strip.className = 'la-residual-strip'
      strip.innerHTML = `
        <span><strong>Iteraciones:</strong> <code>${count}</code></span>
        <span><strong>Error final ‖Ax − b‖₂:</strong> <code>${formatErr(lastStep.error)}</code></span>
        <span><strong>Tolerancia:</strong> <code>${tol.toExponential(1)}</code></span>
      `
      body.appendChild(strip)
    }

    renderExactReferenceBlock(
      body,
      aRat,
      bRat,
      res.iteraciones.length > 0 ? res.x : undefined,
    )

    const colors = getMethodColors()
    if (res.iteraciones.length > 0) {
      body.appendChild(
        createChartBlock(
          [{ label: methodTitle, color: colors[methodKey], steps: res.iteraciones }],
          tol,
          `Convergencia — ${methodTitle}`,
        ),
      )
    }

    body.appendChild(buildIterationTableElement(res.iteraciones, 'Tabla de iteraciones'))
  }

  function runSolve(shouldTrack: boolean): void {
    const readA = gridA.readMatrix()
    if (!readA.ok) {
      showBannerError(inputErrorEl, readA.error)
      readA.firstInvalid.focus()
      return
    }
    const readB = readVector(vectorBInputs, vectorBValues, 'vector b')
    if (!readB.ok) {
      showBannerError(inputErrorEl, readB.error)
      readB.firstInvalid.focus()
      return
    }
    const readX0 = readVector(vectorX0Inputs, vectorX0Values, 'vector inicial x₀')
    if (!readX0.ok) {
      showBannerError(inputErrorEl, readX0.error)
      readX0.firstInvalid.focus()
      return
    }
    const readP = readParams()
    if (!readP.ok) {
      showBannerError(inputErrorEl, readP.error)
      readP.focusEl.focus()
      return
    }

    showBannerError(inputErrorEl, null)
    emptyHintEl.hidden = true
    hasActiveResult = true

    const aRat: Matrix = readA.matriz
    const bRat: Rat[] = readB.rat
    const aNum: number[][] = aRat.map((row) => row.map((v) => v.toNumber()))
    const bNum: number[] = readB.num
    const x0Num: number[] = readX0.num
    const { tol, maxIter } = readP
    const method = methodSelect.value as IterativeSelectorValue

    if (!isSquare(aRat)) {
      lastChartState = null
      setResultError(
        resultCardEl,
        'Sistema no cuadrado',
        `Los solucionadores iterativos requieren una matriz cuadrada A (n×n). Tu matriz A es de ${aRat.length}×${aRat[0].length}. Iguala filas y columnas o usa RREF en la pestaña «Matrices y operaciones».`,
      )
      return
    }

    const n = aRat.length

    if (method === 'jacobi') {
      try {
        const res = jacobi(aNum, bNum, x0Num, tol, maxIter)
        renderSingleMethodResult('jacobi', 'Método de Jacobi', res, aRat, bRat, tol, maxIter)
        if (shouldTrack) {
          track('linealalrl-iter-solve', {
            method: 'jacobi',
            size: `${n}x${n}`,
            converged: res.convergio ? 'yes' : 'no',
          })
        }
      } catch (err) {
        lastChartState = null
        setResultError(
          resultCardEl,
          'Método de Jacobi',
          err instanceof Error ? err.message : String(err),
        )
      }
      return
    }

    if (method === 'gauss-seidel') {
      try {
        const res = gaussSeidel(aNum, bNum, x0Num, tol, maxIter)
        renderSingleMethodResult(
          'gauss-seidel',
          'Método de Gauss-Seidel',
          res,
          aRat,
          bRat,
          tol,
          maxIter,
        )
        if (shouldTrack) {
          track('linealalrl-iter-solve', {
            method: 'gauss-seidel',
            size: `${n}x${n}`,
            converged: res.convergio ? 'yes' : 'no',
          })
        }
      } catch (err) {
        lastChartState = null
        setResultError(
          resultCardEl,
          'Método de Gauss-Seidel',
          err instanceof Error ? err.message : String(err),
        )
      }
      return
    }

    if (method === 'sor') {
      try {
        const res = sor(aNum, bNum, x0Num, currentOmega, tol, maxIter)
        renderSingleMethodResult(
          'sor',
          `SOR (ω = ${currentOmega.toFixed(2)})`,
          res,
          aRat,
          bRat,
          tol,
          maxIter,
        )
        if (shouldTrack) {
          track('linealalrl-iter-solve', {
            method: 'sor',
            size: `${n}x${n}`,
            omega: currentOmega.toFixed(2),
            converged: res.convergio ? 'yes' : 'no',
          })
        }
      } catch (err) {
        lastChartState = null
        setResultError(
          resultCardEl,
          'Método SOR',
          err instanceof Error ? err.message : String(err),
        )
      }
      return
    }

    if (method === 'cg') {
      const res = conjugateGradient(aNum, bNum, x0Num, tol, maxIter)
      if ('error' in res) {
        lastChartState = null
        const body = createResultFrame(
          resultCardEl,
          'Gradiente conjugado (CG)',
          `${n}×${n} · No aplicable`,
        )
        const err = document.createElement('div')
        err.className = 'la-error-banner'
        err.setAttribute('role', 'alert')
        err.textContent = res.error
        body.appendChild(err)

        const actions = document.createElement('div')
        actions.className = 'la-inline-actions'
        const btnGs = document.createElement('button')
        btnGs.type = 'button'
        btnGs.className = 'la-btn la-btn--sm la-btn--primary'
        btnGs.textContent = 'Probar con Gauss-Seidel'
        btnGs.addEventListener('click', () => {
          methodSelect.value = 'gauss-seidel'
          updateMethodUI()
          runSolve(true)
        })
        const btnSor = document.createElement('button')
        btnSor.type = 'button'
        btnSor.className = 'la-btn la-btn--sm'
        btnSor.textContent = 'Probar con SOR'
        btnSor.addEventListener('click', () => {
          methodSelect.value = 'sor'
          updateMethodUI()
          runSolve(true)
        })
        actions.appendChild(btnGs)
        actions.appendChild(btnSor)
        body.appendChild(actions)

        if (shouldTrack) {
          track('linealalrl-iter-solve', {
            method: 'cg',
            size: `${n}x${n}`,
            status: 'na',
          })
        }
        return
      }

      renderSingleMethodResult('cg', 'Gradiente conjugado (CG)', res, aRat, bRat, tol, maxIter)
      if (shouldTrack) {
        track('linealalrl-iter-solve', {
          method: 'cg',
          size: `${n}x${n}`,
          converged: res.convergio ? 'yes' : 'no',
        })
      }
      return
    }

    // Comparar todos
    const resJacobi = jacobi(aNum, bNum, x0Num, tol, maxIter)
    const resGs = gaussSeidel(aNum, bNum, x0Num, tol, maxIter)
    const resSor = sor(aNum, bNum, x0Num, currentOmega, tol, maxIter)
    const resCg = conjugateGradient(aNum, bNum, x0Num, tol, maxIter)

    type CompareEntry = {
      key: 'jacobi' | 'gauss-seidel' | 'sor' | 'cg'
      label: string
      aplicable: boolean
      result?: IterResult
      errorMsg?: string
    }

    const entries: CompareEntry[] = [
      { key: 'jacobi', label: 'Jacobi', aplicable: true, result: resJacobi },
      { key: 'gauss-seidel', label: 'Gauss-Seidel', aplicable: true, result: resGs },
      {
        key: 'sor',
        label: `SOR (ω = ${currentOmega.toFixed(2)})`,
        aplicable: true,
        result: resSor,
      },
      'error' in resCg
        ? {
            key: 'cg',
            label: 'Gradiente conjugado',
            aplicable: false,
            errorMsg: resCg.error,
          }
        : {
            key: 'cg',
            label: 'Gradiente conjugado',
            aplicable: true,
            result: resCg,
          },
    ]

    const convergedCount = entries.filter((e) => e.aplicable && e.result?.convergio).length
    const body = createResultFrame(
      resultCardEl,
      'Comparador de solucionadores iterativos',
      `${n}×${n} · ${convergedCount} de ${entries.length} convergieron · tol = ${tol.toExponential(1)}`,
    )

    const criterionNote = document.createElement('p')
    criterionNote.className = 'la-note'
    criterionNote.textContent =
      'Criterio de error compartido: residuo euclidiano e_k = ‖Ax_k − b‖₂. Mueve el deslizador de ω para comparar cómo cambia SOR en tiempo real.'
    body.appendChild(criterionNote)

    const colors = getMethodColors()
    const chartSeries: ConvergenceSeries[] = []
    for (const e of entries) {
      if (e.aplicable && e.result && e.result.iteraciones.length > 0) {
        chartSeries.push({
          label: e.label,
          color: colors[e.key],
          steps: e.result.iteraciones,
        })
      }
    }

    body.appendChild(
      createChartBlock(
        chartSeries,
        tol,
        'Comparación de convergencia (error ‖Ax_k − b‖₂ en escala logarítmica)',
      ),
    )

    // Tabla resumen comparativa
    const tableWrap = document.createElement('div')
    tableWrap.className = 'la-table-wrap'
    const table = document.createElement('table')
    table.className = 'la-table'
    table.innerHTML = `
      <thead>
        <tr>
          <th scope="col">Método</th>
          <th scope="col">Estado</th>
          <th scope="col">Iteraciones</th>
          <th scope="col">Solución x (6 dec.)</th>
          <th scope="col">Error final ‖Ax − b‖₂</th>
          <th scope="col">Observación</th>
        </tr>
      </thead>
      <tbody></tbody>
    `
    const tbody = table.querySelector('tbody')!

    for (const e of entries) {
      const tr = document.createElement('tr')
      if (!e.aplicable) tr.className = 'is-not-applicable'

      const tdMethod = document.createElement('td')
      tdMethod.className = 'la-table-method'
      const swatch = document.createElement('span')
      swatch.className = 'la-iter-swatch'
      swatch.style.background = colors[e.key]
      tdMethod.appendChild(swatch)
      tdMethod.appendChild(document.createTextNode(e.label))

      const tdStatus = document.createElement('td')
      const pill = document.createElement('span')
      if (!e.aplicable) {
        pill.className = 'la-table-pill la-table-pill--na'
        pill.textContent = 'No aplica'
      } else if (e.result!.convergio) {
        pill.className = 'la-table-pill la-table-pill--exact'
        pill.textContent = 'Convergió'
      } else {
        pill.className = 'la-table-pill la-table-pill--na'
        pill.textContent = 'No convergió'
      }
      tdStatus.appendChild(pill)

      const tdIters = document.createElement('td')
      tdIters.className = 'la-table-mono'
      tdIters.textContent = e.aplicable ? String(e.result!.iteraciones.length) : '—'

      const tdSol = document.createElement('td')
      tdSol.className = 'la-table-mono'
      if (!e.aplicable || !e.result || e.result.iteraciones.length === 0) {
        tdSol.textContent = '—'
      } else {
        tdSol.textContent = formatVectorFixed6(e.result.x)
      }

      const tdErr = document.createElement('td')
      tdErr.className = 'la-table-mono'
      if (!e.aplicable || !e.result || e.result.iteraciones.length === 0) {
        tdErr.textContent = '—'
      } else {
        const last = e.result.iteraciones[e.result.iteraciones.length - 1]
        tdErr.textContent = formatErr(last.error)
      }

      const tdObs = document.createElement('td')
      if (!e.aplicable) {
        const span = document.createElement('span')
        span.className = 'la-table-reason'
        span.textContent = e.errorMsg ?? 'No aplicable.'
        tdObs.appendChild(span)
      } else if (e.result!.convergio) {
        tdObs.textContent = 'Alcanzó la tolerancia objetivo.'
      } else {
        const span = document.createElement('span')
        span.className = 'la-table-reason'
        span.textContent = e.result!.motivo ?? 'No convergió.'
        tdObs.appendChild(span)
      }

      tr.appendChild(tdMethod)
      tr.appendChild(tdStatus)
      tr.appendChild(tdIters)
      tr.appendChild(tdSol)
      tr.appendChild(tdErr)
      tr.appendChild(tdObs)
      tbody.appendChild(tr)
    }

    tableWrap.appendChild(table)
    body.appendChild(tableWrap)

    const bestConverged = entries.find((e) => e.aplicable && e.result?.convergio)?.result?.x
    renderExactReferenceBlock(body, aRat, bRat, bestConverged)

    // Detalle de tabla de iteraciones por método (actualizable al mover ω)
    const detailWrap = document.createElement('div')
    detailWrap.className = 'la-iter-detail-wrap'

    const detailTabs = document.createElement('div')
    detailTabs.className = 'la-inline-actions'
    const detailLabel = document.createElement('span')
    detailLabel.className = 'la-toolbar-label'
    detailLabel.textContent = 'Tabla de iteraciones de:'
    detailTabs.appendChild(detailLabel)

    const detailTableHost = document.createElement('div')
    const detailBtns = new Map<string, HTMLButtonElement>()

    const selectable = entries.filter((e) => e.aplicable && e.result)
    if (!selectable.some((e) => e.key === compareDetailKey) && selectable.length > 0) {
      compareDetailKey = selectable[0].key
    }

    const renderDetailTable = (key: 'jacobi' | 'gauss-seidel' | 'sor' | 'cg'): void => {
      compareDetailKey = key
      for (const [k, b] of detailBtns.entries()) {
        b.classList.toggle('la-btn--active', k === key)
      }
      detailTableHost.innerHTML = ''
      const found = entries.find((e) => e.key === key)
      if (found && found.result) {
        detailTableHost.appendChild(buildIterationTableElement(found.result.iteraciones))
      }
    }

    for (const e of selectable) {
      const btn = document.createElement('button')
      btn.type = 'button'
      btn.className = 'la-btn la-btn--sm'
      btn.textContent = e.label
      btn.addEventListener('click', () => renderDetailTable(e.key))
      detailTabs.appendChild(btn)
      detailBtns.set(e.key, btn)
    }

    detailWrap.appendChild(detailTabs)
    detailWrap.appendChild(detailTableHost)
    body.appendChild(detailWrap)
    renderDetailTable(compareDetailKey)

    if (shouldTrack) {
      track('linealalrl-iter-compare', { size: `${n}x${n}`, omega: currentOmega.toFixed(2) })
    }
  }

  // Eventos de controles
  methodSelect.addEventListener('change', () => {
    updateMethodUI()
    if (hasActiveResult) {
      runSolve(true)
    }
  })

  omegaSlider.addEventListener('input', () => {
    const val = Number(omegaSlider.value)
    if (!Number.isFinite(val) || val <= 0 || val >= 2) return
    currentOmega = val
    omegaValEl.textContent = val.toFixed(2)
    updateSliderProgress(val)
    if (methodSelect.value === 'compare') {
      compareDetailKey = 'sor'
    }
    runSolve(false)
  })

  btnUseOpt.addEventListener('click', () => {
    if (lastOptOmega === null) return
    const clamped = Math.max(0.01, Math.min(1.99, Math.round(lastOptOmega * 100) / 100))
    currentOmega = clamped
    omegaSlider.value = clamped.toFixed(2)
    omegaValEl.textContent = clamped.toFixed(2)
    updateSliderProgress(clamped)
    if (methodSelect.value === 'compare') {
      compareDetailKey = 'sor'
    }
    runSolve(false)
  })

  btnSolve.addEventListener('click', () => runSolve(true))

  // Redibujar gráfica al cambiar tema o redimensionar / mostrar la pestaña
  if (typeof MutationObserver !== 'undefined' && typeof document !== 'undefined') {
    const themeObs = new MutationObserver(() => redrawActiveChart())
    themeObs.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['data-theme'],
    })
  }

  if (typeof ResizeObserver !== 'undefined') {
    const resizeObs = new ResizeObserver(() => redrawActiveChart())
    resizeObs.observe(resultCardEl)
  }
}
