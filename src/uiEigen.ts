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

// Interfaz interactiva del módulo 3: valores y vectores propios
// (polinomio característico exacto, método de la potencia, iteración QR, Jacobi y multiplicidades).

import {
  type JacobiEigenResult,
  type MultiplicidadExacta,
  type MultiplicidadResult,
  type PowerMethodResult,
  type QrIterationResult,
  charPoly,
  charPolyLatex,
  charPolyPlain,
  eigenResidual,
  esDefectiva,
  exactEigenvectors,
  isSymmetricExact,
  jacobiEigen,
  multiplicidades,
  powerMethod,
  qrIteration,
  rationalRoots,
} from './eigen.ts'
import { renderLatex } from './katexRender.ts'
import { type Matrix, isSquare } from './matrix.ts'
import { parsePastedMatrix } from './paste.ts'
import { type Rat, parseRat, ratToDecimal, ratToLatex, ratToString } from './rational.ts'
import { MAX_GRID_DIM } from './share.ts'
import {
  type TrackFn,
  buildLatexCopyBlock,
  createMatrixGrid,
  createResultFrame,
  setResultError,
} from './ui.ts'

const LATEX_HEADER = '% Requiere \\usepackage{amsmath}'
const INITIAL_POWER_ROWS = 100

export interface EigenPreset {
  id: string
  nombre: string
  descripcion: string
  a: string[][]
  x0?: string[]
  shift?: string
  inverse?: boolean
}

export const EIGEN_PRESETS: readonly EigenPreset[] = [
  {
    id: 'sim-2x2',
    nombre: 'Simétrica 2×2 (λ = 1, 3)',
    descripcion:
      'Matriz simétrica 2×2 clásica con polinomio característico λ² − 4λ + 3 y valores propios exactos λ = 1 y λ = 3.',
    a: [
      ['2', '1'],
      ['1', '2'],
    ],
    x0: ['1', '0'],
  },
  {
    id: 'sim-3x3',
    nombre: 'Simétrica 3×3 (λ = 7, 4, 1)',
    descripcion:
      'Matriz simétrica 3×3 definida positiva con valores propios exactos λ = 7, 4, 1: ideal para Jacobi (A = QDQᵀ), potencia (con o sin desplazamiento) y QR.',
    a: [
      ['5', '0', '2'],
      ['0', '3', '2'],
      ['2', '2', '4'],
    ],
    x0: ['1', '2', '3'],
  },
  {
    id: 'defectiva-2x2',
    nombre: 'Defectiva 2×2 (λ = 1 doble)',
    descripcion:
      'Bloque de Jordan [[1, 1], [0, 1]]: valor propio λ = 1 con multiplicidad algebraica 2 y geométrica 1 (matriz defectiva, no diagonalizable).',
    a: [
      ['1', '1'],
      ['0', '1'],
    ],
    x0: ['1', '2'],
  },
  {
    id: 'irracional-2x2',
    nombre: 'Simétrica irracional 2×2 (φ)',
    descripcion:
      'Polinomio exacto λ² − λ − 1 sin raíces racionales: los valores propios φ ≈ 1.618034 y 1 − φ ≈ −0.618034 se obtienen de forma numérica.',
    a: [
      ['1', '1'],
      ['1', '0'],
    ],
    x0: ['1', '0'],
  },
]

function formatNum(v: number): string {
  if (!Number.isFinite(v)) return String(v)
  if (Math.abs(v) < 1e-12) return '0'
  const rounded = Math.round(v)
  if (Math.abs(v - rounded) < 1e-9 && Math.abs(v) < 1e9) {
    return String(rounded)
  }
  if (Math.abs(v) >= 1e5 || Math.abs(v) < 1e-4) {
    return v.toExponential(4)
  }
  return Number(v.toFixed(6)).toString()
}

/**
 * Formatea un número para modo matemático LaTeX/KaTeX.
 * Convierte notación científica `a.bcde+n` en `a.bcd \times 10^{n}` para que
 * la `e` no se interprete como variable en modo matemático.
 */
export function formatNumLatex(v: number): string {
  if (!Number.isFinite(v)) {
    if (Number.isNaN(v)) return '\\text{NaN}'
    return v < 0 ? '-\\infty' : '\\infty'
  }
  const s = formatNum(v)
  const match = /^([+-]?\d+(?:\.\d+)?)[eE]([+-]?\d+)$/.exec(s)
  if (!match) return s
  const mantissa = match[1].replace(/(\.\d*?[1-9])0+$/, '$1').replace(/\.0+$/, '')
  const exp = Number(match[2])
  return `${mantissa} \\times 10^{${exp}}`
}

const DEFAULT_CALC_ERROR = 'No se pudo completar el cálculo.'

/**
 * Devuelve un mensaje en español seguro para mostrar al usuario tras una excepción del núcleo.
 * Conserva `err.message` si ya viene redactado en español; en caso contrario usa un texto genérico.
 */
export function formatEigenErrorMessage(err: unknown): string {
  if (!(err instanceof Error)) return DEFAULT_CALC_ERROR
  if (
    err instanceof TypeError ||
    err instanceof RangeError ||
    err instanceof ReferenceError ||
    err instanceof SyntaxError
  ) {
    return DEFAULT_CALC_ERROR
  }
  const msg = err.message.trim()
  if (!msg) return DEFAULT_CALC_ERROR
  const looksSpanish =
    /[áéíóúüñ¡¿]/i.test(msg) ||
    /\b(la|el|las|los|una|se|del|en|por|para|con|es|son|requiere|debe|puede|pudo|matriz|valor|cero|finito|finita|cuadrada|positiva|positivo|entero|dimensiones|incompatibles|recibido|recibida)\b/i.test(
      msg,
    )
  return looksSpanish ? msg : DEFAULT_CALC_ERROR
}

export function formatRootWithMultiplicity(
  root: Rat,
  algebraica: number | undefined,
): { latex: string; plain: string } {
  const key = ratToString(root)
  const tex = ratToLatex(root)
  if (algebraica === undefined) {
    return {
      latex: `${tex} \\;(\\text{mult. —})`,
      plain: `${key} (mult. —)`,
    }
  }
  if (algebraica > 1) {
    return {
      latex: `${tex} \\;(\\text{mult. } ${algebraica})`,
      plain: `${key} (mult. ${algebraica})`,
    }
  }
  return { latex: tex, plain: key }
}

export const NON_RATIONAL_MULT_NOTE =
  'Para raíces no racionales, la multiplicidad algebraica no está disponible en aritmética exacta de Rat (requiere factorización simbólica fuera de ℚ) y se muestra como «—»; la multiplicidad geométrica se estima mediante el rango numérico de A − λI.'

export function formatNumericMultiplicityCells(
  m: Pick<MultiplicidadResult, 'algebraica' | 'geometrica'>,
): { algText: string; geomText: string } {
  if (m.algebraica === null) {
    return {
      algText: '—',
      geomText: `≈ ${m.geometrica} (estimada)`,
    }
  }
  return {
    algText: String(m.algebraica),
    geomText: String(m.geometrica),
  }
}

export function formatIncompleteRationalNotice(totalRationalMult: number, degree: number): string {
  if (totalRationalMult <= 0) {
    return 'Aviso: el polinomio característico no tiene raíces racionales (p(r) ≠ 0 para todo candidato racional). Las raíces pueden ser irracionales o complejas; las secciones numéricas aproximan valores propios reales y no separan pares complejos.'
  }
  const missing = degree - totalRationalMult
  const missingClause =
    missing === 1
      ? 'La raíz que falta puede ser irracional o compleja'
      : `Las ${missing} raíces que faltan pueden ser irracionales o complejas`
  return `Aviso: las raíces racionales suman multiplicidad ${totalRationalMult} de ${degree}. ${missingClause}; las secciones numéricas aproximan valores propios reales y no separan pares complejos.`
}

export function describeJacobiBadge(
  res: Pick<JacobiEigenResult, 'valores' | 'iteraciones' | 'residuo' | 'residuoOrtogonalidad'>,
): { ok: boolean; text: string } {
  const rotLabel = `${res.iteraciones} ${res.iteraciones === 1 ? 'rotación' : 'rotaciones'}`
  const scale = Math.max(1, ...res.valores.map((v) => (Number.isFinite(v) ? Math.abs(v) : 0)))
  const ok =
    Number.isFinite(res.residuo) &&
    Number.isFinite(res.residuoOrtogonalidad) &&
    res.residuo <= 1e-8 * scale &&
    res.residuoOrtogonalidad <= 1e-8
  if (ok) {
    return {
      ok: true,
      text: `Diagonalización ortogonal completada en ${rotLabel} de Jacobi (residuo ${formatResidual(res.residuo)})`,
    }
  }
  return {
    ok: false,
    text: `Jacobi terminó en ${rotLabel} con residuo elevado (${formatResidual(res.residuo)})`,
  }
}

function formatResidual(r: number): string {
  if (!Number.isFinite(r)) return '—'
  if (r === 0) return '0'
  return r.toExponential(2)
}

function numMatrixToLatex(m: number[][]): string {
  const filas = m.map((row) => row.map(formatNumLatex).join(' & ')).join(' \\\\ ')
  return `\\begin{bmatrix} ${filas} \\end{bmatrix}`
}

function numMatrixToString(m: number[][]): string {
  return m.map((row) => row.map(formatNum).join(' ')).join('\n')
}

function numVectorToLatex(v: number[]): string {
  return `\\begin{bmatrix} ${v.map(formatNumLatex).join(' \\\\ ')} \\end{bmatrix}`
}

function numVectorToPlain(v: number[]): string {
  return `[${v.map(formatNum).join(', ')}]`
}

function ratVectorToLatex(v: Rat[]): string {
  return `\\begin{bmatrix} ${v.map(ratToLatex).join(' \\\\ ')} \\end{bmatrix}`
}

function ratVectorToPlain(v: Rat[]): string {
  return `[${v.map(ratToString).join(', ')}]`
}

function diagMatrixNum(vals: number[]): number[][] {
  const n = vals.length
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? vals[i] : 0)),
  )
}

function createMathBox(tex: string, fallbackText: string, compact = false): HTMLDivElement {
  const box = document.createElement('div')
  box.className = compact ? 'la-result-math la-result-math--compact' : 'la-result-math'
  void renderLatex(tex, box, { displayMode: true, fallbackText })
  return box
}

function defaultX0(n: number): string[] {
  return Array.from({ length: n }, (_, i) => String(i + 1))
}

function parseNumericField(raw: string): number | null {
  const limpio = raw.trim().replace(/−/g, '-').replace(/,/g, '.')
  if (!limpio) return null
  const r = parseRat(limpio)
  if (r !== null) return r.toNumber()
  const n = Number(limpio)
  return Number.isFinite(n) ? n : null
}

/**
 * Monta la interfaz de la pestaña «Valores propios» dentro de `app`.
 */
export function mountEigenUI(app: HTMLElement, track: TrackFn): void {
  const defaultPreset = EIGEN_PRESETS[0]
  let activePresetId: string | null = defaultPreset.id
  let x0Values: string[] = (defaultPreset.x0 ?? defaultX0(defaultPreset.a.length)).slice()
  let x0Inputs: HTMLInputElement[] = []

  app.innerHTML = `
    <div class="la-workspace">
      <section class="la-panel">
        <div class="la-panel-head">
          <h2 class="la-panel-title">Matriz A (valores y vectores propios)</h2>
          <span class="la-panel-meta" id="eigen-dim-meta"></span>
        </div>

        <div class="la-toolbar" id="eigen-presets-bar">
          <span class="la-toolbar-label">Presets:</span>
        </div>
        <p class="la-preset-desc" id="eigen-preset-desc"></p>

        <div class="la-panel-body la-direct-input-layout">
          <div class="la-direct-system-editors">
            <div class="la-direct-matrix-block">
              <span class="la-field-label">Matriz cuadrada A (hasta 6×6)</span>
              <div id="eigen-grid-a-mount"></div>
            </div>

            <div class="la-direct-vector-block">
              <div class="la-vector-head">
                <span class="la-field-label">Vector inicial x₀ (potencia)</span>
                <button class="la-btn la-btn--sm" id="btn-eigen-reset-x0" type="button" aria-label="Restablecer vector inicial x0">Restablecer x₀</button>
              </div>
              <div class="la-matrix-scroll">
                <div class="la-matrix-bracket">
                  <div class="la-vector-grid" id="eigen-x0-grid"></div>
                </div>
              </div>
            </div>
          </div>

          <div class="la-paste-box">
            <label class="la-field-label" for="eigen-paste-input">Pegar matriz A</label>
            <p class="la-hint" id="eigen-paste-hint">Pega una matriz cuadrada (separada por espacios, comas o tabuladores, hasta 6×6).</p>
            <textarea
              id="eigen-paste-input"
              class="la-textarea"
              rows="4"
              spellcheck="false"
              aria-describedby="eigen-paste-hint"
              placeholder="5 0 2&#10;0 3 2&#10;2 2 4"
            ></textarea>
            <div class="la-paste-actions">
              <button class="la-btn la-btn--sm" id="btn-eigen-paste" type="button">Pegar matriz</button>
              <span class="la-feedback" id="eigen-paste-feedback" role="status" aria-live="polite" hidden></span>
            </div>
            <div class="la-error-banner" id="eigen-paste-error" role="alert" hidden></div>
          </div>
        </div>

        <div class="la-error-banner la-panel-error" id="eigen-input-error" role="alert" hidden></div>
      </section>

      <section class="la-panel">
        <div class="la-panel-head">
          <h2 class="la-panel-title">Análisis espectral</h2>
          <span class="la-panel-meta">Polinomio y multiplicidades exactos (Rat) · Potencia, QR y Jacobi numéricos</span>
        </div>

        <div class="la-toolbar la-iter-controls-bar">
          <label class="la-iter-param-field" for="eigen-shift-input">
            <span>Desplazamiento σ:</span>
            <input
              id="eigen-shift-input"
              class="la-input la-input--compact"
              type="text"
              value="0"
              spellcheck="false"
              aria-label="Desplazamiento shift para el método de la potencia"
            />
          </label>

          <label class="la-eigen-check-label" for="eigen-inverse-check">
            <input id="eigen-inverse-check" type="checkbox" />
            <span>Potencia inversa ((A − σI)y = x vía Gauss)</span>
          </label>

          <label class="la-iter-param-field" for="eigen-tol-input">
            <span>Tolerancia:</span>
            <input
              id="eigen-tol-input"
              class="la-input la-input--compact"
              type="text"
              value="1e-8"
              spellcheck="false"
              aria-label="Tolerancia para potencia y QR"
            />
          </label>

          <label class="la-iter-param-field" for="eigen-maxiter-input">
            <span>Máx. iter:</span>
            <input
              id="eigen-maxiter-input"
              class="la-input la-input--compact"
              type="number"
              min="1"
              max="5000"
              step="1"
              value="200"
              aria-label="Máximo de iteraciones para potencia y QR"
            />
          </label>

          <button class="la-btn la-btn--primary" id="btn-eigen-solve" type="button">Calcular valores propios</button>
        </div>

        <div class="la-panel-body la-results-stack">
          <p class="la-note" id="eigen-empty-hint">
            Pulsa <strong>Calcular valores propios</strong> para obtener el polinomio característico exacto, las multiplicidades, el método de la potencia (con desplazamiento o potencia inversa), la iteración QR y la diagonalización ortogonal de Jacobi si la matriz es simétrica.
          </p>
          <div class="la-results-stack" id="eigen-results-container" hidden>
            <article class="la-result-card" id="eigen-card-poly"></article>
            <article class="la-result-card" id="eigen-card-mult"></article>
            <article class="la-result-card" id="eigen-card-power"></article>
            <article class="la-result-card" id="eigen-card-qr"></article>
            <article class="la-result-card" id="eigen-card-jacobi"></article>
          </div>
        </div>
      </section>
    </div>
  `

  const presetsBar = app.querySelector<HTMLDivElement>('#eigen-presets-bar')!
  const presetDescEl = app.querySelector<HTMLParagraphElement>('#eigen-preset-desc')!
  const dimMetaEl = app.querySelector<HTMLSpanElement>('#eigen-dim-meta')!
  const gridAMount = app.querySelector<HTMLDivElement>('#eigen-grid-a-mount')!
  const x0Grid = app.querySelector<HTMLDivElement>('#eigen-x0-grid')!
  const btnResetX0 = app.querySelector<HTMLButtonElement>('#btn-eigen-reset-x0')!
  const pasteInput = app.querySelector<HTMLTextAreaElement>('#eigen-paste-input')!
  const btnPaste = app.querySelector<HTMLButtonElement>('#btn-eigen-paste')!
  const pasteFeedbackEl = app.querySelector<HTMLSpanElement>('#eigen-paste-feedback')!
  const pasteErrorEl = app.querySelector<HTMLDivElement>('#eigen-paste-error')!
  const inputErrorEl = app.querySelector<HTMLDivElement>('#eigen-input-error')!

  const shiftInput = app.querySelector<HTMLInputElement>('#eigen-shift-input')!
  const inverseCheck = app.querySelector<HTMLInputElement>('#eigen-inverse-check')!
  const tolInput = app.querySelector<HTMLInputElement>('#eigen-tol-input')!
  const maxIterInput = app.querySelector<HTMLInputElement>('#eigen-maxiter-input')!
  const btnSolve = app.querySelector<HTMLButtonElement>('#btn-eigen-solve')!
  const emptyHintEl = app.querySelector<HTMLParagraphElement>('#eigen-empty-hint')!
  const resultsContainer = app.querySelector<HTMLDivElement>('#eigen-results-container')!
  const cardPoly = app.querySelector<HTMLElement>('#eigen-card-poly')!
  const cardMult = app.querySelector<HTMLElement>('#eigen-card-mult')!
  const cardPower = app.querySelector<HTMLElement>('#eigen-card-power')!
  const cardQr = app.querySelector<HTMLElement>('#eigen-card-qr')!
  const cardJacobi = app.querySelector<HTMLElement>('#eigen-card-jacobi')!

  const allCards = [cardPoly, cardMult, cardPower, cardQr, cardJacobi]

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

  function clearResultsStale(): void {
    for (const card of allCards) {
      card.classList.remove('is-stale')
      for (const banner of card.querySelectorAll('.la-stale-banner')) {
        banner.remove()
      }
    }
  }

  function markResultsStale(): void {
    if (resultsContainer.hidden) return
    for (const card of allCards) {
      if (card.hidden || card.classList.contains('is-stale')) continue
      card.classList.add('is-stale')
      const head = card.querySelector('.la-result-head')
      const notice = document.createElement('div')
      notice.className = 'la-stale-banner'
      notice.textContent = 'Resultado de una versión anterior de la matriz o parámetros'
      if (head && head.nextSibling) {
        card.insertBefore(notice, head.nextSibling)
      } else {
        card.appendChild(notice)
      }
    }
  }

  function renderX0(rowCount: number): void {
    x0Values = Array.from({ length: rowCount }, (_, i) => x0Values[i] ?? String(i + 1))
    x0Grid.innerHTML = ''
    x0Inputs = []

    for (let i = 0; i < rowCount; i++) {
      const row = document.createElement('div')
      row.className = 'la-vector-row'

      const tag = document.createElement('span')
      tag.className = 'la-vector-idx'
      tag.textContent = `x${i + 1}`

      const input = document.createElement('input')
      input.type = 'text'
      input.className = 'la-cell-input'
      input.value = x0Values[i]
      input.spellcheck = false
      input.autocomplete = 'off'
      input.setAttribute('aria-label', `Vector inicial x₀, componente ${i + 1}`)
      input.setAttribute('aria-invalid', 'false')

      input.addEventListener('input', () => {
        x0Values[i] = input.value
        if (input.classList.contains('is-invalid') && parseNumericField(input.value) !== null) {
          input.classList.remove('is-invalid')
          input.setAttribute('aria-invalid', 'false')
        }
        showBannerError(inputErrorEl, null)
        markResultsStale()
      })

      row.appendChild(tag)
      row.appendChild(input)
      x0Grid.appendChild(row)
      x0Inputs.push(input)
    }
  }

  function readX0():
    | { ok: true; vector: number[] }
    | { ok: false; error: string; firstInvalid: HTMLInputElement } {
    const out: number[] = []
    let firstInvalid: HTMLInputElement | null = null
    const badRows: number[] = []

    for (let i = 0; i < x0Inputs.length; i++) {
      const input = x0Inputs[i]
      x0Values[i] = input.value
      const val = parseNumericField(input.value)
      if (val === null) {
        input.classList.add('is-invalid')
        input.setAttribute('aria-invalid', 'true')
        if (!firstInvalid) firstInvalid = input
        badRows.push(i + 1)
      } else {
        input.classList.remove('is-invalid')
        input.setAttribute('aria-invalid', 'false')
        out.push(val)
      }
    }

    if (firstInvalid) {
      return {
        ok: false,
        firstInvalid,
        error: `Revisa el vector inicial x₀ en ${badRows.length === 1 ? `la componente ${badRows[0]}` : `las componentes ${badRows.join(', ')}`}: escribe números válidos.`,
      }
    }

    if (out.every((v) => v === 0)) {
      const first = x0Inputs[0]
      first.classList.add('is-invalid')
      first.setAttribute('aria-invalid', 'true')
      return {
        ok: false,
        firstInvalid: first,
        error: 'El vector inicial x₀ no puede ser el vector nulo (todas las entradas son 0).',
      }
    }

    return { ok: true, vector: out }
  }

  function updateDimMeta(): void {
    const [r, c] = gridA.getSize()
    dimMetaEl.textContent =
      r === c
        ? `Matriz cuadrada ${r}×${c}`
        : `A es ${r}×${c} (no cuadrada: los valores propios requieren una matriz cuadrada n×n)`
  }

  const presetButtons = new Map<string, HTMLButtonElement>()
  function updatePresetButtons(): void {
    for (const [id, btn] of presetButtons.entries()) {
      const active = id === activePresetId
      btn.classList.toggle('la-btn--active', active)
      btn.setAttribute('aria-pressed', active ? 'true' : 'false')
    }
    const preset = EIGEN_PRESETS.find((p) => p.id === activePresetId)
    presetDescEl.textContent = preset
      ? preset.descripcion
      : 'Matriz personalizada (admite enteros, decimales y fracciones como 3/4).'
  }

  const gridA = createMatrixGrid({
    idPrefix: 'eigen-grid-a',
    matrixName: 'A',
    initialValues: defaultPreset.a,
    onSizeChange: (rows) => {
      renderX0(rows)
      updateDimMeta()
    },
    onInputChange: () => {
      activePresetId = null
      showBannerError(inputErrorEl, null)
      markResultsStale()
      updatePresetButtons()
    },
  })

  gridAMount.appendChild(gridA.element)
  renderX0(defaultPreset.a.length)
  updateDimMeta()

  function applyPreset(preset: EigenPreset): void {
    activePresetId = preset.id
    x0Values = (preset.x0 ?? defaultX0(preset.a.length)).slice()
    shiftInput.value = preset.shift ?? '0'
    inverseCheck.checked = preset.inverse === true
    gridA.setValues(preset.a)
    renderX0(preset.a.length)
    showBannerError(inputErrorEl, null)
    showBannerError(pasteErrorEl, null)
    markResultsStale()
    updatePresetButtons()
    updateDimMeta()
  }

  for (const preset of EIGEN_PRESETS) {
    const btn = document.createElement('button')
    btn.type = 'button'
    btn.className = 'la-btn la-btn--sm'
    btn.textContent = preset.nombre
    btn.addEventListener('click', () => {
      applyPreset(preset)
      track('linealalrl-eigen-preset', { preset: preset.id })
    })
    presetsBar.appendChild(btn)
    presetButtons.set(preset.id, btn)
  }
  updatePresetButtons()

  btnResetX0.addEventListener('click', () => {
    const [rows] = gridA.getSize()
    x0Values = defaultX0(rows)
    renderX0(rows)
    markResultsStale()
  })

  shiftInput.addEventListener('input', () => {
    showBannerError(inputErrorEl, null)
    markResultsStale()
  })
  inverseCheck.addEventListener('change', () => {
    showBannerError(inputErrorEl, null)
    markResultsStale()
  })
  tolInput.addEventListener('input', () => {
    showBannerError(inputErrorEl, null)
    markResultsStale()
  })
  maxIterInput.addEventListener('input', () => {
    showBannerError(inputErrorEl, null)
    markResultsStale()
  })

  btnPaste.addEventListener('click', () => {
    const parsed = parsePastedMatrix(pasteInput.value, MAX_GRID_DIM, MAX_GRID_DIM)
    if (!parsed.ok) {
      showBannerError(pasteErrorEl, parsed.error)
      pasteInput.focus()
      return
    }
    showBannerError(pasteErrorEl, null)
    showBannerError(inputErrorEl, null)
    activePresetId = null
    x0Values = defaultX0(parsed.filas)
    gridA.setMatrix(parsed.matriz)
    renderX0(parsed.filas)
    markResultsStale()
    updatePresetButtons()
    updateDimMeta()
    showTimedFeedback(pasteFeedbackEl, `Matriz cargada (${parsed.filas}×${parsed.cols})`)
    track('linealalrl-eigen-paste', { size: `${parsed.filas}x${parsed.cols}` })
  })

  function renderPolyCard(
    a: Matrix,
    coefs: Rat[],
    roots: Rat[],
    multByRoot: Map<string, MultiplicidadExacta>,
  ): void {
    const n = a.length
    const body = createResultFrame(
      cardPoly,
      'Polinomio característico y raíces racionales',
      `Faddeev–LeVerrier exacto · grado ${n}`,
    )

    const badge = document.createElement('div')
    badge.className = 'la-sys-badge la-sys-badge--unica'
    badge.textContent = 'Cálculo exacto (aritmética racional Rat, sin error de redondeo)'
    body.appendChild(badge)

    const polyTex = charPolyLatex(coefs)
    const polyPlain = charPolyPlain(coefs)
    body.appendChild(createMathBox(polyTex, polyPlain))

    let totalRationalMult = 0
    for (const info of multByRoot.values()) {
      totalRationalMult += info.algebraica
    }

    if (roots.length > 0) {
      const rootsHeading = document.createElement('h4')
      rootsHeading.className = 'la-subheading'
      rootsHeading.textContent =
        roots.length === 1
          ? 'Raíz racional exacta (teorema de la raíz racional)'
          : 'Raíces racionales exactas (teorema de la raíz racional)'
      body.appendChild(rootsHeading)

      const formattedRoots = roots.map((r) =>
        formatRootWithMultiplicity(r, multByRoot.get(ratToString(r))?.algebraica),
      )
      const rootLatexParts = formattedRoots.map((part) => part.latex)
      const rootPlainParts = formattedRoots.map((part) => part.plain)

      body.appendChild(
        createMathBox(
          `\\lambda \\in \\left\\{ ${rootLatexParts.join(',\\; ')} \\right\\}`,
          `Raíces racionales: ${rootPlainParts.join(', ')}`,
          true,
        ),
      )

      const hasFrac = roots.some((r) => r.d !== 1n)
      if (hasFrac) {
        const decLine = document.createElement('p')
        decLine.className = 'la-note'
        decLine.textContent =
          'Aproximación decimal: ' +
          roots.map((r) => `${ratToString(r)} ≈ ${ratToDecimal(r, 6)}`).join(', ')
        body.appendChild(decLine)
      }

      if (totalRationalMult < n) {
        const warn = document.createElement('div')
        warn.className = 'la-eigen-notice'
        warn.textContent = formatIncompleteRationalNotice(totalRationalMult, n)
        body.appendChild(warn)
      }
    } else {
      const warn = document.createElement('div')
      warn.className = 'la-eigen-notice'
      warn.textContent = formatIncompleteRationalNotice(0, n)
      body.appendChild(warn)
    }

    const latexCode = [
      LATEX_HEADER,
      `\\[ ${polyTex} \\]`,
      roots.length > 0
        ? `\\[ \\lambda_{\\mathbb{Q}} \\in \\left\\{ ${roots.map(ratToLatex).join(', ')} \\right\\} \\]`
        : '% Sin raíces racionales exactas.',
    ].join('\n')
    body.appendChild(buildLatexCopyBlock('eigen-poly', 'Polinomio característico', latexCode, track))
  }

  function renderMultiplicitiesCard(
    n: number,
    roots: Rat[],
    multByRoot: Map<string, MultiplicidadExacta>,
    eigenvectorsByRoot: Map<string, Rat[][]>,
    defective: boolean,
    extraNumericRows: Array<{ val: number; mult: MultiplicidadResult }>,
  ): void {
    const body = createResultFrame(
      cardMult,
      'Multiplicidades y diagonalizabilidad',
      `n = ${n} · rango exacto en ℚ`,
    )

    const defectBadge = document.createElement('div')
    defectBadge.className = defective
      ? 'la-sys-badge la-sys-badge--ninguna'
      : 'la-sys-badge la-sys-badge--unica'
    defectBadge.textContent = defective
      ? 'Matriz defectiva (no diagonalizable)'
      : 'No defectiva (diagonalizable sobre ℂ)'
    body.appendChild(defectBadge)

    if (defective) {
      const alertBox = document.createElement('div')
      alertBox.className = 'la-error-banner'
      alertBox.setAttribute('role', 'alert')
      alertBox.textContent =
        'Aviso: matriz defectiva (no diagonalizable). Al menos un valor propio tiene multiplicidad geométrica estrictamente menor que su multiplicidad algebraica, por lo que no existe una base completa de vectores propios.'
      body.appendChild(alertBox)
    }

    if (roots.length === 0 && extraNumericRows.length === 0) {
      const note = document.createElement('p')
      note.className = 'la-note'
      note.textContent =
        'No se encontraron valores propios reales (ni racionales ni numéricos convergentes) para construir la tabla de multiplicidades. La prueba de matriz defectiva anterior sigue siendo exacta sobre ℂ.'
      body.appendChild(note)
      return
    }

    const wrap = document.createElement('div')
    wrap.className = 'la-table-wrap'
    const table = document.createElement('table')
    table.className = 'la-table'
    table.innerHTML = `
      <thead>
        <tr>
          <th scope="col">Valor propio λ</th>
          <th scope="col">Tipo</th>
          <th scope="col">Mult. algebraica</th>
          <th scope="col">Mult. geométrica</th>
          <th scope="col">Espacio propio / Estado</th>
        </tr>
      </thead>
      <tbody></tbody>
    `
    const tbody = table.querySelector('tbody')!

    let hasNonRationalAviso = false

    for (const r of roots) {
      const key = ratToString(r)
      const m = multByRoot.get(key)
      const basis = eigenvectorsByRoot.get(key) ?? []

      const tr = document.createElement('tr')

      const tdVal = document.createElement('td')
      tdVal.className = 'la-table-mono'
      const spanMath = document.createElement('span')
      void renderLatex(ratToLatex(r), spanMath, { displayMode: false, fallbackText: key })
      tdVal.appendChild(spanMath)

      const tdType = document.createElement('td')
      const pill = document.createElement('span')
      pill.className = 'la-table-pill la-table-pill--exact'
      pill.textContent = 'Racional (ℚ)'
      tdType.appendChild(pill)

      const tdAlg = document.createElement('td')
      tdAlg.className = 'la-table-mono'
      tdAlg.textContent = m ? String(m.algebraica) : '—'

      const tdGeom = document.createElement('td')
      tdGeom.className = 'la-table-mono'
      tdGeom.textContent = m ? String(m.geometrica) : '—'

      const tdState = document.createElement('td')
      if (m && m.geometrica < m.algebraica) {
        const warnSpan = document.createElement('span')
        warnSpan.className = 'la-table-reason'
        warnSpan.textContent = `Defectivo (geom ${m.geometrica} < alg ${m.algebraica}) · `
        tdState.appendChild(warnSpan)
      }
      if (basis.length > 0) {
        const basisSpan = document.createElement('span')
        const basisTex = basis.map(ratVectorToLatex).join(',\\; ')
        const basisPlain = basis.map(ratVectorToPlain).join(', ')
        void renderLatex(`\\text{Base: } ${basisTex}`, basisSpan, {
          displayMode: false,
          fallbackText: `Base: ${basisPlain}`,
        })
        tdState.appendChild(basisSpan)
      } else {
        tdState.textContent = '—'
      }

      tr.append(tdVal, tdType, tdAlg, tdGeom, tdState)
      tbody.appendChild(tr)
    }

    for (const { val, mult: m } of extraNumericRows) {
      if (m.algebraica === null) hasNonRationalAviso = true
      const cells = formatNumericMultiplicityCells(m)

      const tr = document.createElement('tr')

      const tdVal = document.createElement('td')
      tdVal.className = 'la-table-mono'
      tdVal.textContent = `≈ ${formatNum(val)}`

      const tdType = document.createElement('td')
      const pill = document.createElement('span')
      pill.className = 'la-table-pill la-table-pill--approx'
      pill.textContent = 'Numérico'
      tdType.appendChild(pill)

      const tdAlg = document.createElement('td')
      tdAlg.className = 'la-table-mono'
      tdAlg.textContent = cells.algText

      const tdGeom = document.createElement('td')
      tdGeom.className = 'la-table-mono'
      tdGeom.textContent = cells.geomText

      const tdState = document.createElement('td')
      tdState.textContent = m.aviso ?? 'Estimación numérica'

      tr.append(tdVal, tdType, tdAlg, tdGeom, tdState)
      tbody.appendChild(tr)
    }

    wrap.appendChild(table)
    body.appendChild(wrap)

    if (hasNonRationalAviso) {
      const note = document.createElement('p')
      note.className = 'la-note'
      note.textContent = NON_RATIONAL_MULT_NOTE
      body.appendChild(note)
    }
  }

  function renderPowerCard(
    res: PowerMethodResult,
    residual: number,
    shiftVal: number,
    useInverse: boolean,
    tol: number,
    maxIter: number,
  ): void {
    const modeLabel = useInverse
      ? `Potencia inversa (σ = ${formatNum(shiftVal)}, vía Gauss)`
      : shiftVal !== 0
        ? `Potencia con desplazamiento (σ = ${formatNum(shiftVal)})`
        : 'Potencia directa (valor propio dominante)'

    const body = createResultFrame(
      cardPower,
      'Método de la potencia (cociente de Rayleigh)',
      `${modeLabel} · ${res.iteraciones.length} iter.`,
    )

    if (res.convergio) {
      const badge = document.createElement('div')
      badge.className = 'la-sys-badge la-sys-badge--unica'
      badge.textContent = `Convergió en ${res.iteraciones.length} ${res.iteraciones.length === 1 ? 'iteración' : 'iteraciones'} (tol = ${tol.toExponential(0)})`
      body.appendChild(badge)

      body.appendChild(
        createMathBox(
          `\\lambda \\approx ${formatNumLatex(res.lambda)}, \\qquad \\mathbf{v} \\approx ${numVectorToLatex(res.vector)}`,
          `λ ≈ ${formatNum(res.lambda)}, v ≈ ${numVectorToPlain(res.vector)}`,
        ),
      )
    } else {
      const err = document.createElement('div')
      err.className = 'la-error-banner'
      err.setAttribute('role', 'alert')
      err.textContent =
        res.motivo ??
        `El método de la potencia no convergió en ${maxIter} iteraciones.`
      body.appendChild(err)

      const unconvWrap = document.createElement('div')
      unconvWrap.className = 'la-iter-ref-box'
      const label = document.createElement('p')
      label.className = 'la-note'
      label.textContent =
        'Último iterado, sin convergencia (solo referencia; no es un valor propio verificado):'
      unconvWrap.appendChild(label)
      const iterIdx = res.iteraciones.length
      unconvWrap.appendChild(
        createMathBox(
          `\\lambda^{(${iterIdx})} = ${formatNumLatex(res.lambda)}, \\qquad \\mathbf{v}^{(${iterIdx})} = ${numVectorToLatex(res.vector)}`,
          `Último iterado (sin convergencia): λ^(${iterIdx}) = ${formatNum(res.lambda)}, v^(${iterIdx}) = ${numVectorToPlain(res.vector)}`,
          true,
        ),
      )
      body.appendChild(unconvWrap)
    }

    const strip = document.createElement('div')
    strip.className = 'la-residual-strip'
    strip.innerHTML = `
      <span>Cociente de Rayleigh: <code>${formatNum(res.lambda)}</code></span>
      <span>Residuo ‖Av − λv‖₂: <code>${formatResidual(residual)}</code></span>
      <span>Iteraciones: <code>${res.iteraciones.length} / ${maxIter}</code></span>
    `
    body.appendChild(strip)

    if (res.iteraciones.length > 0) {
      const tableSec = document.createElement('div')
      tableSec.className = 'la-iter-table-section'

      const subh = document.createElement('h4')
      subh.className = 'la-subheading'
      subh.textContent = 'Tabla de iteraciones (cociente de Rayleigh)'
      tableSec.appendChild(subh)

      const tableWrap = document.createElement('div')
      tableWrap.className = 'la-table-wrap la-iter-table-wrap'
      const table = document.createElement('table')
      table.className = 'la-table'
      table.innerHTML = `
        <thead>
          <tr>
            <th scope="col">k</th>
            <th scope="col">λ_k (Rayleigh)</th>
            <th scope="col">|λ_k − λ_{k−1}|</th>
          </tr>
        </thead>
        <tbody></tbody>
      `
      const tbody = table.querySelector('tbody')!

      const appendRows = (from: number, to: number): void => {
        const end = Math.min(to, res.iteraciones.length)
        for (let i = from; i < end; i++) {
          const step = res.iteraciones[i]
          const prev = i > 0 ? res.iteraciones[i - 1].lambda : null
          const diff = prev !== null ? Math.abs(step.lambda - prev) : null
          const tr = document.createElement('tr')

          const tdK = document.createElement('td')
          tdK.className = 'la-table-mono'
          tdK.textContent = String(step.k)

          const tdLam = document.createElement('td')
          tdLam.className = 'la-table-mono'
          tdLam.textContent = formatNum(step.lambda)

          const tdDiff = document.createElement('td')
          tdDiff.className = 'la-table-mono'
          tdDiff.textContent = diff === null ? '—' : formatResidual(diff)

          tr.append(tdK, tdLam, tdDiff)
          tbody.appendChild(tr)
        }
      }

      const initial = Math.min(res.iteraciones.length, INITIAL_POWER_ROWS)
      appendRows(0, initial)
      tableWrap.appendChild(table)
      tableSec.appendChild(tableWrap)

      if (res.iteraciones.length > initial) {
        const remaining = res.iteraciones.length - initial
        const moreBar = document.createElement('div')
        moreBar.className = 'la-steps-more'
        const moreBtn = document.createElement('button')
        moreBtn.type = 'button'
        moreBtn.className = 'la-btn la-btn--sm'
        moreBtn.textContent = `Mostrar las ${remaining} iteraciones restantes`
        moreBtn.addEventListener('click', () => {
          appendRows(initial, res.iteraciones.length)
          moreBar.remove()
        })
        moreBar.appendChild(moreBtn)
        tableSec.appendChild(moreBar)
      }

      body.appendChild(tableSec)
    }
  }

  function renderQrCard(res: QrIterationResult, tol: number, maxIter: number): void {
    const body = createResultFrame(
      cardQr,
      'Iteración QR (Householder sin desplazamiento)',
      `${res.iteraciones} ${res.iteraciones === 1 ? 'iteración' : 'iteraciones'}`,
    )

    const sortedVals = res.valores.slice().sort((x, y) => y - x)

    if (res.convergio) {
      const badge = document.createElement('div')
      badge.className = 'la-sys-badge la-sys-badge--unica'
      badge.textContent =
        res.iteraciones === 0
          ? 'Matriz ya triangular superior: valores propios leídos directamente de la diagonal (0 iteraciones)'
          : `Convergió en ${res.iteraciones} ${res.iteraciones === 1 ? 'iteración' : 'iteraciones'} (elementos bajo la diagonal < ${tol.toExponential(0)})`
      body.appendChild(badge)

      const valsTex = sortedVals
        .map((v, i) => `\\lambda_${i + 1} \\approx ${formatNumLatex(v)}`)
        .join(', \\quad ')
      const valsPlain = sortedVals.map((v, i) => `λ${i + 1} ≈ ${formatNum(v)}`).join(', ')
      body.appendChild(createMathBox(valsTex, valsPlain, true))
    } else {
      const err = document.createElement('div')
      err.className = 'la-error-banner'
      err.setAttribute('role', 'alert')
      err.textContent =
        res.motivo ??
        `La iteración QR no convergió en ${maxIter} iteraciones.`
      body.appendChild(err)

      const unconvWrap = document.createElement('div')
      unconvWrap.className = 'la-iter-ref-box'
      const label = document.createElement('p')
      label.className = 'la-note'
      label.textContent =
        'Último iterado, sin convergencia (diagonal actual de A_k; no representa valores propios convergidos):'
      unconvWrap.appendChild(label)
      const diagTex = sortedVals
        .map((v, i) => `d_${i + 1} = ${formatNumLatex(v)}`)
        .join(', \\quad ')
      const diagPlain = sortedVals.map((v, i) => `d${i + 1} = ${formatNum(v)}`).join(', ')
      unconvWrap.appendChild(
        createMathBox(diagTex, `Último iterado (sin convergencia): ${diagPlain}`, true),
      )
      body.appendChild(unconvWrap)
    }

    const strip = document.createElement('div')
    strip.className = 'la-residual-strip'
    strip.innerHTML = `
      <span>Estado: <code>${res.convergio ? 'Convergió' : 'No convergió'}</code></span>
      <span>Iteraciones QR: <code>${res.iteraciones} / ${maxIter}</code></span>
      <span>Tolerancia subdiagonal: <code>${tol.toExponential(0)}</code></span>
    `
    body.appendChild(strip)
  }

  function renderJacobiCard(
    a: Matrix,
    sym: { ok: true } | { ok: false; i: number; j: number },
    res: JacobiEigenResult | null,
  ): void {
    if (!sym.ok || !res) {
      const body = createResultFrame(
        cardJacobi,
        'Diagonalización ortogonal de Jacobi (matrices simétricas)',
        'No aplicable · A ≠ Aᵀ',
      )
      const badge = document.createElement('div')
      badge.className = 'la-sys-badge la-sys-badge--approx'
      badge.textContent = 'Matriz no simétrica: Jacobi solo aplica cuando A = Aᵀ'
      body.appendChild(badge)

      if (!sym.ok) {
        const note = document.createElement('p')
        note.className = 'la-note'
        note.textContent = `Se encontró a_${sym.i + 1},${sym.j + 1} = ${ratToString(a[sym.i][sym.j])} ≠ a_${sym.j + 1},${sym.i + 1} = ${ratToString(a[sym.j][sym.i])}. La diagonalización ortogonal A = Q D Qᵀ está garantizada por el teorema espectral únicamente para matrices simétricas.`
        body.appendChild(note)
      }
      return
    }

    const body = createResultFrame(
      cardJacobi,
      'Diagonalización ortogonal de Jacobi (A = Q D Qᵀ)',
      `Matriz simétrica · ${res.iteraciones} ${res.iteraciones === 1 ? 'rotación' : 'rotaciones'}`,
    )

    const badgeInfo = describeJacobiBadge(res)
    const badge = document.createElement('div')
    badge.className = badgeInfo.ok
      ? 'la-sys-badge la-sys-badge--unica'
      : 'la-sys-badge la-sys-badge--approx'
    badge.textContent = badgeInfo.text
    body.appendChild(badge)

    const D = diagMatrixNum(res.valores)
    const qTex = numMatrixToLatex(res.vectores)
    const dTex = numMatrixToLatex(D)

    body.appendChild(
      createMathBox(
        `A \\approx Q \\, D \\, Q^{\\mathsf{T}}, \\qquad \\lambda = (${res.valores.map(formatNumLatex).join(',\\; ')})`,
        `Valores propios (Jacobi): ${res.valores.map(formatNum).join(', ')}`,
        true,
      ),
    )

    const factorsGrid = document.createElement('div')
    factorsGrid.className = 'la-factors-grid'
    factorsGrid.appendChild(
      createMathBox(`Q = ${qTex}`, `Q =\n${numMatrixToString(res.vectores)}`, true),
    )
    factorsGrid.appendChild(
      createMathBox(`D = ${dTex}`, `D =\n${numMatrixToString(D)}`, true),
    )
    body.appendChild(factorsGrid)

    const strip = document.createElement('div')
    strip.className = 'la-residual-strip'
    strip.innerHTML = `
      <span>Residuo reconstrucción ‖A − QDQᵀ‖_F: <code>${formatResidual(res.residuo)}</code></span>
      <span>Ortogonalidad ‖QᵀQ − I‖_F: <code>${formatResidual(res.residuoOrtogonalidad)}</code></span>
      <span>Rotaciones de Jacobi: <code>${res.iteraciones}</code></span>
    `
    body.appendChild(strip)

    const latexCode = [
      LATEX_HEADER,
      '% Diagonalización ortogonal A = Q D Q^T (Jacobi)',
      `\\[ Q = ${qTex} \\]`,
      `\\[ D = ${dTex} \\]`,
    ].join('\n')
    body.appendChild(
      buildLatexCopyBlock('eigen-jacobi', 'Diagonalización ortogonal Q y D', latexCode, track),
    )
  }

  function runSolve(): void {
    const readA = gridA.readMatrix()
    if (!readA.ok) {
      showBannerError(inputErrorEl, readA.error)
      readA.firstInvalid.focus()
      return
    }

    const a = readA.matriz
    if (!isSquare(a)) {
      const msg = `La matriz A es ${a.length}×${a[0]?.length ?? 0} y no es cuadrada. Los valores y vectores propios solo están definidos para matrices cuadradas n×n.`
      showBannerError(inputErrorEl, msg)
      emptyHintEl.hidden = true
      resultsContainer.hidden = false
      clearResultsStale()
      for (const card of allCards) card.hidden = true
      setResultError(cardPoly, 'Error: matriz no cuadrada', msg)
      return
    }

    const readVec = readX0()
    if (!readVec.ok) {
      showBannerError(inputErrorEl, readVec.error)
      readVec.firstInvalid.focus()
      return
    }

    const shiftRaw = shiftInput.value.trim()
    const shiftVal = shiftRaw === '' ? 0 : parseNumericField(shiftRaw)
    if (shiftVal === null) {
      showBannerError(inputErrorEl, 'El desplazamiento σ debe ser un número real finito (o 0).')
      shiftInput.focus()
      return
    }

    const tolVal = parseNumericField(tolInput.value)
    if (tolVal === null || tolVal <= 0) {
      showBannerError(inputErrorEl, 'La tolerancia debe ser un número positivo (por ejemplo, 1e-8).')
      tolInput.focus()
      return
    }

    const maxIterVal = Number(maxIterInput.value)
    if (!Number.isInteger(maxIterVal) || maxIterVal < 1 || maxIterVal > 5000) {
      showBannerError(
        inputErrorEl,
        'El máximo de iteraciones debe ser un entero entre 1 y 5000.',
      )
      maxIterInput.focus()
      return
    }

    const n = a.length
    const aNum = a.map((row) => row.map((r) => r.toNumber()))
    const useInverse = inverseCheck.checked

    // Fase 1: calcular todo el análisis espectral antes de tocar el DOM
    let computed: {
      coefs: Rat[]
      roots: Rat[]
      multByRoot: Map<string, MultiplicidadExacta>
      eigenvectorsByRoot: Map<string, Rat[][]>
      defective: boolean
      powerRes: PowerMethodResult
      powerResidual: number
      qrRes: QrIterationResult
      sym: { ok: true } | { ok: false; i: number; j: number }
      jacobiRes: JacobiEigenResult | null
      extraNumericRows: Array<{ val: number; mult: MultiplicidadResult }>
    }

    try {
      const coefs = charPoly(a)
      const roots = rationalRoots(coefs)
      const multByRoot = new Map<string, MultiplicidadExacta>()
      const eigenvectorsByRoot = new Map<string, Rat[][]>()
      for (const r of roots) {
        const key = ratToString(r)
        multByRoot.set(key, multiplicidades(a, r))
        eigenvectorsByRoot.set(key, exactEigenvectors(a, r))
      }
      const defective = esDefectiva(a)

      const powerRes = powerMethod(aNum, readVec.vector, tolVal, maxIterVal, shiftVal, {
        inverse: useInverse,
      })
      const powerResidual = eigenResidual(aNum, powerRes.vector, powerRes.lambda)

      const qrRes = qrIteration(aNum, tolVal, maxIterVal)

      const sym = isSymmetricExact(a)
      const jacobiRes = sym.ok ? jacobiEigen(aNum) : null

      // Recopilar y deduplicar candidatos numéricos para la tabla de multiplicidades
      const numericCandidates: number[] = []
      if (jacobiRes) {
        numericCandidates.push(...jacobiRes.valores)
      } else if (qrRes.convergio) {
        numericCandidates.push(...qrRes.valores)
      } else if (powerRes.convergio) {
        numericCandidates.push(powerRes.lambda)
      }

      const extraNumeric: number[] = []
      for (const val of numericCandidates) {
        if (!Number.isFinite(val)) continue
        const tolMatch = 1e-6 * Math.max(1, Math.abs(val))
        const matchesRat = roots.some((r) => Math.abs(r.toNumber() - val) <= tolMatch)
        if (matchesRat) continue
        const dup = extraNumeric.some((prev) => Math.abs(prev - val) <= tolMatch)
        if (!dup) extraNumeric.push(val)
      }
      extraNumeric.sort((x, y) => y - x)

      const extraNumericRows = extraNumeric.map((val) => ({
        val,
        mult: multiplicidades(a, val),
      }))

      computed = {
        coefs,
        roots,
        multByRoot,
        eigenvectorsByRoot,
        defective,
        powerRes,
        powerResidual,
        qrRes,
        sym,
        jacobiRes,
        extraNumericRows,
      }
    } catch (err) {
      showBannerError(inputErrorEl, formatEigenErrorMessage(err))
      if (!resultsContainer.hidden) {
        markResultsStale()
      }
      return
    }

    // Fase 2: todo el cálculo terminó correctamente; limpiar avisos obsoletos y pintar las 5 tarjetas
    showBannerError(inputErrorEl, null)
    emptyHintEl.hidden = true
    resultsContainer.hidden = false
    clearResultsStale()

    renderPolyCard(a, computed.coefs, computed.roots, computed.multByRoot)
    renderMultiplicitiesCard(
      n,
      computed.roots,
      computed.multByRoot,
      computed.eigenvectorsByRoot,
      computed.defective,
      computed.extraNumericRows,
    )
    renderPowerCard(
      computed.powerRes,
      computed.powerResidual,
      shiftVal,
      useInverse,
      tolVal,
      maxIterVal,
    )
    renderQrCard(computed.qrRes, tolVal, maxIterVal)
    renderJacobiCard(a, computed.sym, computed.jacobiRes)

    const powerMode = useInverse ? 'inverse' : shiftVal !== 0 ? 'shifted' : 'direct'
    track('linealalrl-eigen-solve', {
      size: `${n}x${n}`,
      symmetric: computed.jacobiRes ? 'yes' : 'no',
      defective: computed.defective ? 'yes' : 'no',
      powerMode,
    })
  }

  btnSolve.addEventListener('click', runSolve)
}
