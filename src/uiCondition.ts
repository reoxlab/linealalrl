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

// Interfaz interactiva del módulo 4: normas matriciales, número de condición κ(A),
// semáforo didáctico, cota de error de la solución, equilibrado de filas y matrices de Hilbert.

import {
  type CondNumberResult,
  type CondVerdict,
  type CondVerdictLevel,
  type MatrixNorm,
  condNumber,
  equilibrarFilas,
  estimaErrorSolucion,
  hilbert,
  normMatrix,
  veredictoCondicion,
} from './condition.ts'
import { renderLatex } from './katexRender.ts'
import { type Matrix, det, inverse, isSquare, matrixToLatex, rank } from './matrix.ts'
import { parsePastedMatrix } from './paste.ts'
import { parseRat, ratToDecimal, ratToString } from './rational.ts'
import { MAX_GRID_DIM } from './share.ts'
import {
  type TrackFn,
  buildLatexCopyBlock,
  createMatrixGrid,
  createResultFrame,
  setResultError,
} from './ui.ts'

const LATEX_HEADER = '% Requiere \\usepackage{amsmath}'

export interface ConditionPreset {
  id: string
  nombre: string
  descripcion: string
  a: string[][]
}

function hilbertCellStrings(n: number): string[][] {
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i + j === 0 ? '1' : `1/${i + j + 1}`)),
  )
}

export const CONDITION_PRESETS: readonly ConditionPreset[] = [
  {
    id: 'identidad-3x3',
    nombre: 'Identidad 3×3 (κ = 1)',
    descripcion:
      'Matriz identidad I₃: referencia ideal perfectamente condicionada (κ₁ = κ₂ = κ∞ = 1, sin amplificación de error).',
    a: [
      ['1', '0', '0'],
      ['0', '1', '0'],
      ['0', '0', '1'],
    ],
  },
  {
    id: 'bien-cond-3x3',
    nombre: 'Bien condicionada 3×3',
    descripcion:
      'Matriz simétrica estrictamente diagonalmente dominante (κ₂ ≈ 2.09): estable numéricamente, con menos de 1 dígito de pérdida.',
    a: [
      ['4', '1', '0'],
      ['1', '4', '1'],
      ['0', '1', '4'],
    ],
  },
  {
    id: 'hilbert-3x3',
    nombre: 'Hilbert 3×3 (κ₂ ≈ 524)',
    descripcion:
      'Matriz de Hilbert H₃ con entradas 1/(i+j+1): condicionamiento moderado (κ₂ ≈ 524, ~2.7 dígitos de posible pérdida).',
    a: hilbertCellStrings(3),
  },
  {
    id: 'hilbert-4x4',
    nombre: 'Hilbert 4×4 (κ₂ ≈ 15514)',
    descripcion:
      'Matriz de Hilbert H₄: mal condicionada (κ₂ ≈ 15514, ~4.2 dígitos decimales que pueden perderse al resolver Ax = b).',
    a: hilbertCellStrings(4),
  },
  {
    id: 'mal-escalada-3x3',
    nombre: 'Mal escalada 3×3',
    descripcion:
      'Filas con órdenes de magnitud muy distintos (10³ frente a 10⁻³): κ₂ supera 10⁶, pero «Equilibrar filas» lo reduce drásticamente.',
    a: [
      ['1000', '1000', '0'],
      ['0', '1', '1'],
      ['0', '0', '1/1000'],
    ],
  },
]

const SMALL_DET_WELL_COND_CELLS: string[][] = [
  ['0.01', '0', '0'],
  ['0', '0.01', '0'],
  ['0', '0', '0.01'],
]

const NORM_META: Record<
  MatrixNorm,
  {
    label: string
    symbolPlain: string
    symbolTex: string
    defTex: string
    defPlain: string
    kappaTex: string
  }
> = {
  1: {
    label: 'Norma 1 (columnas)',
    symbolPlain: '‖A‖₁',
    symbolTex: '\\|A\\|_1',
    defTex: '\\|A\\|_1 = \\max_{1 \\le j \\le n} \\sum_{i=1}^{m} |a_{ij}|',
    defPlain: 'Máxima suma de valores absolutos por columna',
    kappaTex: '\\kappa_1(A)',
  },
  2: {
    label: 'Norma 2 (espectral)',
    symbolPlain: '‖A‖₂',
    symbolTex: '\\|A\\|_2',
    defTex: '\\|A\\|_2 = \\sigma_{\\max}(A) = \\sqrt{\\lambda_{\\max}(A^{\\mathsf{T}}A)}',
    defPlain: 'Mayor valor singular (potencia sobre AᵀA)',
    kappaTex: '\\kappa_2(A)',
  },
  inf: {
    label: 'Norma ∞ (filas)',
    symbolPlain: '‖A‖∞',
    symbolTex: '\\|A\\|_{\\infty}',
    defTex: '\\|A\\|_{\\infty} = \\max_{1 \\le i \\le m} \\sum_{j=1}^{n} |a_{ij}|',
    defPlain: 'Máxima suma de valores absolutos por fila',
    kappaTex: '\\kappa_{\\infty}(A)',
  },
  fro: {
    label: 'Norma de Frobenius',
    symbolPlain: '‖A‖_F',
    symbolTex: '\\|A\\|_F',
    defTex: '\\|A\\|_F = \\sqrt{\\sum_{i=1}^{m} \\sum_{j=1}^{n} a_{ij}^2}',
    defPlain: 'Raíz de la suma de todos los cuadrados',
    kappaTex: '\\kappa_F(A)',
  },
}

const ALL_NORMS: readonly MatrixNorm[] = [1, 2, 'inf', 'fro']

export function formatCondNum(v: number): string {
  if (!Number.isFinite(v)) {
    if (Number.isNaN(v)) return 'NaN'
    return v < 0 ? '−∞' : '∞'
  }
  if (Math.abs(v) < 1e-15) return '0'
  const rounded = Math.round(v)
  if (Math.abs(v - rounded) < 1e-9 && Math.abs(v) < 1e7) {
    return String(rounded)
  }
  if (Math.abs(v) >= 1e6 || Math.abs(v) < 1e-4) {
    return v.toExponential(4)
  }
  return Number(v.toPrecision(6)).toString()
}

export function formatCondNumLatex(v: number): string {
  if (!Number.isFinite(v)) {
    if (Number.isNaN(v)) return '\\text{NaN}'
    return v < 0 ? '-\\infty' : '\\infty'
  }
  const s = formatCondNum(v)
  const match = /^([+-]?\d+(?:\.\d+)?)[eE]([+-]?\d+)$/.exec(s)
  if (!match) return s
  const mantissa = match[1].replace(/(\.\d*?[1-9])0+$/, '$1').replace(/\.0+$/, '')
  const exp = Number(match[2])
  return `${mantissa} \\times 10^{${exp}}`
}

export function verdictBadgeLabel(nivel: CondVerdictLevel): string {
  if (nivel === 'bien') return 'Bien condicionada (κ < 100)'
  if (nivel === 'moderada') return 'Condicionamiento moderado (100 ≤ κ < 10⁴)'
  return 'Mal condicionada (κ ≥ 10⁴)'
}

function numMatrixToLatex(m: number[][]): string {
  const filas = m.map((row) => row.map(formatCondNumLatex).join(' & ')).join(' \\\\ ')
  return `\\begin{bmatrix} ${filas} \\end{bmatrix}`
}

function numMatrixToPlain(m: number[][]): string {
  return m.map((row) => row.map(formatCondNum).join(' ')).join('\n')
}

function createMathBox(tex: string, fallbackText: string, compact = false): HTMLDivElement {
  const box = document.createElement('div')
  box.className = compact ? 'la-result-math la-result-math--compact' : 'la-result-math'
  void renderLatex(tex, box, { displayMode: true, fallbackText })
  return box
}

function parseNormValue(raw: string): MatrixNorm {
  if (raw === '1') return 1
  if (raw === 'inf') return 'inf'
  if (raw === 'fro') return 'fro'
  return 2
}

function parseNonNegativeNum(raw: string): number | null {
  const limpio = raw.trim().replace(/−/g, '-').replace(/,/g, '.')
  if (!limpio) return null
  const r = parseRat(limpio)
  if (r !== null) {
    const val = r.toNumber()
    return Number.isFinite(val) && val >= 0 ? val : null
  }
  const n = Number(limpio)
  return Number.isFinite(n) && n >= 0 ? n : null
}

function showBannerError(el: HTMLElement, message: string | null): void {
  if (!message) {
    el.hidden = true
    el.textContent = ''
    return
  }
  el.hidden = false
  el.textContent = message
}

function formatErrorInsight(bound: number, kappa: number): string {
  if (!Number.isFinite(kappa) || !Number.isFinite(bound)) {
    return 'La matriz es singular (κ = ∞): no existe una cota finita para el error relativo de la solución.'
  }
  if (bound === 0) {
    return 'Con error relativo 0 en b, esta cota no añade amplificación por perturbación en el lado derecho (aunque el redondeo interno sigue sujeto a κ).'
  }
  const pct = bound * 100
  if (bound >= 1) {
    return `Cota ≥ 100 % (${formatCondNum(pct)} %): en el peor caso el error relativo puede superar el tamaño de la propia solución y no quedar ningún dígito confiable.`
  }
  const reliableDigits = Math.max(0, -Math.log10(bound))
  return `El error relativo en x queda acotado por ~${formatCondNum(pct)} % (al menos ~${reliableDigits.toFixed(1)} dígitos decimales confiables frente a esta perturbación en b).`
}

interface ComputedConditionState {
  a: Matrix
  aNum: number[][]
  norms: Record<MatrixNorm, number>
  conds: Record<MatrixNorm, CondNumberResult>
  exactRank: number
  exactSingular: boolean
  exactAtasco: string | null
  exactDetText: string | null
  equilibrated: {
    matriz: number[][]
    escalas: number[]
    condsAfter: Record<MatrixNorm, CondNumberResult>
  } | null
}

/**
 * Monta la interfaz de la pestaña «Condicionamiento» dentro de `app`.
 */
export function mountConditionUI(app: HTMLElement, track: TrackFn): void {
  const defaultPreset = CONDITION_PRESETS[2] // Hilbert 3x3: educativo al instante
  let activePresetId: string | null = defaultPreset.id
  let lastComputed: ComputedConditionState | null = null
  let showEquilibration = false

  app.innerHTML = `
    <div class="la-workspace">
      <section class="la-panel">
        <div class="la-panel-head">
          <h2 class="la-panel-title">Matriz A (normas y condicionamiento)</h2>
          <span class="la-panel-meta" id="cond-dim-meta"></span>
        </div>

        <div class="la-toolbar" id="cond-presets-bar">
          <span class="la-toolbar-label">Presets:</span>
        </div>
        <p class="la-preset-desc" id="cond-preset-desc"></p>

        <div class="la-panel-body la-matrix-a-layout">
          <div class="la-matrix-a-editor">
            <span class="la-field-label">Matriz cuadrada A (hasta 6×6)</span>
            <div id="cond-grid-a-mount"></div>
          </div>

          <div class="la-paste-box">
            <label class="la-field-label" for="cond-paste-input">Pegar matriz A</label>
            <p class="la-hint" id="cond-paste-hint">Pega una matriz cuadrada (espacios, comas o tabuladores, fracciones como 1/3, hasta 6×6).</p>
            <textarea
              id="cond-paste-input"
              class="la-textarea"
              rows="4"
              spellcheck="false"
              aria-describedby="cond-paste-hint"
              placeholder="1 1/2 1/3&#10;1/2 1/3 1/4&#10;1/3 1/4 1/5"
            ></textarea>
            <div class="la-paste-actions">
              <button class="la-btn la-btn--sm" id="btn-cond-paste" type="button">Pegar matriz</button>
              <span class="la-feedback" id="cond-paste-feedback" role="status" aria-live="polite" hidden></span>
            </div>
            <div class="la-error-banner" id="cond-paste-error" role="alert" hidden></div>
          </div>
        </div>

        <div class="la-error-banner la-panel-error" id="cond-input-error" role="alert" hidden></div>
      </section>

      <section class="la-panel">
        <div class="la-panel-head">
          <h2 class="la-panel-title">Análisis de condicionamiento</h2>
          <span class="la-panel-meta">Normas 1, 2, ∞ y Frobenius · κ(A) = ‖A‖·‖A⁻¹‖ · Semáforo y escalado</span>
        </div>

        <div class="la-toolbar la-iter-controls-bar">
          <label class="la-direct-method-label" for="cond-norm-select">
            <span>Norma para κ(A):</span>
            <select id="cond-norm-select" class="la-select la-direct-method-select" aria-label="Norma matricial para el número de condición">
              <option value="2" selected>Norma 2 (espectral · mayor valor singular)</option>
              <option value="1">Norma 1 (máxima suma por columna)</option>
              <option value="inf">Norma ∞ (máxima suma por fila)</option>
              <option value="fro">Norma de Frobenius (raíz de suma de cuadrados)</option>
            </select>
          </label>

          <button class="la-btn la-btn--primary" id="btn-cond-solve" type="button">Calcular condicionamiento</button>
          <button class="la-btn" id="btn-cond-equilibrate" type="button">Equilibrar filas</button>
        </div>

        <div class="la-panel-body la-results-stack">
          <p class="la-note" id="cond-empty-hint">
            Pulsa <strong>Calcular condicionamiento</strong> para obtener las cuatro normas de A, el número de condición κ(A) con su veredicto en palabras, la estimación de error de la solución y la comparación antes y después de equilibrar filas.
          </p>
          <div class="la-results-stack" id="cond-results-container" hidden>
            <article class="la-result-card" id="cond-card-norms"></article>
            <article class="la-result-card" id="cond-card-kappa"></article>
            <article class="la-result-card" id="cond-card-error-est"></article>
            <article class="la-result-card" id="cond-card-equilibrate" hidden></article>
          </div>
        </div>
      </section>

      <section class="la-panel">
        <div class="la-panel-head">
          <h2 class="la-panel-title">Avisos didácticos clave</h2>
          <span class="la-panel-meta">Determinante vs. condicionamiento · Crecimiento de las matrices de Hilbert</span>
        </div>

        <div class="la-panel-body la-cond-didactic-grid">
          <article class="la-cond-callout" id="cond-didactic-det"></article>
          <article class="la-cond-callout" id="cond-didactic-hilbert"></article>
        </div>
      </section>
    </div>
  `

  const presetsBar = app.querySelector<HTMLDivElement>('#cond-presets-bar')!
  const presetDescEl = app.querySelector<HTMLParagraphElement>('#cond-preset-desc')!
  const dimMetaEl = app.querySelector<HTMLSpanElement>('#cond-dim-meta')!
  const gridAMount = app.querySelector<HTMLDivElement>('#cond-grid-a-mount')!
  const pasteInput = app.querySelector<HTMLTextAreaElement>('#cond-paste-input')!
  const btnPaste = app.querySelector<HTMLButtonElement>('#btn-cond-paste')!
  const pasteFeedbackEl = app.querySelector<HTMLSpanElement>('#cond-paste-feedback')!
  const pasteErrorEl = app.querySelector<HTMLDivElement>('#cond-paste-error')!
  const inputErrorEl = app.querySelector<HTMLDivElement>('#cond-input-error')!

  const normSelect = app.querySelector<HTMLSelectElement>('#cond-norm-select')!
  const btnSolve = app.querySelector<HTMLButtonElement>('#btn-cond-solve')!
  const btnEquilibrate = app.querySelector<HTMLButtonElement>('#btn-cond-equilibrate')!
  const emptyHintEl = app.querySelector<HTMLParagraphElement>('#cond-empty-hint')!
  const resultsContainer = app.querySelector<HTMLDivElement>('#cond-results-container')!

  const cardNorms = app.querySelector<HTMLElement>('#cond-card-norms')!
  const cardKappa = app.querySelector<HTMLElement>('#cond-card-kappa')!
  const cardErrorEst = app.querySelector<HTMLElement>('#cond-card-error-est')!
  const cardEquilibrate = app.querySelector<HTMLElement>('#cond-card-equilibrate')!
  const didacticDetEl = app.querySelector<HTMLElement>('#cond-didactic-det')!
  const didacticHilbertEl = app.querySelector<HTMLElement>('#cond-didactic-hilbert')!

  const allCards = [cardNorms, cardKappa, cardErrorEst, cardEquilibrate]

  let userErrBText = '1e-6'

  function markResultsStale(): void {
    for (const card of allCards) {
      if (!card.hidden) {
        card.classList.add('is-stale')
        if (!card.querySelector('.la-stale-banner')) {
          const banner = document.createElement('div')
          banner.className = 'la-stale-banner'
          banner.textContent =
            'Matriz modificada: pulsa «Calcular condicionamiento» para actualizar.'
          card.prepend(banner)
        }
      }
    }
  }

  function clearResultsStale(): void {
    for (const card of allCards) {
      card.classList.remove('is-stale')
      card.querySelector('.la-stale-banner')?.remove()
    }
  }

  const presetButtons = new Map<string, HTMLButtonElement>()
  function updatePresetUI(preset: ConditionPreset | null): void {
    activePresetId = preset ? preset.id : null
    for (const [id, btn] of presetButtons) {
      btn.classList.toggle('la-btn--active', id === activePresetId)
    }
    presetDescEl.textContent = preset
      ? preset.descripcion
      : 'Matriz personalizada: edita las celdas o pega una matriz cuadrada para estudiar su condicionamiento.'
  }

  function updateDimMeta(): void {
    const [r, c] = gridA.getSize()
    dimMetaEl.textContent = r === c ? `A: ${r}×${c} (cuadrada)` : `A: ${r}×${c} (no cuadrada)`
  }

  const gridA = createMatrixGrid({
    idPrefix: 'cond-a',
    matrixName: 'A',
    initialValues: defaultPreset.a,
    onSizeChange: () => {
      updateDimMeta()
    },
    onInputChange: () => {
      updatePresetUI(null)
      showBannerError(inputErrorEl, null)
      if (!resultsContainer.hidden) {
        markResultsStale()
      }
    },
  })
  gridAMount.appendChild(gridA.element)
  updateDimMeta()

  function loadMatrixCells(cells: string[][], preset: ConditionPreset | null, customDesc?: string): void {
    gridA.setValues(cells)
    updateDimMeta()
    if (preset) {
      updatePresetUI(preset)
    } else {
      updatePresetUI(null)
      if (customDesc) presetDescEl.textContent = customDesc
    }
    showBannerError(inputErrorEl, null)
    showBannerError(pasteErrorEl, null)
    if (!resultsContainer.hidden) {
      markResultsStale()
    }
  }

  for (const preset of CONDITION_PRESETS) {
    const btn = document.createElement('button')
    btn.type = 'button'
    btn.className = 'la-btn la-btn--sm'
    btn.textContent = preset.nombre
    btn.addEventListener('click', () => {
      loadMatrixCells(preset.a, preset)
    })
    presetsBar.appendChild(btn)
    presetButtons.set(preset.id, btn)
  }
  updatePresetUI(defaultPreset)

  let pasteFeedbackTimer: number | null = null
  btnPaste.addEventListener('click', () => {
    const parsed = parsePastedMatrix(pasteInput.value, MAX_GRID_DIM)
    if (!parsed.ok) {
      showBannerError(pasteErrorEl, parsed.error)
      pasteFeedbackEl.hidden = true
      return
    }
    showBannerError(pasteErrorEl, null)
    loadMatrixCells(
      parsed.matriz.map((row) => row.map(ratToString)),
      null,
    )
    pasteFeedbackEl.hidden = false
    pasteFeedbackEl.textContent = `Matriz ${parsed.filas}×${parsed.cols} pegada en A.`
    if (pasteFeedbackTimer !== null) window.clearTimeout(pasteFeedbackTimer)
    pasteFeedbackTimer = window.setTimeout(() => {
      pasteFeedbackEl.hidden = true
    }, 3000)
  })

  // -------------------------------------------------------------------------
  // Renderizado de tarjetas de resultados
  // -------------------------------------------------------------------------

  function renderNormsCard(state: ComputedConditionState, selectedNorm: MatrixNorm): void {
    const n = state.a.length
    const body = createResultFrame(
      cardNorms,
      'Normas matriciales de A',
      `Matriz ${n}×${n} · Norma activa: ${NORM_META[selectedNorm].symbolPlain}`,
    )

    const wrap = document.createElement('div')
    wrap.className = 'la-table-wrap la-cond-table-wrap'
    wrap.tabIndex = 0
    wrap.setAttribute('role', 'region')
    wrap.setAttribute('aria-labelledby', 'cond-norms-caption')

    const table = document.createElement('table')
    table.className = 'la-table la-cond-table'
    table.style.setProperty('--label-1', '"Norma"')
    table.style.setProperty('--label-2', '"Definición"')
    table.style.setProperty('--label-3', '"Valor"')

    const caption = document.createElement('caption')
    caption.id = 'cond-norms-caption'
    caption.className = 'la-sr-only'
    caption.textContent = 'Tabla de normas matriciales 1, 2, infinito y Frobenius de la matriz A'
    table.appendChild(caption)

    const thead = document.createElement('thead')
    thead.innerHTML = `
      <tr>
        <th scope="col">Norma</th>
        <th scope="col">Definición</th>
        <th scope="col">Valor calculado</th>
      </tr>
    `
    table.appendChild(thead)

    const tbody = document.createElement('tbody')
    for (const p of ALL_NORMS) {
      const meta = NORM_META[p]
      const tr = document.createElement('tr')
      if (p === selectedNorm) tr.className = 'is-cond-selected'

      const th = document.createElement('th')
      th.scope = 'row'
      th.className = 'la-table-method'
      th.textContent = `${meta.symbolPlain} — ${meta.label}`

      const tdDef = document.createElement('td')
      const formulaEl = document.createElement('div')
      formulaEl.className = 'la-cond-def-formula'
      void renderLatex(meta.defTex, formulaEl, {
        displayMode: false,
        fallbackText: meta.defPlain,
      })
      const descEl = document.createElement('div')
      descEl.className = 'la-cond-def-plain'
      descEl.textContent = meta.defPlain
      tdDef.appendChild(formulaEl)
      tdDef.appendChild(descEl)

      const tdVal = document.createElement('td')
      tdVal.className = 'la-table-mono la-cond-num-cell'
      tdVal.textContent = formatCondNum(state.norms[p])

      tr.appendChild(th)
      tr.appendChild(tdDef)
      tr.appendChild(tdVal)
      tbody.appendChild(tr)
    }

    table.appendChild(tbody)
    wrap.appendChild(table)
    body.appendChild(wrap)

    const note = document.createElement('p')
    note.className = 'la-note'
    note.textContent =
      'Todas las normas matriciales son equivalentes en dimensión finita, pero la norma 2 captura exactamente la razón entre el mayor y el menor factor de estiramiento (σ_max / σ_min).'
    body.appendChild(note)
  }

  function buildVerdictBlock(verdict: CondVerdict, kappa: number): HTMLElement {
    const box = document.createElement('div')
    box.className = `la-cond-verdict la-cond-verdict--${verdict.nivel}`
    box.setAttribute('role', 'status')

    const header = document.createElement('div')
    header.className = 'la-cond-verdict-head'

    const dot = document.createElement('span')
    dot.className = `la-cond-light la-cond-light--${verdict.nivel}`
    dot.setAttribute('aria-hidden', 'true')

    const label = document.createElement('strong')
    label.className = 'la-cond-verdict-label'
    label.textContent = Number.isFinite(kappa)
      ? verdictBadgeLabel(verdict.nivel)
      : 'Singular (κ = ∞)'

    header.appendChild(dot)
    header.appendChild(label)

    const text = document.createElement('p')
    text.className = 'la-cond-verdict-text'
    text.textContent = verdict.texto

    box.appendChild(header)
    box.appendChild(text)
    return box
  }

  function renderKappaCard(state: ComputedConditionState, selectedNorm: MatrixNorm): void {
    const n = state.a.length
    const meta = NORM_META[selectedNorm]
    const cond = state.conds[selectedNorm]
    const effectiveKappa = state.exactSingular ? Number.POSITIVE_INFINITY : cond.kappa
    const effectiveInvNorm = state.exactSingular ? Number.POSITIVE_INFINITY : cond.inversaNorma
    const verdict = veredictoCondicion(effectiveKappa)

    const body = createResultFrame(
      cardKappa,
      `Número de condición ${meta.symbolPlain.replace('‖A‖', 'κ')}`,
      `${meta.label} · Rango exacto: ${state.exactRank}/${n}`,
    )

    body.appendChild(buildVerdictBlock(verdict, effectiveKappa))

    const normA = state.norms[selectedNorm]
    if (!Number.isFinite(effectiveKappa)) {
      body.appendChild(
        createMathBox(
          `${meta.kappaTex} = \\infty`,
          `κ(A) = ∞ (matriz singular)`,
          true,
        ),
      )

      const errBox = document.createElement('div')
      errBox.className = 'la-error-banner'
      errBox.setAttribute('role', 'alert')
      errBox.textContent = state.exactAtasco
        ? `Matriz singular (rango ${state.exactRank} < ${n}): ${state.exactAtasco}`
        : `La matriz A es singular o numéricamente singular (pivote nulo en eliminación gaussiana con pivoteo parcial), por lo que no existe A⁻¹ y κ(A) = ∞.`
      body.appendChild(errBox)
    } else {
      const kDigits = effectiveKappa > 1 ? Math.log10(effectiveKappa) : 0
      const tex = `${meta.kappaTex} = ${meta.symbolTex} \\cdot \\|A^{-1}\\|${selectedNorm === 'inf' ? '_{\\infty}' : selectedNorm === 'fro' ? '_F' : `_${selectedNorm}`} = ${formatCondNumLatex(normA)} \\cdot ${formatCondNumLatex(effectiveInvNorm)} \\approx ${formatCondNumLatex(effectiveKappa)}`
      const plain = `κ(A) = ${formatCondNum(normA)} · ${formatCondNum(effectiveInvNorm)} ≈ ${formatCondNum(effectiveKappa)}`
      body.appendChild(createMathBox(tex, plain, true))

      const strip = document.createElement('div')
      strip.className = 'la-residual-strip'
      const detChip = state.exactDetText
        ? `<span>det(A): <code>${state.exactDetText}</code></span>`
        : `<span>det(A): <code>— (&gt; 5×5)</code></span>`
      strip.innerHTML = `
        <span>${meta.symbolPlain}: <code>${formatCondNum(normA)}</code></span>
        <span>${meta.symbolPlain.replace('A', 'A⁻¹')}: <code>${formatCondNum(effectiveInvNorm)}</code></span>
        <span>Dígitos que pueden perderse (log₁₀ κ): <code>~${kDigits < 0.05 ? '0' : kDigits.toFixed(1)}</code></span>
        <span>Rango exacto: <code>${state.exactRank}/${n}</code></span>
        ${detChip}
      `
      body.appendChild(strip)

      if (selectedNorm === 'fro' && n > 1) {
        const froNote = document.createElement('p')
        froNote.className = 'la-note'
        froNote.textContent = `Nota sobre Frobenius: como ‖I_${n}‖_F = √${n}, la norma de Frobenius satisface κ_F(I_${n}) = ${n} en lugar de 1 (a diferencia de las normas inducidas 1, 2 e ∞).`
        body.appendChild(froNote)
      }
    }

    const latexCode = [
      LATEX_HEADER,
      `% Análisis de condicionamiento (${meta.label})`,
      `\\[ A = ${matrixToLatex(state.a)} \\]`,
      `\\[ \\|A\\|_1 = ${formatCondNumLatex(state.norms[1])}, \\quad \\|A\\|_2 \\approx ${formatCondNumLatex(state.norms[2])}, \\quad \\|A\\|_{\\infty} = ${formatCondNumLatex(state.norms.inf)}, \\quad \\|A\\|_F \\approx ${formatCondNumLatex(state.norms.fro)} \\]`,
      Number.isFinite(effectiveKappa)
        ? `\\[ ${meta.kappaTex} \\approx ${formatCondNumLatex(effectiveKappa)} \\]`
        : `\\[ ${meta.kappaTex} = \\infty \\quad (\\text{matriz singular}) \\]`,
    ].join('\n')
    body.appendChild(
      buildLatexCopyBlock('cond-kappa', 'Normas y número de condición', latexCode, track),
    )
  }

  function renderErrorEstimateCard(state: ComputedConditionState, selectedNorm: MatrixNorm): void {
    const meta = NORM_META[selectedNorm]
    const cond = state.conds[selectedNorm]
    const effectiveKappa = state.exactSingular ? Number.POSITIVE_INFINITY : cond.kappa

    const body = createResultFrame(
      cardErrorEst,
      'Estimación del error en la solución de Ax = b',
      `Cota ‖δx‖/‖x‖ ≤ κ · ‖δb‖/‖b‖ con ${meta.label}`,
    )

    const controlRow = document.createElement('div')
    controlRow.className = 'la-cond-error-controls'

    const fieldLabel = document.createElement('label')
    fieldLabel.className = 'la-iter-param-field'
    fieldLabel.htmlFor = 'cond-rel-error-b'

    const labelSpan = document.createElement('span')
    labelSpan.textContent = 'Error relativo en b (‖δb‖ / ‖b‖):'

    const input = document.createElement('input')
    input.id = 'cond-rel-error-b'
    input.type = 'text'
    input.className = 'la-input la-input--compact'
    input.value = userErrBText
    input.spellcheck = false
    input.setAttribute('aria-label', 'Error relativo del vector b')

    fieldLabel.appendChild(labelSpan)
    fieldLabel.appendChild(input)
    controlRow.appendChild(fieldLabel)

    const quickBtns = ['1e-12', '1e-8', '1e-6', '1e-3']
    const quickWrap = document.createElement('div')
    quickWrap.className = 'la-inline-actions'
    for (const q of quickBtns) {
      const b = document.createElement('button')
      b.type = 'button'
      b.className = 'la-btn la-btn--sm'
      b.textContent = q
      b.setAttribute('aria-label', `Usar error relativo ${q}`)
      b.addEventListener('click', () => {
        input.value = q
        userErrBText = q
        updateEstimateOutput()
      })
      quickWrap.appendChild(b)
    }
    controlRow.appendChild(quickWrap)
    body.appendChild(controlRow)

    const errBanner = document.createElement('div')
    errBanner.className = 'la-error-banner'
    errBanner.setAttribute('role', 'alert')
    errBanner.hidden = true
    body.appendChild(errBanner)

    const outputContainer = document.createElement('div')
    outputContainer.className = 'la-stack'
    body.appendChild(outputContainer)

    function updateEstimateOutput(): void {
      userErrBText = input.value
      const errB = parseNonNegativeNum(input.value)
      if (errB === null) {
        input.setAttribute('aria-invalid', 'true')
        showBannerError(
          errBanner,
          'Escribe un número finito no negativo para el error relativo de b (por ejemplo 1e-6 o 0.001).',
        )
        outputContainer.innerHTML = ''
        return
      }

      input.setAttribute('aria-invalid', 'false')
      showBannerError(errBanner, null)
      outputContainer.innerHTML = ''

      const bound = estimaErrorSolucion(effectiveKappa, errB)
      const tex = Number.isFinite(bound)
        ? `\\frac{\\|\\delta x\\|}{\\|x\\|} \\le ${meta.kappaTex} \\cdot \\frac{\\|\\delta b\\|}{\\|b\\|} \\approx ${formatCondNumLatex(effectiveKappa)} \\cdot ${formatCondNumLatex(errB)} = ${formatCondNumLatex(bound)}`
        : `\\frac{\\|\\delta x\\|}{\\|x\\|} \\le \\infty \\quad (\\text{A es singular})`
      const plain = Number.isFinite(bound)
        ? `‖δx‖/‖x‖ ≤ ${formatCondNum(effectiveKappa)} · ${formatCondNum(errB)} = ${formatCondNum(bound)}`
        : '‖δx‖/‖x‖ ≤ ∞ (A es singular)'

      outputContainer.appendChild(createMathBox(tex, plain, true))

      const insight = document.createElement('p')
      insight.className = 'la-note'
      insight.textContent = formatErrorInsight(bound, effectiveKappa)
      outputContainer.appendChild(insight)
    }

    input.addEventListener('input', updateEstimateOutput)
    updateEstimateOutput()
  }

  function renderEquilibrateCard(state: ComputedConditionState, selectedNorm: MatrixNorm): void {
    if (!showEquilibration || !state.equilibrated) {
      cardEquilibrate.hidden = true
      return
    }

    const eq = state.equilibrated
    const meta = NORM_META[selectedNorm]
    const beforeKappa = state.exactSingular
      ? Number.POSITIVE_INFINITY
      : state.conds[selectedNorm].kappa
    const afterKappa = state.exactSingular
      ? Number.POSITIVE_INFINITY
      : eq.condsAfter[selectedNorm].kappa

    const body = createResultFrame(
      cardEquilibrate,
      'Escalado: equilibrado de filas (D⁻¹ A)',
      `Cada fila dividida por su máximo absoluto · Norma ${meta.label}`,
    )

    const summary = document.createElement('div')
    summary.className = 'la-residual-strip'
    const improvementText =
      Number.isFinite(beforeKappa) && Number.isFinite(afterKappa) && afterKappa > 0
        ? beforeKappa / afterKappa >= 1.05
          ? `Mejora: ${(beforeKappa / afterKappa).toPrecision(3)}× menor`
          : afterKappa / beforeKappa >= 1.05
            ? `Variación: ${(afterKappa / beforeKappa).toPrecision(3)}× mayor`
            : 'Sin cambio significativo'
        : 'Matriz singular (el escalado no elimina la singularidad)'

    summary.innerHTML = `
      <span>κ antes (${meta.symbolPlain}): <code>${formatCondNum(beforeKappa)}</code></span>
      <span>κ después (${meta.symbolPlain}): <code>${formatCondNum(afterKappa)}</code></span>
      <span>Resultado: <code>${improvementText}</code></span>
    `
    body.appendChild(summary)

    const verdictAfter = veredictoCondicion(afterKappa)
    body.appendChild(buildVerdictBlock(verdictAfter, afterKappa))

    const wrap = document.createElement('div')
    wrap.className = 'la-table-wrap la-cond-table-wrap'
    wrap.tabIndex = 0
    wrap.setAttribute('role', 'region')
    wrap.setAttribute('aria-labelledby', 'cond-eq-caption')

    const table = document.createElement('table')
    table.className = 'la-table la-cond-table'
    table.style.setProperty('--label-1', '"Norma"')
    table.style.setProperty('--label-2', '"κ antes"')
    table.style.setProperty('--label-3', '"κ después"')

    const caption = document.createElement('caption')
    caption.id = 'cond-eq-caption'
    caption.className = 'la-sr-only'
    caption.textContent = 'Comparación del número de condición antes y después de equilibrar las filas de A'
    table.appendChild(caption)

    const eqThead = document.createElement('thead')
    eqThead.innerHTML = `
      <tr>
        <th scope="col">Norma</th>
        <th scope="col">κ antes</th>
        <th scope="col">κ después (equilibrada)</th>
      </tr>
    `
    table.appendChild(eqThead)

    const tbody = document.createElement('tbody')
    for (const p of ALL_NORMS) {
      const tr = document.createElement('tr')
      if (p === selectedNorm) tr.className = 'is-cond-selected'
      const kBefore = state.exactSingular ? Number.POSITIVE_INFINITY : state.conds[p].kappa
      const kAfter = state.exactSingular ? Number.POSITIVE_INFINITY : eq.condsAfter[p].kappa
      tr.innerHTML = `
        <th scope="row" class="la-table-method">${NORM_META[p].symbolPlain}</th>
        <td class="la-table-mono">${formatCondNum(kBefore)}</td>
        <td class="la-table-mono">${formatCondNum(kAfter)}</td>
      `
      tbody.appendChild(tr)
    }
    table.appendChild(tbody)
    wrap.appendChild(table)
    body.appendChild(wrap)

    const scalesLine = document.createElement('p')
    scalesLine.className = 'la-note'
    scalesLine.textContent = `Factores de escala por fila (máx |aᵢⱼ|): [${eq.escalas.map(formatCondNum).join(', ')}].`
    body.appendChild(scalesLine)

    body.appendChild(
      createMathBox(
        `A_{\\text{eq}} = D^{-1} A = ${numMatrixToLatex(eq.matriz)}`,
        `A_eq =\n${numMatrixToPlain(eq.matriz)}`,
        true,
      ),
    )

    const applyRow = document.createElement('div')
    applyRow.className = 'la-inline-actions'
    const applyBtn = document.createElement('button')
    applyBtn.type = 'button'
    applyBtn.className = 'la-btn la-btn--sm'
    applyBtn.textContent = 'Cargar matriz equilibrada en A'
    applyBtn.addEventListener('click', () => {
      const cells = eq.matriz.map((row) =>
        row.map((v) => {
          if (Math.abs(v) < 1e-15) return '0'
          return Number(v.toPrecision(8)).toString()
        }),
      )
      loadMatrixCells(
        cells,
        null,
        'Matriz equilibrada cargada (cada fila dividida por su mayor valor absoluto).',
      )
      runSolve(false)
    })
    applyRow.appendChild(applyBtn)

    const eqNote = document.createElement('span')
    eqNote.className = 'la-note'
    eqNote.textContent =
      'El equilibrado corrige el mal condicionamiento artificial por escalas dispares, pero no arregla matrices intrínsecamente mal condicionadas (como Hilbert).'
    applyRow.appendChild(eqNote)
    body.appendChild(applyRow)
  }

  function renderAllResults(): void {
    if (!lastComputed) return
    const selectedNorm = parseNormValue(normSelect.value)
    emptyHintEl.hidden = true
    resultsContainer.hidden = false
    clearResultsStale()

    renderNormsCard(lastComputed, selectedNorm)
    renderKappaCard(lastComputed, selectedNorm)
    renderErrorEstimateCard(lastComputed, selectedNorm)
    renderEquilibrateCard(lastComputed, selectedNorm)
  }

  function runSolve(triggerEquilibrate = false): void {
    const readA = gridA.readMatrix()
    if (!readA.ok) {
      showBannerError(inputErrorEl, readA.error)
      readA.firstInvalid.focus()
      return
    }

    const a = readA.matriz
    if (!isSquare(a)) {
      const msg = `La matriz A es ${a.length}×${a[0]?.length ?? 0} y no es cuadrada. El número de condición κ(A) = ‖A‖ · ‖A⁻¹‖ solo está definido para matrices cuadradas n×n.`
      showBannerError(inputErrorEl, msg)
      emptyHintEl.hidden = true
      resultsContainer.hidden = false
      clearResultsStale()
      for (const card of allCards) card.hidden = true
      setResultError(cardKappa, 'Error: matriz no cuadrada', msg)
      return
    }

    const n = a.length
    const aNum = a.map((row) => row.map((r) => r.toNumber()))

    try {
      const norms: Record<MatrixNorm, number> = {
        1: normMatrix(aNum, 1),
        2: normMatrix(aNum, 2),
        inf: normMatrix(aNum, 'inf'),
        fro: normMatrix(aNum, 'fro'),
      }

      const exactRank = rank(a)
      const exactInv = inverse(a)
      const exactSingular = exactInv.inversa === null
      const exactAtasco = exactInv.inversa === null ? exactInv.atasco : null

      let exactDetText: string | null = null
      if (n <= 5) {
        const d = det(a).valor
        exactDetText = d.d === 1n ? ratToString(d) : `${ratToString(d)} (≈ ${ratToDecimal(d, 6)})`
      }

      const conds: Record<MatrixNorm, CondNumberResult> = {
        1: condNumber(aNum, 1),
        2: condNumber(aNum, 2),
        inf: condNumber(aNum, 'inf'),
        fro: condNumber(aNum, 'fro'),
      }

      const eqRes = equilibrarFilas(aNum)
      const condsAfter: Record<MatrixNorm, CondNumberResult> = {
        1: condNumber(eqRes.matriz, 1),
        2: condNumber(eqRes.matriz, 2),
        inf: condNumber(eqRes.matriz, 'inf'),
        fro: condNumber(eqRes.matriz, 'fro'),
      }

      lastComputed = {
        a,
        aNum,
        norms,
        conds,
        exactRank,
        exactSingular,
        exactAtasco,
        exactDetText,
        equilibrated: {
          matriz: eqRes.matriz,
          escalas: eqRes.escalas,
          condsAfter,
        },
      }
      if (triggerEquilibrate) {
        showEquilibration = true
      }
    } catch (err) {
      const msg =
        err instanceof Error && err.message.trim()
          ? err.message
          : 'No se pudo completar el análisis de condicionamiento.'
      showBannerError(inputErrorEl, msg)
      if (!resultsContainer.hidden) {
        markResultsStale()
      }
      return
    }

    showBannerError(inputErrorEl, null)
    renderAllResults()

    const selectedNorm = parseNormValue(normSelect.value)
    const activeKappa = lastComputed.exactSingular
      ? Number.POSITIVE_INFINITY
      : lastComputed.conds[selectedNorm].kappa
    const verdict = veredictoCondicion(activeKappa)
    track('linealalrl-cond-solve', {
      size: `${n}x${n}`,
      norm: String(selectedNorm),
      nivel: verdict.nivel,
      equilibrate: triggerEquilibrate ? 'yes' : 'no',
    })
  }

  btnSolve.addEventListener('click', () => runSolve(false))
  btnEquilibrate.addEventListener('click', () => runSolve(true))
  normSelect.addEventListener('change', () => {
    if (lastComputed && !resultsContainer.hidden && !cardKappa.classList.contains('is-stale')) {
      renderAllResults()
    }
  })

  // -------------------------------------------------------------------------
  // Avisos didácticos (determinante casi cero y tabla de Hilbert H₃ / H₄)
  // -------------------------------------------------------------------------

  function renderDidacticSections(): void {
    // 1. Aviso: determinante casi cero no significa singular ni mal condicionada
    const smallDetNum = [
      [0.01, 0, 0],
      [0, 0.01, 0],
      [0, 0, 0.01],
    ]
    const smallDetCond2 = condNumber(smallDetNum, 2).kappa

    didacticDetEl.innerHTML = ''
    const h3Det = document.createElement('h3')
    h3Det.className = 'la-subheading'
    h3Det.textContent = 'Determinante casi cero no significa singular (ni mal condicionada)'
    didacticDetEl.appendChild(h3Det)

    const pDet = document.createElement('p')
    pDet.className = 'la-note'
    pDet.textContent =
      'El determinante depende de la escala de la matriz (det(cA) = cⁿ det(A)), mientras que el número de condición κ(cA) = κ(A) es invariante bajo escalado uniforme. Por eso det(A) ≈ 0 NO sirve para medir si una matriz está cerca de ser singular.'
    didacticDetEl.appendChild(pDet)

    didacticDetEl.appendChild(
      createMathBox(
        `A = 10^{-2} I_3 = \\begin{bmatrix} 0.01 & 0 & 0 \\\\ 0 & 0.01 & 0 \\\\ 0 & 0 & 0.01 \\end{bmatrix}, \\quad \\det(A) = 10^{-6}, \\quad \\kappa_2(A) = ${formatCondNumLatex(smallDetCond2)}`,
        `A = 0.01 I₃: det(A) = 1e-6, pero κ₂(A) = ${formatCondNum(smallDetCond2)}`,
        true,
      ),
    )

    const pDet2 = document.createElement('p')
    pDet2.className = 'la-note'
    pDet2.textContent =
      'Ejemplo: esta matriz tiene det(A) = 10⁻⁶, pero es perfectamente condicionada (κ₁ = κ₂ = κ∞ = 1). A la inversa, una matriz puede tener det(A) = 1 y tener un número de condición enorme.'
    didacticDetEl.appendChild(pDet2)

    const loadDetBtn = document.createElement('button')
    loadDetBtn.type = 'button'
    loadDetBtn.className = 'la-btn la-btn--sm'
    loadDetBtn.textContent = 'Cargar ejemplo (det = 10⁻⁶, κ = 1) en A'
    loadDetBtn.addEventListener('click', () => {
      loadMatrixCells(
        SMALL_DET_WELL_COND_CELLS,
        null,
        'Ejemplo didáctico: A = 0.01·I₃ tiene det = 10⁻⁶ pero κ₁ = κ₂ = κ∞ = 1 (perfectamente condicionada).',
      )
      runSolve(false)
    })
    didacticDetEl.appendChild(loadDetBtn)

    // 2. Aviso: ejemplo clásico de las matrices de Hilbert (H₃ y H₄ destacados)
    didacticHilbertEl.innerHTML = ''
    const h3Hilb = document.createElement('h3')
    h3Hilb.className = 'la-subheading'
    h3Hilb.textContent = 'Ejemplo clásico: matrices de Hilbert Hₙ'
    didacticHilbertEl.appendChild(h3Hilb)

    const pHilb = document.createElement('p')
    pHilb.className = 'la-note'
    pHilb.textContent =
      'Las matrices de Hilbert H[i][j] = 1/(i+j+1) son simétricas y definidas positivas (siempre invertibles), pero sus columnas son casi colineales: κ(Hₙ) crece exponencialmente con n.'
    didacticHilbertEl.appendChild(pHilb)

    const hilbWrap = document.createElement('div')
    hilbWrap.className = 'la-table-wrap la-cond-table-wrap'
    hilbWrap.tabIndex = 0
    hilbWrap.setAttribute('role', 'region')
    hilbWrap.setAttribute('aria-labelledby', 'cond-hilbert-caption')

    const hilbTable = document.createElement('table')
    hilbTable.className = 'la-table la-cond-table'
    hilbTable.style.setProperty('--label-1', '"Matriz"')
    hilbTable.style.setProperty('--label-2', '"κ₂(Hₙ)"')
    hilbTable.style.setProperty('--label-3', '"κ∞(Hₙ)"')
    hilbTable.style.setProperty('--label-4', '"Dígitos perdidos"')
    hilbTable.style.setProperty('--label-5', '"Veredicto"')

    const hilbCaption = document.createElement('caption')
    hilbCaption.id = 'cond-hilbert-caption'
    hilbCaption.className = 'la-sr-only'
    hilbCaption.textContent = 'Crecimiento del número de condición en las matrices de Hilbert H2, H3, H4 y H5'
    hilbTable.appendChild(hilbCaption)

    const hilbThead = document.createElement('thead')
    hilbThead.innerHTML = `
      <tr>
        <th scope="col">Matriz</th>
        <th scope="col">κ₂(Hₙ)</th>
        <th scope="col">κ∞(Hₙ)</th>
        <th scope="col">Dígitos (~log₁₀ κ₂)</th>
        <th scope="col">Veredicto</th>
      </tr>
    `
    hilbTable.appendChild(hilbThead)

    const hilbBody = document.createElement('tbody')
    for (const order of [2, 3, 4, 5]) {
      const hMat = hilbert(order)
      const k2 = condNumber(hMat, 2).kappa
      const kInf = condNumber(hMat, 'inf').kappa
      const verd = veredictoCondicion(k2)
      const digits = Math.log10(k2).toFixed(1)

      const tr = document.createElement('tr')
      if (order === 3 || order === 4) {
        tr.className = 'is-cond-selected'
      }

      const levelText =
        verd.nivel === 'bien' ? 'Bien' : verd.nivel === 'moderada' ? 'Moderada' : 'Mal'

      tr.innerHTML = `
        <th scope="row" class="la-table-method">H<sub>${order}</sub> (${order}×${order})</th>
        <td class="la-table-mono">${formatCondNum(k2)}</td>
        <td class="la-table-mono">${formatCondNum(kInf)}</td>
        <td class="la-table-mono">~${digits}</td>
        <td><span class="la-cond-pill la-cond-pill--${verd.nivel}">${levelText}</span></td>
      `
      hilbBody.appendChild(tr)
    }
    hilbTable.appendChild(hilbBody)
    hilbWrap.appendChild(hilbTable)
    didacticHilbertEl.appendChild(hilbWrap)

    const hilbActions = document.createElement('div')
    hilbActions.className = 'la-inline-actions'
    for (const order of [3, 4] as const) {
      const btn = document.createElement('button')
      btn.type = 'button'
      btn.className = 'la-btn la-btn--sm'
      btn.textContent = `Cargar H${order === 3 ? '₃' : '₄'} (${order}×${order})`
      btn.addEventListener('click', () => {
        const preset = CONDITION_PRESETS.find((p) => p.id === `hilbert-${order}x${order}`) ?? null
        loadMatrixCells(hilbertCellStrings(order), preset)
        runSolve(false)
      })
      hilbActions.appendChild(btn)
    }
    didacticHilbertEl.appendChild(hilbActions)
  }

  renderDidacticSections()
}
