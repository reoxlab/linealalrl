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

// Interfaz interactiva del módulo 1: solucionadores directos (Gauss, LU, Cholesky, QR y comparador).

import {
  type DirectCompareRow,
  choleskySolve,
  directCompare,
  gaussSolve,
  luSolve,
  qrSolve,
} from './direct.ts'
import { renderLatex } from './katexRender.ts'
import { type Matrix, isSquare, matrixToLatex } from './matrix.ts'
import { parsePastedAugmented } from './paste.ts'
import { type Rat, parseRat, ratToDecimal, ratToLatex, ratToString } from './rational.ts'
import { MAX_GRID_DIM } from './share.ts'
import {
  type TrackFn,
  buildLatexCopyBlock,
  createMatrixGrid,
  createResultFrame,
  formatStepsLatex,
  renderStepsElement,
  setResultError,
} from './ui.ts'

const LATEX_HEADER = '% Requiere \\usepackage{amsmath}'

type DirectSelectorValue = 'gauss' | 'lu' | 'cholesky' | 'qr' | 'compare'

interface DirectPreset {
  id: string
  nombre: string
  descripcion: string
  a: string[][]
  b: string[]
}

export const DIRECT_PRESETS: readonly DirectPreset[] = [
  {
    id: 'sistema-3x3',
    nombre: 'Sistema 3×3 (clásico)',
    descripcion:
      'Sistema 3×3 con solución exacta conocida x = (2, 3, −1). Ideal para Gauss, LU, QR y el comparador (no simétrica, Cholesky explica por qué no aplica).',
    a: [
      ['2', '1', '-1'],
      ['-3', '-1', '2'],
      ['-2', '1', '2'],
    ],
    b: ['8', '-11', '-3'],
  },
  {
    id: 'spd-3x3',
    nombre: 'Simétrica def. positiva 3×3 (SPD)',
    descripcion:
      'Matriz simétrica definida positiva con solución exacta x = (1, 2, −1): los 4 métodos (incluyendo Cholesky) aplican.',
    a: [
      ['4', '2', '-2'],
      ['2', '10', '2'],
      ['-2', '2', '6'],
    ],
    b: ['10', '20', '-4'],
  },
  {
    id: 'pivote-cero',
    nombre: 'Pivote cero (a₁₁ = 0)',
    descripcion:
      'Requiere pivoteo parcial (intercambio de filas desde la columna 1). Solución exacta x = (−4, −5, 2).',
    a: [
      ['0', '2', '1'],
      ['1', '-2', '-3'],
      ['-1', '1', '2'],
    ],
    b: ['-8', '0', '3'],
  },
  {
    id: 'indefinida-2x2',
    nombre: 'Simétrica indefinida 2×2',
    descripcion:
      'Simétrica pero no definida positiva (pivote 2 negativo, x = (1, 2)): muestra el diagnóstico de Cholesky y la sugerencia de LU.',
    a: [
      ['1', '2'],
      ['2', '1'],
    ],
    b: ['5', '4'],
  },
]

function formatNum(v: number): string {
  if (!Number.isFinite(v)) return String(v)
  if (Math.abs(v) < 1e-12) return '0'
  const rounded = Math.round(v)
  if (Math.abs(v - rounded) < 1e-10 && Math.abs(v) < 1e9) {
    return String(rounded)
  }
  if (Math.abs(v) >= 1e5 || Math.abs(v) < 1e-4) {
    return v.toExponential(4)
  }
  return Number(v.toFixed(6)).toString()
}

function formatResidual(r: number): string {
  if (!Number.isFinite(r)) return '—'
  if (r === 0) return '0'
  return r.toExponential(2)
}

function numMatrixToLatex(m: number[][]): string {
  const filas = m.map((row) => row.map(formatNum).join(' & ')).join(' \\\\ ')
  return `\\begin{bmatrix} ${filas} \\end{bmatrix}`
}

function numMatrixToString(m: number[][]): string {
  return m.map((row) => row.map(formatNum).join(' ')).join('\n')
}

function numVectorToLatex(v: number[]): string {
  return `\\begin{bmatrix} ${v.map(formatNum).join(' \\\\ ')} \\end{bmatrix}`
}

function ratVectorToLatex(v: Rat[]): string {
  return `\\begin{bmatrix} ${v.map(ratToLatex).join(' \\\\ ')} \\end{bmatrix}`
}

function ratVectorToPlain(v: Rat[], name = 'x'): string {
  return v.map((val, idx) => `${name}${idx + 1} = ${ratToString(val)}`).join(', ')
}

function numVectorToPlain(v: number[], name = 'x'): string {
  return v.map((val, idx) => `${name}${idx + 1} = ${formatNum(val)}`).join(', ')
}

function createModeBadge(exacto: boolean): HTMLDivElement {
  const badge = document.createElement('div')
  badge.className = exacto ? 'la-sys-badge la-sys-badge--unica' : 'la-sys-badge la-sys-badge--approx'
  badge.textContent = exacto
    ? 'Resultado exacto (fracciones sin error de redondeo)'
    : 'Resultado numérico aproximado (punto flotante)'
  return badge
}

function createMathBox(tex: string, fallbackText: string, compact = false): HTMLDivElement {
  const box = document.createElement('div')
  box.className = compact ? 'la-result-math la-result-math--compact' : 'la-result-math'
  void renderLatex(tex, box, { displayMode: true, fallbackText })
  return box
}

export function mountDirectUI(app: HTMLElement, track: TrackFn): void {
  const defaultPreset = DIRECT_PRESETS[0]
  let vectorValues: string[] = defaultPreset.b.slice()
  let activePresetId: string | null = defaultPreset.id
  let vectorInputs: HTMLInputElement[] = []

  app.innerHTML = `
    <div class="la-workspace">
      <section class="la-panel">
        <div class="la-panel-head">
          <h2 class="la-panel-title">Sistema lineal Ax = b (solucionadores directos)</h2>
          <span class="la-panel-meta" id="direct-dim-meta"></span>
        </div>

        <div class="la-toolbar" id="direct-presets-bar">
          <span class="la-toolbar-label">Presets:</span>
        </div>
        <p class="la-preset-desc" id="direct-preset-desc"></p>

        <div class="la-panel-body la-direct-input-layout">
          <div class="la-direct-system-editors">
            <div class="la-direct-matrix-block">
              <span class="la-field-label">Matriz de coeficientes A (cuadrada, hasta 6×6)</span>
              <div id="direct-grid-a-mount"></div>
            </div>

            <div class="la-direct-vector-block">
              <div class="la-vector-head">
                <span class="la-field-label">Vector b</span>
                <button class="la-btn la-btn--sm" id="btn-direct-clear-b" type="button" aria-label="Limpiar vector b">Limpiar b</button>
              </div>
              <div class="la-matrix-scroll">
                <div class="la-matrix-bracket">
                  <div class="la-vector-grid" id="direct-vector-b-grid"></div>
                </div>
              </div>
            </div>
          </div>

          <div class="la-paste-box">
            <label class="la-field-label" for="direct-paste-input">Pegar sistema [A | b]</label>
            <p class="la-hint" id="direct-paste-hint">Pega la matriz aumentada (última columna = vector b; separador «|» opcional, hasta 6×6).</p>
            <textarea
              id="direct-paste-input"
              class="la-textarea"
              rows="4"
              spellcheck="false"
              aria-describedby="direct-paste-hint"
              placeholder="2 1 -1 | 8&#10;-3 -1 2 | -11&#10;-2 1 2 | -3"
            ></textarea>
            <div class="la-paste-actions">
              <button class="la-btn la-btn--sm" id="btn-direct-paste" type="button">Pegar sistema</button>
              <span class="la-feedback" id="direct-paste-feedback" role="status" aria-live="polite" hidden></span>
            </div>
            <div class="la-error-banner" id="direct-paste-error" role="alert" hidden></div>
          </div>
        </div>

        <div class="la-error-banner la-panel-error" id="direct-input-error" role="alert" hidden></div>
      </section>

      <section class="la-panel">
        <div class="la-panel-head">
          <h2 class="la-panel-title">Método directo</h2>
          <span class="la-panel-meta">Gauss y LU exactos · Cholesky y QR numéricos</span>
        </div>

        <div class="la-toolbar la-direct-solve-bar">
          <label class="la-direct-method-label" for="direct-method-select">
            <span>Método:</span>
            <select class="la-select la-direct-method-select" id="direct-method-select">
              <option value="gauss" selected>Eliminación de Gauss (exacto)</option>
              <option value="lu">Factorización LU · PA = LU (exacto)</option>
              <option value="cholesky">Cholesky · A = LLᵀ (numérico, SPD)</option>
              <option value="qr">QR · Householder (numérico)</option>
              <option value="compare">Comparar todos</option>
            </select>
          </label>

          <button class="la-btn la-btn--primary" id="btn-direct-solve" type="button">Resolver</button>
        </div>

        <div class="la-panel-body la-results-stack">
          <p class="la-note" id="direct-empty-hint">
            Selecciona un método directo o «Comparar todos» y pulsa <strong>Resolver</strong> para ver la solución, los pasos, las factorizaciones y el residuo.
          </p>
          <article class="la-result-card" id="direct-result-card" hidden></article>
        </div>
      </section>
    </div>
  `

  const presetsBar = app.querySelector<HTMLDivElement>('#direct-presets-bar')!
  const presetDescEl = app.querySelector<HTMLParagraphElement>('#direct-preset-desc')!
  const dimMetaEl = app.querySelector<HTMLSpanElement>('#direct-dim-meta')!
  const gridAMount = app.querySelector<HTMLDivElement>('#direct-grid-a-mount')!
  const vectorBGrid = app.querySelector<HTMLDivElement>('#direct-vector-b-grid')!
  const btnClearB = app.querySelector<HTMLButtonElement>('#btn-direct-clear-b')!
  const pasteInput = app.querySelector<HTMLTextAreaElement>('#direct-paste-input')!
  const btnPaste = app.querySelector<HTMLButtonElement>('#btn-direct-paste')!
  const pasteFeedbackEl = app.querySelector<HTMLSpanElement>('#direct-paste-feedback')!
  const pasteErrorEl = app.querySelector<HTMLDivElement>('#direct-paste-error')!
  const inputErrorEl = app.querySelector<HTMLDivElement>('#direct-input-error')!

  const methodSelect = app.querySelector<HTMLSelectElement>('#direct-method-select')!
  const btnSolve = app.querySelector<HTMLButtonElement>('#btn-direct-solve')!
  const emptyHintEl = app.querySelector<HTMLParagraphElement>('#direct-empty-hint')!
  const resultCardEl = app.querySelector<HTMLElement>('#direct-result-card')!

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

  function renderVectorB(rowCount: number): void {
    vectorValues = Array.from({ length: rowCount }, (_, i) => vectorValues[i] ?? '0')
    vectorBGrid.innerHTML = ''
    vectorInputs = []

    for (let i = 0; i < rowCount; i++) {
      const row = document.createElement('div')
      row.className = 'la-vector-row'

      const tag = document.createElement('span')
      tag.className = 'la-vector-idx'
      tag.textContent = `b${i + 1}`

      const input = document.createElement('input')
      input.type = 'text'
      input.className = 'la-cell-input'
      input.value = vectorValues[i]
      input.spellcheck = false
      input.autocomplete = 'off'
      input.setAttribute('aria-label', `Vector b (directos), fila ${i + 1}`)
      input.setAttribute('aria-invalid', 'false')

      input.addEventListener('input', () => {
        vectorValues[i] = input.value
        if (input.classList.contains('is-invalid') && parseRat(input.value) !== null) {
          input.classList.remove('is-invalid')
          input.setAttribute('aria-invalid', 'false')
        }
        activePresetId = null
        showBannerError(inputErrorEl, null)
        markResultStale()
        updatePresetButtons()
      })

      row.appendChild(tag)
      row.appendChild(input)
      vectorBGrid.appendChild(row)
      vectorInputs.push(input)
    }
  }

  function readVectorB():
    | { ok: true; vector: Rat[] }
    | { ok: false; error: string; firstInvalid: HTMLInputElement } {
    const out: Rat[] = []
    let firstInvalid: HTMLInputElement | null = null
    const badRows: number[] = []

    for (let i = 0; i < vectorInputs.length; i++) {
      const input = vectorInputs[i]
      vectorValues[i] = input.value
      const val = parseRat(input.value)
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
        error: `Revisa el vector b en ${badRows.length === 1 ? `la fila ${badRows[0]}` : `las filas ${badRows.join(', ')}`}: escribe enteros, decimales o fracciones como 3/4.`,
      }
    }

    return { ok: true, vector: out }
  }

  function updateDimMeta(): void {
    const [r, c] = gridA.getSize()
    dimMetaEl.textContent =
      r === c
        ? `Sistema cuadrado ${r}×${c} (Ax = b)`
        : `A es ${r}×${c} (no cuadrada: los solucionadores directos requieren n×n)`
  }

  const presetButtons = new Map<string, HTMLButtonElement>()
  function updatePresetButtons(): void {
    for (const [id, btn] of presetButtons.entries()) {
      const active = id === activePresetId
      btn.classList.toggle('la-btn--active', active)
      btn.setAttribute('aria-pressed', active ? 'true' : 'false')
    }
    const preset = DIRECT_PRESETS.find((p) => p.id === activePresetId)
    presetDescEl.textContent = preset
      ? preset.descripcion
      : 'Sistema personalizado (admite enteros, decimales y fracciones como 3/4).'
  }

  const gridA = createMatrixGrid({
    idPrefix: 'direct-grid-a',
    matrixName: 'A',
    initialValues: defaultPreset.a,
    onSizeChange: (rows) => {
      renderVectorB(rows)
      updateDimMeta()
    },
    onInputChange: () => {
      activePresetId = null
      showBannerError(inputErrorEl, null)
      markResultStale()
      updatePresetButtons()
    },
  })

  gridAMount.appendChild(gridA.element)
  renderVectorB(defaultPreset.a.length)
  updateDimMeta()

  function applyPreset(preset: DirectPreset): void {
    activePresetId = preset.id
    vectorValues = preset.b.slice()
    gridA.setValues(preset.a)
    renderVectorB(preset.a.length)
    showBannerError(inputErrorEl, null)
    showBannerError(pasteErrorEl, null)
    markResultStale()
    updatePresetButtons()
    updateDimMeta()
  }

  for (const preset of DIRECT_PRESETS) {
    const btn = document.createElement('button')
    btn.type = 'button'
    btn.className = 'la-btn la-btn--sm'
    btn.textContent = preset.nombre
    btn.addEventListener('click', () => {
      applyPreset(preset)
      track('linealalrl-direct-preset', { preset: preset.id })
    })
    presetsBar.appendChild(btn)
    presetButtons.set(preset.id, btn)
  }
  updatePresetButtons()

  btnClearB.addEventListener('click', () => {
    vectorValues = vectorValues.map(() => '0')
    renderVectorB(vectorValues.length)
    activePresetId = null
    markResultStale()
    updatePresetButtons()
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
    vectorValues = parsed.vector.map(ratToString)
    gridA.setMatrix(parsed.a)
    renderVectorB(parsed.filas)
    markResultStale()
    updatePresetButtons()
    updateDimMeta()
    showTimedFeedback(
      pasteFeedbackEl,
      `Sistema cargado (${parsed.filas}×${parsed.cols} + vector b)`,
    )
    track('linealalrl-direct-paste', { size: `${parsed.filas}x${parsed.cols}` })
  })

  function renderExactSolutionSummary(body: HTMLElement, x: Rat[]): void {
    const varsTex = x.map((_, i) => `x_${i + 1}`).join(' \\\\ ')
    body.appendChild(
      createMathBox(
        `\\mathbf{x} = \\begin{bmatrix} ${varsTex} \\end{bmatrix} = ${ratVectorToLatex(x)}`,
        ratVectorToPlain(x),
      ),
    )

    const hasFraction = x.some((v) => v.d !== 1n)
    if (hasFraction) {
      const approx = document.createElement('p')
      approx.className = 'la-result-meta-line'
      approx.textContent =
        'Aproximación decimal: ' +
        x.map((v, idx) => `x${idx + 1} ≈ ${ratToDecimal(v, 6)}`).join(', ')
      body.appendChild(approx)
    }

    const resLine = document.createElement('p')
    resLine.className = 'la-note'
    resLine.textContent = 'Residuo exacto: ‖Ax − b‖₂ = 0 (cálculo exacto en fracciones).'
    body.appendChild(resLine)
  }

  function renderNumericSolutionSummary(
    body: HTMLElement,
    x: number[],
    residuo: number,
    factorLabel: string,
    residuoFactor: number,
  ): void {
    const varsTex = x.map((_, i) => `x_${i + 1}`).join(' \\\\ ')
    body.appendChild(
      createMathBox(
        `\\mathbf{x} \\approx \\begin{bmatrix} ${varsTex} \\end{bmatrix} = ${numVectorToLatex(x)}`,
        numVectorToPlain(x),
      ),
    )

    const resBox = document.createElement('div')
    resBox.className = 'la-residual-strip'
    resBox.innerHTML = `
      <span><strong>Residuo ‖Ax − b‖₂:</strong> <code>${formatResidual(residuo)}</code></span>
      <span><strong>${factorLabel}:</strong> <code>${formatResidual(residuoFactor)}</code></span>
    `
    body.appendChild(resBox)
  }

  function renderCompareTable(body: HTMLElement, rows: DirectCompareRow[]): void {
    const note = document.createElement('p')
    note.className = 'la-note'
    note.textContent =
      'Gauss y LU usan fracciones exactas (residuo 0). Cholesky y QR son resultados numéricos aproximados en punto flotante y muestran el residuo ‖Ax − b‖₂.'
    body.appendChild(note)

    const tableWrap = document.createElement('div')
    tableWrap.className = 'la-table-wrap'

    const table = document.createElement('table')
    table.className = 'la-table'
    table.innerHTML = `
      <thead>
        <tr>
          <th scope="col">Método</th>
          <th scope="col">Aritmética</th>
          <th scope="col">Solución x</th>
          <th scope="col">Residuo ‖Ax − b‖₂</th>
          <th scope="col">Operaciones aprox.</th>
          <th scope="col">Cuándo usarlo</th>
        </tr>
      </thead>
      <tbody></tbody>
    `
    const tbody = table.querySelector('tbody')!

    for (const row of rows) {
      const tr = document.createElement('tr')
      if (!row.aplicable) tr.className = 'is-not-applicable'

      const tdMethod = document.createElement('td')
      tdMethod.className = 'la-table-method'
      tdMethod.textContent = row.metodo

      const tdType = document.createElement('td')
      const pill = document.createElement('span')
      if (!row.aplicable) {
        pill.className = 'la-table-pill la-table-pill--na'
        pill.textContent = 'No aplica'
      } else if (row.exacto) {
        pill.className = 'la-table-pill la-table-pill--exact'
        pill.textContent = 'Exacto (Rat)'
      } else {
        pill.className = 'la-table-pill la-table-pill--approx'
        pill.textContent = 'Numérico aprox.'
      }
      tdType.appendChild(pill)

      const tdSol = document.createElement('td')
      if (!row.aplicable) {
        const reason = document.createElement('span')
        reason.className = 'la-table-reason'
        reason.textContent = row.motivo ?? 'No aplicable para esta matriz.'
        tdSol.appendChild(reason)
      } else if (row.exacto) {
        const vec = row.x as Rat[]
        const code = document.createElement('code')
        code.textContent = `(${vec.map(ratToString).join(', ')})`
        tdSol.appendChild(code)
      } else {
        const vec = row.x as number[]
        const code = document.createElement('code')
        code.textContent = `(${vec.map(formatNum).join(', ')})`
        tdSol.appendChild(code)
      }

      const tdRes = document.createElement('td')
      tdRes.className = 'la-table-mono'
      if (!row.aplicable) {
        tdRes.textContent = '—'
      } else if (row.exacto) {
        tdRes.textContent = '0 (exacto)'
      } else {
        tdRes.textContent = formatResidual(row.residuo)
      }

      const tdOps = document.createElement('td')
      tdOps.textContent = row.operaciones

      const tdWhen = document.createElement('td')
      tdWhen.textContent = row.cuandoUsar

      tr.appendChild(tdMethod)
      tr.appendChild(tdType)
      tr.appendChild(tdSol)
      tr.appendChild(tdRes)
      tr.appendChild(tdOps)
      tr.appendChild(tdWhen)
      tbody.appendChild(tr)
    }

    tableWrap.appendChild(table)
    body.appendChild(tableWrap)
  }

  function runSolve(): void {
    const readA = gridA.readMatrix()
    if (!readA.ok) {
      showBannerError(inputErrorEl, readA.error)
      readA.firstInvalid.focus()
      return
    }
    const readB = readVectorB()
    if (!readB.ok) {
      showBannerError(inputErrorEl, readB.error)
      readB.firstInvalid.focus()
      return
    }
    showBannerError(inputErrorEl, null)

    const a: Matrix = readA.matriz
    const b: Rat[] = readB.vector
    const method = methodSelect.value as DirectSelectorValue
    emptyHintEl.hidden = true

    if (!isSquare(a)) {
      setResultError(
        resultCardEl,
        'Sistema no cuadrado',
        `Los solucionadores directos requieren una matriz cuadrada A (n×n). Tu matriz A es de ${a.length}×${a[0].length}. Iguala filas y columnas, o usa RREF en la pestaña «Matrices y operaciones» para sistemas rectangulares.`,
      )
      return
    }

    const n = a.length

    if (method === 'gauss') {
      try {
        const res = gaussSolve(a, b)
        const body = createResultFrame(
          resultCardEl,
          'Eliminación de Gauss con pivoteo parcial',
          `${n}×${n} · Exacto · ${res.pasos.length} paso${res.pasos.length === 1 ? '' : 's'}`,
        )
        body.appendChild(createModeBadge(true))
        renderExactSolutionSummary(body, res.x)
        body.appendChild(renderStepsElement(res.pasos))

        const latexCode = [
          LATEX_HEADER,
          '',
          '% Sistema Ax = b',
          `\\[ A = ${matrixToLatex(a)}, \\quad \\mathbf{b} = ${ratVectorToLatex(b)} \\]`,
          '',
          '% Solución exacta por eliminación de Gauss',
          `\\[ \\mathbf{x} = ${ratVectorToLatex(res.x)} \\]`,
          '',
          '% Procedimiento',
          formatStepsLatex(res.pasos),
        ].join('\n')
        body.appendChild(buildLatexCopyBlock('direct-gauss', 'Eliminación de Gauss', latexCode, track))
        track('linealalrl-direct-solve', { method: 'gauss', size: `${n}x${n}` })
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err)
        setResultError(resultCardEl, 'Eliminación de Gauss', msg)
      }
      return
    }

    if (method === 'lu') {
      try {
        const res = luSolve(a, b)
        const body = createResultFrame(
          resultCardEl,
          'Factorización LU con pivoteo (PA = LU)',
          `${n}×${n} · Exacto · ${res.pasos.length} paso${res.pasos.length === 1 ? '' : 's'}`,
        )
        body.appendChild(createModeBadge(true))
        renderExactSolutionSummary(body, res.x)

        const facHeading = document.createElement('h4')
        facHeading.className = 'la-subheading'
        facHeading.textContent = 'Matrices de la factorización (P · A = L · U)'
        body.appendChild(facHeading)

        const facGrid = document.createElement('div')
        facGrid.className = 'la-factors-grid'
        facGrid.appendChild(
          createMathBox(`P = ${matrixToLatex(res.P)}`, 'P', true),
        )
        facGrid.appendChild(
          createMathBox(`L = ${matrixToLatex(res.L)}`, 'L', true),
        )
        facGrid.appendChild(
          createMathBox(`U = ${matrixToLatex(res.U)}`, 'U', true),
        )
        body.appendChild(facGrid)

        const subHeading = document.createElement('h4')
        subHeading.className = 'la-subheading'
        subHeading.textContent = 'Resolución en dos pasos: L·y = P·b y U·x = y'
        body.appendChild(subHeading)

        body.appendChild(
          createMathBox(
            `P\\mathbf{b} = ${ratVectorToLatex(res.pb)}, \\quad \\mathbf{y} = ${ratVectorToLatex(res.y)}, \\quad \\mathbf{x} = ${ratVectorToLatex(res.x)}`,
            `y: ${ratVectorToPlain(res.y, 'y')} | x: ${ratVectorToPlain(res.x, 'x')}`,
            true,
          ),
        )

        body.appendChild(renderStepsElement(res.pasos))

        const latexCode = [
          LATEX_HEADER,
          '',
          '% Factorización PA = LU',
          `\\[ P = ${matrixToLatex(res.P)}, \\quad L = ${matrixToLatex(res.L)}, \\quad U = ${matrixToLatex(res.U)} \\]`,
          '',
          '% Resolución Ly = Pb y Ux = y',
          `\\[ P\\mathbf{b} = ${ratVectorToLatex(res.pb)}, \\quad \\mathbf{y} = ${ratVectorToLatex(res.y)}, \\quad \\mathbf{x} = ${ratVectorToLatex(res.x)} \\]`,
        ].join('\n')
        body.appendChild(buildLatexCopyBlock('direct-lu', 'Factorización LU', latexCode, track))
        track('linealalrl-direct-solve', { method: 'lu', size: `${n}x${n}` })
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err)
        setResultError(resultCardEl, 'Factorización LU', msg)
      }
      return
    }

    if (method === 'cholesky') {
      const res = choleskySolve(a, b)
      if ('error' in res) {
        const body = createResultFrame(
          resultCardEl,
          'Cholesky (A = LLᵀ)',
          `${n}×${n} · No aplicable`,
        )
        const err = document.createElement('div')
        err.className = 'la-error-banner'
        err.setAttribute('role', 'alert')
        err.textContent = res.error
        body.appendChild(err)

        const actions = document.createElement('div')
        actions.className = 'la-inline-actions'
        const switchBtn = document.createElement('button')
        switchBtn.type = 'button'
        switchBtn.className = 'la-btn la-btn--primary la-btn--sm'
        switchBtn.textContent = 'Resolver con factorización LU'
        switchBtn.addEventListener('click', () => {
          methodSelect.value = 'lu'
          runSolve()
        })
        actions.appendChild(switchBtn)
        body.appendChild(actions)
        track('linealalrl-direct-solve', { method: 'cholesky', size: `${n}x${n}`, status: 'na' })
        return
      }

      const body = createResultFrame(
        resultCardEl,
        'Factorización de Cholesky (A = LLᵀ)',
        `${n}×${n} · Numérico aproximado`,
      )
      body.appendChild(createModeBadge(false))
      renderNumericSolutionSummary(body, res.x, res.residuo, 'Residuo ‖A − LLᵀ‖_F', res.residuoFactor)

      const facHeading = document.createElement('h4')
      facHeading.className = 'la-subheading'
      facHeading.textContent = 'Factor triangular inferior L y vector intermedio y (L·y = b, Lᵀ·x = y)'
      body.appendChild(facHeading)

      const facGrid = document.createElement('div')
      facGrid.className = 'la-factors-grid'
      facGrid.appendChild(
        createMathBox(`L \\approx ${numMatrixToLatex(res.L)}`, `L =\n${numMatrixToString(res.L)}`, true),
      )
      facGrid.appendChild(
        createMathBox(`\\mathbf{y} \\approx ${numVectorToLatex(res.y)}`, numVectorToPlain(res.y, 'y'), true),
      )
      body.appendChild(facGrid)

      const latexCode = [
        LATEX_HEADER,
        '',
        '% Factorización de Cholesky (resultado numérico aproximado)',
        `\\[ L \\approx ${numMatrixToLatex(res.L)} \\]`,
        `\\[ \\mathbf{y} \\approx ${numVectorToLatex(res.y)}, \\quad \\mathbf{x} \\approx ${numVectorToLatex(res.x)} \\]`,
        `% Residuo ||Ax - b||_2 = ${formatResidual(res.residuo)}`,
      ].join('\n')
      body.appendChild(buildLatexCopyBlock('direct-cholesky', 'Cholesky', latexCode, track))
      track('linealalrl-direct-solve', { method: 'cholesky', size: `${n}x${n}`, status: 'ok' })
      return
    }

    if (method === 'qr') {
      try {
        const res = qrSolve(a, b)
        const body = createResultFrame(
          resultCardEl,
          'Factorización QR por reflexiones de Householder',
          `${n}×${n} · Numérico aproximado`,
        )
        body.appendChild(createModeBadge(false))
        renderNumericSolutionSummary(body, res.x, res.residuo, 'Residuo ‖A − QR‖_F', res.residuoQR)

        const facHeading = document.createElement('h4')
        facHeading.className = 'la-subheading'
        facHeading.textContent = 'Matrices ortogonal Q y triangular superior R (A = Q·R)'
        body.appendChild(facHeading)

        const facGrid = document.createElement('div')
        facGrid.className = 'la-factors-grid'
        facGrid.appendChild(
          createMathBox(`Q \\approx ${numMatrixToLatex(res.Q)}`, `Q =\n${numMatrixToString(res.Q)}`, true),
        )
        facGrid.appendChild(
          createMathBox(`R \\approx ${numMatrixToLatex(res.R)}`, `R =\n${numMatrixToString(res.R)}`, true),
        )
        body.appendChild(facGrid)

        body.appendChild(
          createMathBox(
            `Q^{\\mathsf{T}}\\mathbf{b} \\approx ${numVectorToLatex(res.qtb)}`,
            numVectorToPlain(res.qtb, 'c'),
            true,
          ),
        )

        const latexCode = [
          LATEX_HEADER,
          '',
          '% Factorización QR por Householder (resultado numérico aproximado)',
          `\\[ Q \\approx ${numMatrixToLatex(res.Q)} \\]`,
          `\\[ R \\approx ${numMatrixToLatex(res.R)} \\]`,
          `\\[ \\mathbf{x} \\approx ${numVectorToLatex(res.x)} \\]`,
          `% Residuo ||Ax - b||_2 = ${formatResidual(res.residuo)}`,
        ].join('\n')
        body.appendChild(buildLatexCopyBlock('direct-qr', 'Factorización QR', latexCode, track))
        track('linealalrl-direct-solve', { method: 'qr', size: `${n}x${n}` })
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err)
        setResultError(resultCardEl, 'Factorización QR (Householder)', msg)
      }
      return
    }

    // Comparar todos
    try {
      const rows = directCompare(a, b)
      const applicableCount = rows.filter((r) => r.aplicable).length
      const body = createResultFrame(
        resultCardEl,
        'Comparador de solucionadores directos',
        `${n}×${n} · ${applicableCount} de ${rows.length} métodos aplican`,
      )
      renderCompareTable(body, rows)
      track('linealalrl-direct-compare', { size: `${n}x${n}` })
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err)
      setResultError(resultCardEl, 'Comparador de solucionadores directos', msg)
    }
  }

  btnSolve.addEventListener('click', runSolve)
}
