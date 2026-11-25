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

// Interfaz interactiva del módulo 0: matrices exactas, pasos KaTeX, LaTeX, presets y compartir.

import { renderLatex } from './katexRender.ts'
import {
  type Matrix,
  type Step,
  det,
  inverse,
  isSquare,
  matrixToLatex,
  matrixToString,
  multiply,
  rank,
  rref,
  solveRref,
  transpose,
} from './matrix.ts'
import { parsePastedAugmented, parsePastedMatrix } from './paste.ts'
import { PRESETS, type MatrixPreset } from './presets.ts'
import { type Rat, parseRat, ratToDecimal, ratToLatex, ratToString } from './rational.ts'
import { MAX_GRID_DIM, type SharedState, buildShareUrl, readStateFromUrl } from './share.ts'

export type TrackFn = (name: string, data?: Record<string, string>) => void

export const INITIAL_VISIBLE_STEPS = 50
const LATEX_HEADER = '% Requiere \\usepackage{amsmath}'

export async function copyText(
  text: string,
  fallbackSelectEl?: HTMLElement | null,
): Promise<'copied' | 'selected' | 'failed'> {
  try {
    if (navigator.clipboard && typeof navigator.clipboard.writeText === 'function') {
      await navigator.clipboard.writeText(text)
      return 'copied'
    }
  } catch {
    // Respaldo con textarea oculto o selección manual.
  }

  try {
    const ta = document.createElement('textarea')
    ta.value = text
    ta.setAttribute('readonly', '')
    ta.style.position = 'fixed'
    ta.style.top = '-9999px'
    document.body.appendChild(ta)
    ta.select()
    const ok = document.execCommand('copy')
    ta.remove()
    if (ok) return 'copied'
  } catch {
    // Continuar al respaldo de selección manual.
  }

  if (fallbackSelectEl && typeof window.getSelection === 'function') {
    try {
      const range = document.createRange()
      range.selectNodeContents(fallbackSelectEl)
      const sel = window.getSelection()
      if (sel) {
        sel.removeAllRanges()
        sel.addRange(range)
        return 'selected'
      }
    } catch {
      // Ignorar si la selección no está disponible.
    }
  }

  return 'failed'
}

export interface MatrixGridHandle {
  element: HTMLElement
  getSize: () => [number, number]
  setValues: (cells: string[][]) => void
  setMatrix: (m: Matrix) => void
  clear: () => void
  readMatrix: () =>
    | { ok: true; matriz: Matrix }
    | { ok: false; error: string; firstInvalid: HTMLInputElement }
  tryReadMatrix: () => Matrix | null
}

function createDimSelect(id: string, labelText: string, initial: number): {
  wrapper: HTMLLabelElement
  select: HTMLSelectElement
} {
  const wrapper = document.createElement('label')
  wrapper.className = 'la-dim-label'
  wrapper.htmlFor = id

  const span = document.createElement('span')
  span.textContent = labelText

  const select = document.createElement('select')
  select.id = id
  select.className = 'la-select la-select--compact'
  for (let n = 1; n <= MAX_GRID_DIM; n++) {
    const opt = document.createElement('option')
    opt.value = String(n)
    opt.textContent = String(n)
    if (n === initial) opt.selected = true
    select.appendChild(opt)
  }

  wrapper.appendChild(span)
  wrapper.appendChild(select)
  return { wrapper, select }
}

/**
 * Componente reutilizable de rejilla matricial editable (1×1 hasta 6×6).
 * Se usa tanto para la matriz A como para la matriz B en el producto matricial.
 */
export function createMatrixGrid(options: {
  idPrefix: string
  matrixName: string
  initialValues: string[][]
  onSizeChange?: (rows: number, cols: number) => void
  onInputChange?: () => void
}): MatrixGridHandle {
  const { idPrefix, matrixName, onSizeChange, onInputChange } = options
  let rows = Math.max(1, Math.min(MAX_GRID_DIM, options.initialValues.length))
  let cols = Math.max(1, Math.min(MAX_GRID_DIM, options.initialValues[0]?.length ?? 1))
  let values: string[][] = Array.from({ length: rows }, (_, i) =>
    Array.from({ length: cols }, (_, j) => options.initialValues[i]?.[j] ?? '0'),
  )
  let inputs: HTMLInputElement[][] = []

  const root = document.createElement('div')
  root.className = 'la-grid-component'

  const controls = document.createElement('div')
  controls.className = 'la-grid-controls'

  const rowControl = createDimSelect(`${idPrefix}-rows`, 'Filas', rows)
  const timesSep = document.createElement('span')
  timesSep.className = 'la-dim-sep'
  timesSep.setAttribute('aria-hidden', 'true')
  timesSep.textContent = '×'
  const colControl = createDimSelect(`${idPrefix}-cols`, 'Columnas', cols)

  const clearBtn = document.createElement('button')
  clearBtn.type = 'button'
  clearBtn.className = 'la-btn la-btn--sm'
  clearBtn.textContent = 'Limpiar'
  clearBtn.setAttribute('aria-label', `Limpiar matriz ${matrixName}`)

  controls.appendChild(rowControl.wrapper)
  controls.appendChild(timesSep)
  controls.appendChild(colControl.wrapper)
  controls.appendChild(clearBtn)

  const matrixWrap = document.createElement('div')
  matrixWrap.className = 'la-matrix-scroll'

  const bracket = document.createElement('div')
  bracket.className = 'la-matrix-bracket'

  const gridEl = document.createElement('div')
  gridEl.className = 'la-matrix-grid'
  bracket.appendChild(gridEl)
  matrixWrap.appendChild(bracket)

  root.appendChild(controls)
  root.appendChild(matrixWrap)

  function syncValuesFromDom(): void {
    for (let i = 0; i < inputs.length; i++) {
      for (let j = 0; j < inputs[i].length; j++) {
        if (!values[i]) values[i] = []
        values[i][j] = inputs[i][j].value
      }
    }
  }

  function renderGrid(): void {
    gridEl.innerHTML = ''
    gridEl.style.gridTemplateColumns = `repeat(${cols}, minmax(3.6rem, 4.75rem))`
    inputs = []

    for (let i = 0; i < rows; i++) {
      const rowInputs: HTMLInputElement[] = []
      for (let j = 0; j < cols; j++) {
        const input = document.createElement('input')
        input.type = 'text'
        input.className = 'la-cell-input'
        input.value = values[i]?.[j] ?? '0'
        input.spellcheck = false
        input.autocomplete = 'off'
        input.setAttribute('aria-label', `Matriz ${matrixName}, fila ${i + 1}, columna ${j + 1}`)
        input.setAttribute('aria-invalid', 'false')

        input.addEventListener('input', () => {
          values[i][j] = input.value
          if (input.classList.contains('is-invalid') && parseRat(input.value) !== null) {
            input.classList.remove('is-invalid')
            input.setAttribute('aria-invalid', 'false')
          }
          onInputChange?.()
        })

        gridEl.appendChild(input)
        rowInputs.push(input)
      }
      inputs.push(rowInputs)
    }

    rowControl.select.value = String(rows)
    colControl.select.value = String(cols)
  }

  function resize(newRows: number, newCols: number): void {
    syncValuesFromDom()
    const nextRows = Math.max(1, Math.min(MAX_GRID_DIM, newRows))
    const nextCols = Math.max(1, Math.min(MAX_GRID_DIM, newCols))
    const nextValues: string[][] = Array.from({ length: nextRows }, (_, i) =>
      Array.from({ length: nextCols }, (_, j) => values[i]?.[j] ?? '0'),
    )
    rows = nextRows
    cols = nextCols
    values = nextValues
    renderGrid()
    onSizeChange?.(rows, cols)
    onInputChange?.()
  }

  rowControl.select.addEventListener('change', () => {
    resize(Number(rowControl.select.value), cols)
  })

  colControl.select.addEventListener('change', () => {
    resize(rows, Number(colControl.select.value))
  })

  clearBtn.addEventListener('click', () => {
    values = Array.from({ length: rows }, () => Array.from({ length: cols }, () => '0'))
    renderGrid()
    onInputChange?.()
  })

  renderGrid()

  return {
    element: root,
    getSize: () => [rows, cols],
    setValues(cells: string[][]): void {
      if (cells.length === 0 || cells[0].length === 0) return
      rows = Math.max(1, Math.min(MAX_GRID_DIM, cells.length))
      cols = Math.max(1, Math.min(MAX_GRID_DIM, cells[0].length))
      values = Array.from({ length: rows }, (_, i) =>
        Array.from({ length: cols }, (_, j) => cells[i]?.[j] ?? '0'),
      )
      renderGrid()
      onSizeChange?.(rows, cols)
    },
    setMatrix(m: Matrix): void {
      this.setValues(m.map((row) => row.map(ratToString)))
    },
    clear(): void {
      values = Array.from({ length: rows }, () => Array.from({ length: cols }, () => '0'))
      renderGrid()
    },
    readMatrix() {
      syncValuesFromDom()
      const parsed: Matrix = []
      let firstInvalid: HTMLInputElement | null = null
      const invalidCoords: string[] = []

      for (let i = 0; i < rows; i++) {
        const row: Rat[] = []
        for (let j = 0; j < cols; j++) {
          const input = inputs[i][j]
          const val = parseRat(input.value)
          if (val === null) {
            input.classList.add('is-invalid')
            input.setAttribute('aria-invalid', 'true')
            if (!firstInvalid) firstInvalid = input
            invalidCoords.push(`(${i + 1}, ${j + 1})`)
          } else {
            input.classList.remove('is-invalid')
            input.setAttribute('aria-invalid', 'false')
            row.push(val)
          }
        }
        parsed.push(row)
      }

      if (firstInvalid) {
        const detalle =
          invalidCoords.length === 1
            ? `la celda ${invalidCoords[0]}`
            : `las celdas ${invalidCoords.slice(0, 4).join(', ')}${invalidCoords.length > 4 ? '…' : ''}`
        return {
          ok: false,
          firstInvalid,
          error: `Revisa ${detalle} en la matriz ${matrixName}: escribe enteros, decimales o fracciones como 3/4 (sin denominador 0).`,
        }
      }

      return { ok: true, matriz: parsed }
    },
    tryReadMatrix(): Matrix | null {
      syncValuesFromDom()
      const parsed: Matrix = []
      for (let i = 0; i < rows; i++) {
        const row: Rat[] = []
        for (let j = 0; j < cols; j++) {
          const val = parseRat(values[i]?.[j] ?? '')
          if (val === null) return null
          row.push(val)
        }
        parsed.push(row)
      }
      return parsed
    },
  }
}

export function getInverseEmptyStepsNote(hasInverse: boolean): string {
  return hasInverse
    ? 'La matriz ya es la identidad, por lo que su inversa es ella misma (0 pasos).'
    : 'Gauss-Jordan no necesitó operaciones: el lado izquierdo ya está reducido y tiene una fila de ceros, así que A es singular.'
}

export function formatFreeVariables(libres: readonly number[]): {
  tex: string
  plain: string
} {
  return {
    tex: libres.map((idx) => `x_${idx + 1}`).join(', '),
    plain: libres.map((idx) => `x${idx + 1}`).join(', '),
  }
}

export function filterTopLevelDetSteps(pasos: Step[], n: number): Step[] {
  if (n < 5) return pasos
  return pasos.filter(
    (step) =>
      step.matriz.length === n ||
      (step.matriz.length === n - 1 && step.texto.startsWith('Cofactor ')),
  )
}

export function formatStepsLatex(pasos: Step[]): string {
  if (pasos.length === 0) return '% Sin pasos adicionales.'
  return pasos
    .map((p, idx) => `% Paso ${idx + 1}: ${p.texto}\n\\[ ${matrixToLatex(p.matriz)} \\]`)
    .join('\n\n')
}

function appendStepItems(list: HTMLOListElement, pasos: Step[], from: number, to: number): void {
  const end = Math.min(to, pasos.length)
  for (let i = from; i < end; i++) {
    const step = pasos[i]
    const li = document.createElement('li')
    li.className = 'la-step-item'

    const textEl = document.createElement('div')
    textEl.className = 'la-step-text'
    textEl.textContent = step.texto

    const matEl = document.createElement('div')
    matEl.className = 'la-step-matrix'
    void renderLatex(matrixToLatex(step.matriz), matEl, {
      displayMode: true,
      fallbackText: matrixToString(step.matriz),
    })

    li.appendChild(textEl)
    li.appendChild(matEl)
    list.appendChild(li)
  }
}

export function renderStepsElement(pasos: Step[], emptyNote?: string): HTMLElement {
  const wrap = document.createElement('div')
  wrap.className = 'la-steps-wrap'

  const heading = document.createElement('h4')
  heading.className = 'la-subheading'
  heading.textContent = 'Procedimiento paso a paso'
  wrap.appendChild(heading)

  if (pasos.length === 0) {
    const note = document.createElement('p')
    note.className = 'la-note'
    note.textContent = emptyNote ?? 'La matriz ya se encuentra en la forma final (0 pasos).'
    wrap.appendChild(note)
    return wrap
  }

  const list = document.createElement('ol')
  list.className = 'la-steps-list'

  const initialCount = Math.min(pasos.length, INITIAL_VISIBLE_STEPS)
  appendStepItems(list, pasos, 0, initialCount)
  wrap.appendChild(list)

  if (pasos.length > initialCount) {
    const remaining = pasos.length - initialCount
    const moreBar = document.createElement('div')
    moreBar.className = 'la-steps-more'

    const moreBtn = document.createElement('button')
    moreBtn.type = 'button'
    moreBtn.className = 'la-btn la-btn--sm'
    moreBtn.textContent =
      remaining === 1 ? 'Mostrar el paso restante' : `Mostrar los ${remaining} pasos restantes`

    const moreInfo = document.createElement('span')
    moreInfo.className = 'la-note'
    moreInfo.textContent = `Mostrando ${initialCount} de ${pasos.length} pasos.`

    moreBtn.addEventListener('click', () => {
      moreBtn.disabled = true
      let cursor = initialCount
      const batchSize = INITIAL_VISIBLE_STEPS
      const renderBatch = (): void => {
        const next = Math.min(cursor + batchSize, pasos.length)
        appendStepItems(list, pasos, cursor, next)
        cursor = next
        if (cursor < pasos.length) {
          moreInfo.textContent = `Cargando pasos (${cursor} de ${pasos.length})…`
          if (typeof window !== 'undefined' && typeof window.requestAnimationFrame === 'function') {
            window.requestAnimationFrame(renderBatch)
          } else {
            renderBatch()
          }
        } else {
          moreBar.remove()
        }
      }
      renderBatch()
    })

    moreBar.appendChild(moreBtn)
    moreBar.appendChild(moreInfo)
    wrap.appendChild(moreBar)
  }

  return wrap
}

export function buildLatexCopyBlock(
  opKey: string,
  opTitle: string,
  latexCode: string,
  track: TrackFn,
): HTMLElement {
  const section = document.createElement('div')
  section.className = 'la-latex-block'

  const head = document.createElement('div')
  head.className = 'la-latex-head'

  const label = document.createElement('span')
  label.className = 'la-latex-title'
  label.textContent = 'Código LaTeX'

  const actions = document.createElement('div')
  actions.className = 'la-latex-actions'

  const feedback = document.createElement('span')
  feedback.className = 'la-feedback'
  feedback.setAttribute('role', 'status')
  feedback.setAttribute('aria-live', 'polite')
  feedback.hidden = true

  const btn = document.createElement('button')
  btn.type = 'button'
  btn.className = 'la-btn la-btn--sm la-btn--primary'
  btn.textContent = 'Copiar LaTeX'
  btn.setAttribute('aria-label', `Copiar LaTeX de ${opTitle}`)

  const pre = document.createElement('pre')
  pre.className = 'la-latex-code'
  const code = document.createElement('code')
  code.textContent = latexCode
  pre.appendChild(code)

  let timer: number | null = null
  btn.addEventListener('click', async () => {
    const outcome = await copyText(latexCode, code)
    track('linealalrl-copy-latex', { op: opKey })
    feedback.hidden = false
    if (outcome === 'copied') {
      feedback.textContent = 'LaTeX copiado'
      feedback.classList.remove('la-feedback--error')
    } else if (outcome === 'selected') {
      feedback.textContent = 'Seleccionado: pulsa Ctrl+C para copiar'
      feedback.classList.remove('la-feedback--error')
    } else {
      feedback.textContent = 'No se pudo copiar automáticamente'
      feedback.classList.add('la-feedback--error')
    }
    if (timer !== null) window.clearTimeout(timer)
    timer = window.setTimeout(() => {
      feedback.hidden = true
    }, 2800)
  })

  actions.appendChild(feedback)
  actions.appendChild(btn)
  head.appendChild(label)
  head.appendChild(actions)
  section.appendChild(head)
  section.appendChild(pre)
  return section
}

export function setResultError(container: HTMLElement, title: string, message: string): void {
  container.hidden = false
  container.classList.remove('is-stale')
  container.innerHTML = ''

  const head = document.createElement('div')
  head.className = 'la-result-head'

  const h3 = document.createElement('h3')
  h3.className = 'la-result-title'
  h3.textContent = title

  const closeBtn = document.createElement('button')
  closeBtn.type = 'button'
  closeBtn.className = 'la-btn la-btn--sm'
  closeBtn.textContent = 'Ocultar'
  closeBtn.setAttribute('aria-label', `Ocultar resultado de ${title}`)
  closeBtn.addEventListener('click', () => {
    container.hidden = true
    container.classList.remove('is-stale')
  })

  head.appendChild(h3)
  head.appendChild(closeBtn)

  const err = document.createElement('div')
  err.className = 'la-error-banner'
  err.setAttribute('role', 'alert')
  err.textContent = message

  container.appendChild(head)
  container.appendChild(err)
}

export function createResultFrame(
  container: HTMLElement,
  title: string,
  meta: string,
): HTMLDivElement {
  container.hidden = false
  container.classList.remove('is-stale')
  container.innerHTML = ''

  const head = document.createElement('div')
  head.className = 'la-result-head'

  const titleGroup = document.createElement('div')
  titleGroup.className = 'la-result-title-group'

  const h3 = document.createElement('h3')
  h3.className = 'la-result-title'
  h3.textContent = title

  const metaEl = document.createElement('span')
  metaEl.className = 'la-panel-meta'
  metaEl.textContent = meta

  titleGroup.appendChild(h3)
  titleGroup.appendChild(metaEl)

  const closeBtn = document.createElement('button')
  closeBtn.type = 'button'
  closeBtn.className = 'la-btn la-btn--sm'
  closeBtn.textContent = 'Ocultar'
  closeBtn.setAttribute('aria-label', `Ocultar resultado de ${title}`)
  closeBtn.addEventListener('click', () => {
    container.hidden = true
    container.classList.remove('is-stale')
  })

  head.appendChild(titleGroup)
  head.appendChild(closeBtn)

  const body = document.createElement('div')
  body.className = 'la-result-body'

  container.appendChild(head)
  container.appendChild(body)
  return body
}

export function buildProductExampleFormula(a: Matrix, b: Matrix, prod: Matrix): {
  tex: string
  plain: string
} {
  const k = a[0].length
  const symParts: string[] = []
  const numParts: string[] = []
  const plainParts: string[] = []

  for (let t = 0; t < k; t++) {
    symParts.push(`a_{1${t + 1}} b_{${t + 1}1}`)
    numParts.push(`(${ratToLatex(a[0][t])})(${ratToLatex(b[t][0])})`)
    plainParts.push(`(${ratToString(a[0][t])})(${ratToString(b[t][0])})`)
  }

  const c11Tex = ratToLatex(prod[0][0])
  const c11Plain = ratToString(prod[0][0])
  return {
    tex: `c_{11} = ${symParts.join(' + ')} = ${numParts.join(' + ')} = ${c11Tex}`,
    plain: `c11 = ${plainParts.join(' + ')} = ${c11Plain}`,
  }
}

export function mountMatrixUI(app: HTMLElement, track: TrackFn): void {
  const initialHash = typeof window !== 'undefined' && window.location ? window.location.hash : ''
  const hasInitialShareHash = initialHash.startsWith('#m=')
  const fromUrl = readStateFromUrl()
  const defaultPreset = PRESETS.find((p) => p.id === 'invertible-3x3')!

  const initialA: string[][] = fromUrl
    ? fromUrl.a.map((row) => row.map(ratToString))
    : defaultPreset.a
  const initialB: string[][] = fromUrl?.b
    ? fromUrl.b.map((row) => row.map(ratToString))
    : [
        ['1', '0'],
        ['0', '1'],
        ['1', '1'],
      ]
  let vectorValues: string[] = fromUrl?.vector
    ? fromUrl.vector.map(ratToString)
    : Array.from({ length: initialA.length }, (_, i) => String(i + 1))

  let activePresetId: string | null = fromUrl ? null : defaultPreset.id
  let urlNotice: 'loaded' | 'invalid' | null = fromUrl
    ? 'loaded'
    : hasInitialShareHash
      ? 'invalid'
      : null

  app.innerHTML = `
    <div class="la-workspace">
      <section class="la-panel">
        <div class="la-panel-head">
          <h2 class="la-panel-title">Matriz A</h2>
          <div class="la-head-actions">
            <span class="la-feedback" id="share-feedback" role="status" aria-live="polite" hidden></span>
            <button class="la-btn la-btn--sm la-btn--gold" id="btn-share" type="button">Compartir</button>
          </div>
        </div>

        <div class="la-toolbar" id="presets-bar">
          <span class="la-toolbar-label">Presets:</span>
        </div>
        <p class="la-preset-desc" id="preset-desc"></p>

        <div class="la-panel-body la-matrix-a-layout">
          <div class="la-matrix-a-editor">
            <div id="grid-a-mount"></div>
            <div class="la-error-banner" id="error-a" role="alert" hidden></div>
          </div>

          <div class="la-paste-box">
            <label class="la-field-label" for="paste-a-input">Pegar matriz desde el portapapeles</label>
            <p class="la-hint" id="paste-a-hint">Separa valores con tabulador, punto y coma (;), coma o espacio, y filas con salto de línea (hasta 6×6).</p>
            <textarea
              id="paste-a-input"
              class="la-textarea"
              rows="4"
              spellcheck="false"
              aria-describedby="paste-a-hint"
              placeholder="2 -1 0&#10;1 3 2&#10;4 0 -1"
            ></textarea>
            <div class="la-paste-actions">
              <button class="la-btn la-btn--sm la-btn--primary" id="btn-paste-a" type="button">Pegar matriz</button>
              <span class="la-feedback" id="paste-a-feedback" role="status" aria-live="polite" hidden></span>
            </div>
            <div class="la-error-banner" id="error-paste-a" role="alert" hidden></div>
          </div>
        </div>
      </section>

      <section class="la-panel">
        <div class="la-panel-head">
          <h2 class="la-panel-title">Operaciones sobre A</h2>
          <span class="la-panel-meta">Aritmética exacta con fracciones</span>
        </div>

        <div class="la-toolbar la-ops-bar">
          <button class="la-btn la-btn--primary" id="btn-op-det" type="button">Determinante</button>
          <button class="la-btn la-btn--primary" id="btn-op-inv" type="button">Inversa</button>
          <button class="la-btn la-btn--primary" id="btn-op-rref" type="button">RREF</button>
          <button class="la-btn la-btn--primary" id="btn-op-tr" type="button">Transpuesta y rango</button>
        </div>

        <div class="la-panel-body la-results-stack">
          <p class="la-note" id="ops-empty-hint">
            Elige una operación sobre la matriz A para ver el resultado exacto, los pasos en KaTeX y el código LaTeX listo para copiar.
          </p>
          <article class="la-result-card" id="result-det" hidden></article>
          <article class="la-result-card" id="result-inv" hidden></article>
          <article class="la-result-card" id="result-rref" hidden></article>
          <article class="la-result-card" id="result-tr" hidden></article>
        </div>
      </section>

      <div class="la-two-col">
        <section class="la-panel">
          <div class="la-panel-head">
            <h2 class="la-panel-title">Producto matricial (A · B)</h2>
            <span class="la-dim-pill" id="mul-dim-pill"></span>
          </div>

          <div class="la-panel-body la-stack">
            <div id="grid-b-mount"></div>
            <div class="la-inline-actions">
              <button class="la-btn la-btn--primary" id="btn-op-mul" type="button">Calcular producto A · B</button>
            </div>
            <article class="la-result-card" id="result-mul" hidden></article>
          </div>
        </section>

        <section class="la-panel">
          <div class="la-panel-head">
            <h2 class="la-panel-title">Sistema lineal (Ax = b)</h2>
            <span class="la-panel-meta" id="sys-dim-meta"></span>
          </div>

          <div class="la-panel-body la-stack">
            <div class="la-vector-section">
              <div class="la-vector-head">
                <span class="la-field-label">Vector b (términos independientes)</span>
                <button class="la-btn la-btn--sm" id="btn-clear-b-vec" type="button" aria-label="Limpiar vector b">Limpiar b</button>
              </div>
              <div class="la-matrix-scroll">
                <div class="la-matrix-bracket">
                  <div class="la-vector-grid" id="vector-b-grid"></div>
                </div>
              </div>
            </div>

            <div class="la-paste-box la-paste-box--compact">
              <label class="la-field-label" for="paste-aug-input">O pegar matriz aumentada [A | b]</label>
              <p class="la-hint" id="paste-aug-hint">La última columna se carga como vector b (puedes usar «|» como separador opcional).</p>
              <textarea
                id="paste-aug-input"
                class="la-textarea"
                rows="3"
                spellcheck="false"
                aria-describedby="paste-aug-hint"
                placeholder="1 2 -1 | -4&#10;2 3 -1 | -11&#10;-2 0 -3 | 22"
              ></textarea>
              <div class="la-paste-actions">
                <button class="la-btn la-btn--sm" id="btn-paste-aug" type="button">Pegar aumentada [A | b]</button>
                <span class="la-feedback" id="paste-aug-feedback" role="status" aria-live="polite" hidden></span>
              </div>
              <div class="la-error-banner" id="error-paste-aug" role="alert" hidden></div>
            </div>

            <div class="la-inline-actions">
              <button class="la-btn la-btn--primary" id="btn-op-sys" type="button">Resolver Ax = b</button>
            </div>
            <article class="la-result-card" id="result-sys" hidden></article>
          </div>
        </section>
      </div>
    </div>
  `

  // Referencias del DOM
  const presetsBar = app.querySelector<HTMLDivElement>('#presets-bar')!
  const presetDescEl = app.querySelector<HTMLParagraphElement>('#preset-desc')!
  const gridAMount = app.querySelector<HTMLDivElement>('#grid-a-mount')!
  const gridBMount = app.querySelector<HTMLDivElement>('#grid-b-mount')!
  const errorAEl = app.querySelector<HTMLDivElement>('#error-a')!
  const btnShare = app.querySelector<HTMLButtonElement>('#btn-share')!
  const shareFeedbackEl = app.querySelector<HTMLSpanElement>('#share-feedback')!

  const pasteAInput = app.querySelector<HTMLTextAreaElement>('#paste-a-input')!
  const btnPasteA = app.querySelector<HTMLButtonElement>('#btn-paste-a')!
  const pasteAFeedbackEl = app.querySelector<HTMLSpanElement>('#paste-a-feedback')!
  const errorPasteAEl = app.querySelector<HTMLDivElement>('#error-paste-a')!

  const opsEmptyHint = app.querySelector<HTMLParagraphElement>('#ops-empty-hint')!
  const btnOpDet = app.querySelector<HTMLButtonElement>('#btn-op-det')!
  const btnOpInv = app.querySelector<HTMLButtonElement>('#btn-op-inv')!
  const btnOpRref = app.querySelector<HTMLButtonElement>('#btn-op-rref')!
  const btnOpTr = app.querySelector<HTMLButtonElement>('#btn-op-tr')!
  const btnOpMul = app.querySelector<HTMLButtonElement>('#btn-op-mul')!
  const btnOpSys = app.querySelector<HTMLButtonElement>('#btn-op-sys')!

  const resultDetEl = app.querySelector<HTMLElement>('#result-det')!
  const resultInvEl = app.querySelector<HTMLElement>('#result-inv')!
  const resultRrefEl = app.querySelector<HTMLElement>('#result-rref')!
  const resultTrEl = app.querySelector<HTMLElement>('#result-tr')!
  const resultMulEl = app.querySelector<HTMLElement>('#result-mul')!
  const resultSysEl = app.querySelector<HTMLElement>('#result-sys')!

  const mulDimPill = app.querySelector<HTMLSpanElement>('#mul-dim-pill')!
  const sysDimMeta = app.querySelector<HTMLSpanElement>('#sys-dim-meta')!
  const vectorBGrid = app.querySelector<HTMLDivElement>('#vector-b-grid')!
  const btnClearBVec = app.querySelector<HTMLButtonElement>('#btn-clear-b-vec')!
  const pasteAugInput = app.querySelector<HTMLTextAreaElement>('#paste-aug-input')!
  const btnPasteAug = app.querySelector<HTMLButtonElement>('#btn-paste-aug')!
  const pasteAugFeedbackEl = app.querySelector<HTMLSpanElement>('#paste-aug-feedback')!
  const errorPasteAugEl = app.querySelector<HTMLDivElement>('#error-paste-aug')!

  let vectorInputs: HTMLInputElement[] = []
  const allResultCards: HTMLElement[] = [
    resultDetEl,
    resultInvEl,
    resultRrefEl,
    resultTrEl,
    resultMulEl,
    resultSysEl,
  ]

  function markResultsStale(cards: HTMLElement[]): void {
    for (const card of cards) {
      if (card.hidden || card.classList.contains('is-stale')) continue
      card.classList.add('is-stale')
      const head = card.querySelector('.la-result-head')
      const notice = document.createElement('div')
      notice.className = 'la-stale-banner'
      notice.textContent = 'Resultado de una versión anterior de la matriz'
      if (head && head.nextSibling) {
        card.insertBefore(notice, head.nextSibling)
      } else {
        card.appendChild(notice)
      }
    }
  }

  function showTimedFeedback(el: HTMLElement, msg: string, isError = false): void {
    el.textContent = msg
    el.classList.toggle('la-feedback--error', isError)
    el.hidden = false
    const prev = Number(el.dataset.timerId ?? '0')
    if (prev) window.clearTimeout(prev)
    const id = window.setTimeout(() => {
      el.hidden = true
    }, 3000)
    el.dataset.timerId = String(id)
  }

  function showBannerError(el: HTMLElement, msg: string | null): void {
    if (!msg) {
      el.hidden = true
      el.textContent = ''
      return
    }
    el.textContent = msg
    el.hidden = false
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
      input.setAttribute('aria-label', `Vector b, fila ${i + 1}`)
      input.setAttribute('aria-invalid', 'false')

      input.addEventListener('input', () => {
        vectorValues[i] = input.value
        if (input.classList.contains('is-invalid') && parseRat(input.value) !== null) {
          input.classList.remove('is-invalid')
          input.setAttribute('aria-invalid', 'false')
        }
        markResultsStale([resultSysEl])
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

  function tryReadVectorB(): Rat[] | null {
    const out: Rat[] = []
    for (let i = 0; i < vectorInputs.length; i++) {
      const val = parseRat(vectorInputs[i].value)
      if (val === null) return null
      out.push(val)
    }
    return out
  }

  const presetButtons = new Map<string, HTMLButtonElement>()
  function updatePresetButtons(): void {
    for (const [id, btn] of presetButtons.entries()) {
      const active = id === activePresetId
      btn.classList.toggle('la-btn--active', active)
      btn.setAttribute('aria-pressed', active ? 'true' : 'false')
    }
    if (urlNotice === 'invalid') {
      presetDescEl.textContent = 'El enlace compartido no es válido; se cargó el ejemplo por defecto'
      return
    }
    if (urlNotice === 'loaded' && activePresetId === null) {
      presetDescEl.textContent = 'Estado cargado desde el enlace compartido.'
      return
    }
    const preset = PRESETS.find((p) => p.id === activePresetId)
    if (preset) {
      presetDescEl.textContent = preset.descripcion
    } else {
      presetDescEl.textContent = 'Matriz personalizada (admite enteros, decimales y fracciones como 3/4).'
    }
  }

  function updateDimensionMeta(): void {
    const [ra, ca] = gridA.getSize()
    const [rb, cb] = gridB.getSize()
    const compatible = ca === rb
    mulDimPill.textContent = compatible
      ? `A (${ra}×${ca}) · B (${rb}×${cb}) → ${ra}×${cb}`
      : `Incompatible: A (${ra}×${ca}) y B (${rb}×${cb})`
    mulDimPill.classList.toggle('is-compatible', compatible)
    mulDimPill.classList.toggle('is-incompatible', !compatible)

    sysDimMeta.textContent = `A (${ra}×${ca}) · x (${ca}×1) = b (${ra}×1)`
  }

  const gridA = createMatrixGrid({
    idPrefix: 'grid-a',
    matrixName: 'A',
    initialValues: initialA,
    onSizeChange: (rows) => {
      renderVectorB(rows)
      updateDimensionMeta()
    },
    onInputChange: () => {
      activePresetId = null
      urlNotice = null
      showBannerError(errorAEl, null)
      markResultsStale(allResultCards)
      updatePresetButtons()
    },
  })

  const gridB = createMatrixGrid({
    idPrefix: 'grid-b',
    matrixName: 'B',
    initialValues: initialB,
    onSizeChange: () => {
      updateDimensionMeta()
    },
    onInputChange: () => {
      markResultsStale([resultMulEl])
    },
  })

  gridAMount.appendChild(gridA.element)
  gridBMount.appendChild(gridB.element)
  renderVectorB(initialA.length)
  updateDimensionMeta()

  // Botones de presets
  function applyPreset(preset: MatrixPreset): void {
    activePresetId = preset.id
    urlNotice = null
    gridA.setValues(preset.a)
    if (preset.vector) {
      vectorValues = preset.vector.slice()
      renderVectorB(preset.a.length)
    }
    if (preset.b) {
      gridB.setValues(preset.b)
    }
    showBannerError(errorAEl, null)
    showBannerError(errorPasteAEl, null)
    markResultsStale(allResultCards)
    updatePresetButtons()
    updateDimensionMeta()
  }

  function applySharedState(state: SharedState): void {
    activePresetId = null
    urlNotice = 'loaded'
    gridA.setMatrix(state.a)
    if (state.b) {
      gridB.setMatrix(state.b)
    }
    if (state.vector) {
      vectorValues = state.vector.map(ratToString)
      renderVectorB(state.a.length)
    }
    showBannerError(errorAEl, null)
    showBannerError(errorPasteAEl, null)
    showBannerError(errorPasteAugEl, null)
    markResultsStale(allResultCards)
    updatePresetButtons()
    updateDimensionMeta()
  }

  if (typeof window !== 'undefined') {
    window.addEventListener('hashchange', () => {
      const hash = window.location.hash
      if (!hash.startsWith('#m=')) return
      const state = readStateFromUrl()
      if (state) {
        applySharedState(state)
      } else {
        applyPreset(defaultPreset)
        urlNotice = 'invalid'
        updatePresetButtons()
      }
    })
  }

  for (const preset of PRESETS) {
    const btn = document.createElement('button')
    btn.type = 'button'
    btn.className = 'la-btn la-btn--sm'
    btn.textContent = preset.nombre
    btn.addEventListener('click', () => {
      applyPreset(preset)
      track('linealalrl-preset', { preset: preset.id })
    })
    presetsBar.appendChild(btn)
    presetButtons.set(preset.id, btn)
  }
  updatePresetButtons()

  // Pegar matriz A
  btnPasteA.addEventListener('click', () => {
    const parsed = parsePastedMatrix(pasteAInput.value, MAX_GRID_DIM, MAX_GRID_DIM)
    if (!parsed.ok) {
      showBannerError(errorPasteAEl, parsed.error)
      pasteAInput.focus()
      return
    }
    showBannerError(errorPasteAEl, null)
    showBannerError(errorAEl, null)
    activePresetId = null
    urlNotice = null
    gridA.setMatrix(parsed.matriz)
    markResultsStale(allResultCards)
    updatePresetButtons()
    updateDimensionMeta()
    showTimedFeedback(pasteAFeedbackEl, `Matriz A actualizada (${parsed.filas}×${parsed.cols})`)
    track('linealalrl-paste', { target: 'A', size: `${parsed.filas}x${parsed.cols}` })
  })

  // Pegar matriz aumentada [A | b]
  btnPasteAug.addEventListener('click', () => {
    const parsed = parsePastedAugmented(pasteAugInput.value, MAX_GRID_DIM, MAX_GRID_DIM)
    if (!parsed.ok) {
      showBannerError(errorPasteAugEl, parsed.error)
      pasteAugInput.focus()
      return
    }
    showBannerError(errorPasteAugEl, null)
    showBannerError(errorAEl, null)
    activePresetId = null
    urlNotice = null
    vectorValues = parsed.vector.map(ratToString)
    gridA.setMatrix(parsed.a)
    renderVectorB(parsed.filas)
    markResultsStale(allResultCards)
    updatePresetButtons()
    updateDimensionMeta()
    showTimedFeedback(
      pasteAugFeedbackEl,
      `Cargada [A | b] (${parsed.filas}×${parsed.cols} + vector b)`,
    )
    track('linealalrl-paste', { target: 'augmented', size: `${parsed.filas}x${parsed.cols}` })
  })

  btnClearBVec.addEventListener('click', () => {
    vectorValues = vectorValues.map(() => '0')
    renderVectorB(vectorValues.length)
    markResultsStale([resultSysEl])
  })

  // Compartir estado por URL (#m=...)
  btnShare.addEventListener('click', async () => {
    const readA = gridA.readMatrix()
    if (!readA.ok) {
      showBannerError(errorAEl, readA.error)
      readA.firstInvalid.focus()
      return
    }
    showBannerError(errorAEl, null)
    const bMat = gridB.tryReadMatrix()
    const vec = tryReadVectorB()
    const url = buildShareUrl(readA.matriz, bMat, vec)
    try {
      window.history.replaceState(null, '', url)
    } catch {
      // Ignorar si replaceState está restringido en el entorno.
    }
    const outcome = await copyText(url)
    track('linealalrl-share')
    if (outcome === 'copied') {
      showTimedFeedback(shareFeedbackEl, 'Enlace copiado al portapapeles')
    } else {
      showTimedFeedback(shareFeedbackEl, 'Enlace actualizado en la barra de direcciones')
    }
  })

  function readValidMatrixA(errorTarget?: {
    container: HTMLElement
    title: string
  }): Matrix | null {
    const res = gridA.readMatrix()
    if (!res.ok) {
      showBannerError(errorAEl, res.error)
      if (errorTarget) {
        setResultError(errorTarget.container, errorTarget.title, res.error)
      }
      res.firstInvalid.focus()
      return null
    }
    showBannerError(errorAEl, null)
    return res.matriz
  }

  // 1. Determinante
  btnOpDet.addEventListener('click', () => {
    const a = readValidMatrixA()
    if (!a) return
    opsEmptyHint.hidden = true

    if (!isSquare(a)) {
      setResultError(
        resultDetEl,
        'Determinante',
        `El determinante solo existe para matrices cuadradas. Tu matriz A es de ${a.length}×${a[0].length}; iguala el número de filas y columnas para calcularlo.`,
      )
      return
    }

    const res = det(a)
    const body = createResultFrame(
      resultDetEl,
      'Determinante',
      `${a.length}×${a.length} · ${res.pasos.length} paso${res.pasos.length === 1 ? '' : 's'}`,
    )

    const summary = document.createElement('div')
    summary.className = 'la-result-math'
    const exactTex = `\\det(A) = ${ratToLatex(res.valor)}`
    void renderLatex(exactTex, summary, {
      displayMode: true,
      fallbackText: `det(A) = ${ratToString(res.valor)}`,
    })
    body.appendChild(summary)

    const valuesRow = document.createElement('p')
    valuesRow.className = 'la-result-meta-line'
    const exactStr = ratToString(res.valor)
    const decStr = ratToDecimal(res.valor, 6)
    valuesRow.textContent =
      res.valor.d === 1n
        ? `Valor exacto: ${exactStr} · Decimal: ${decStr}`
        : `Valor exacto: ${exactStr} · Aproximación decimal: ≈ ${decStr}`
    body.appendChild(valuesRow)

    body.appendChild(renderStepsElement(res.pasos))

    const detStepsForLatex = filterTopLevelDetSteps(res.pasos, a.length)
    const omittedSteps = res.pasos.length - detStepsForLatex.length
    const detStepsComment =
      omittedSteps > 0
        ? `% Procedimiento (pasos de primer nivel; se omiten ${omittedSteps} sub-pasos de menores en matriz ${a.length}×${a.length})`
        : '% Procedimiento (expansión por cofactores)'

    const latexCode = [
      LATEX_HEADER,
      '',
      '% Matriz de entrada',
      `\\[ A = ${matrixToLatex(a)} \\]`,
      '',
      '% Resultado',
      `\\[ \\det(A) = ${ratToLatex(res.valor)} \\]`,
      '',
      detStepsComment,
      formatStepsLatex(detStepsForLatex),
    ].join('\n')

    body.appendChild(buildLatexCopyBlock('det', 'Determinante', latexCode, track))
    track('linealalrl-solve', { op: 'det', size: `${a.length}x${a.length}` })
  })

  // 2. Inversa
  btnOpInv.addEventListener('click', () => {
    const a = readValidMatrixA()
    if (!a) return
    opsEmptyHint.hidden = true

    if (!isSquare(a)) {
      setResultError(
        resultInvEl,
        'Inversa',
        `La inversa solo está definida para matrices cuadradas. Tu matriz A es de ${a.length}×${a[0].length}; ajusta filas y columnas al mismo valor.`,
      )
      return
    }

    const res = inverse(a)
    const body = createResultFrame(
      resultInvEl,
      'Inversa',
      res.inversa
        ? `${a.length}×${a.length} · ${res.pasos.length} paso${res.pasos.length === 1 ? '' : 's'}`
        : `${a.length}×${a.length} · Singular`,
    )

    let latexResultSection: string
    if (res.inversa === null) {
      const warn = document.createElement('div')
      warn.className = 'la-error-banner'
      warn.setAttribute('role', 'alert')
      warn.textContent = res.atasco
      body.appendChild(warn)
      latexResultSection = `% Matriz singular (sin inversa)\n% ${res.atasco}`
    } else {
      const mathBox = document.createElement('div')
      mathBox.className = 'la-result-math'
      void renderLatex(`A^{-1} = ${matrixToLatex(res.inversa)}`, mathBox, {
        displayMode: true,
        fallbackText: `A^-1 =\n${matrixToString(res.inversa)}`,
      })
      body.appendChild(mathBox)
      latexResultSection = `\\[ A^{-1} = ${matrixToLatex(res.inversa)} \\]`
    }

    body.appendChild(
      renderStepsElement(res.pasos, getInverseEmptyStepsNote(res.inversa !== null)),
    )

    const latexCode = [
      LATEX_HEADER,
      '',
      '% Matriz de entrada',
      `\\[ A = ${matrixToLatex(a)} \\]`,
      '',
      '% Resultado',
      latexResultSection,
      '',
      '% Procedimiento (Gauss-Jordan sobre [A | I])',
      formatStepsLatex(res.pasos),
    ].join('\n')

    body.appendChild(buildLatexCopyBlock('inverse', 'Inversa', latexCode, track))
    track('linealalrl-solve', {
      op: 'inverse',
      size: `${a.length}x${a.length}`,
      singular: res.inversa === null ? 'true' : 'false',
    })
  })

  // 3. RREF
  btnOpRref.addEventListener('click', () => {
    const a = readValidMatrixA()
    if (!a) return
    opsEmptyHint.hidden = true

    const res = rref(a)
    const body = createResultFrame(
      resultRrefEl,
      'Forma escalonada reducida (RREF)',
      `${a.length}×${a[0].length} · ${res.pasos.length} paso${res.pasos.length === 1 ? '' : 's'}`,
    )

    const mathBox = document.createElement('div')
    mathBox.className = 'la-result-math'
    void renderLatex(`\\operatorname{rref}(A) = ${matrixToLatex(res.resultado)}`, mathBox, {
      displayMode: true,
      fallbackText: `rref(A) =\n${matrixToString(res.resultado)}`,
    })
    body.appendChild(mathBox)

    body.appendChild(
      renderStepsElement(
        res.pasos,
        'La matriz ya estaba en forma escalonada reducida por filas (0 pasos).',
      ),
    )

    const latexCode = [
      LATEX_HEADER,
      '',
      '% Matriz de entrada',
      `\\[ A = ${matrixToLatex(a)} \\]`,
      '',
      '% Resultado',
      `\\[ \\operatorname{rref}(A) = ${matrixToLatex(res.resultado)} \\]`,
      '',
      '% Procedimiento (Gauss-Jordan)',
      formatStepsLatex(res.pasos),
    ].join('\n')

    body.appendChild(buildLatexCopyBlock('rref', 'RREF', latexCode, track))
    track('linealalrl-solve', { op: 'rref', size: `${a.length}x${a[0].length}` })
  })

  // 4. Transpuesta y rango
  btnOpTr.addEventListener('click', () => {
    const a = readValidMatrixA()
    if (!a) return
    opsEmptyHint.hidden = true

    const tr = transpose(a)
    const rg = rank(a)
    const body = createResultFrame(
      resultTrEl,
      'Transpuesta y rango',
      `Aᵀ (${tr.length}×${tr[0].length}) · rango = ${rg}`,
    )

    const mathBox = document.createElement('div')
    mathBox.className = 'la-result-math'
    void renderLatex(`A^{\\mathsf{T}} = ${matrixToLatex(tr)}`, mathBox, {
      displayMode: true,
      fallbackText: `A^T =\n${matrixToString(tr)}`,
    })
    body.appendChild(mathBox)

    const rankBox = document.createElement('div')
    rankBox.className = 'la-result-meta-line'
    rankBox.textContent = `Rango(A) = ${rg} (${rg} fila${rg === 1 ? '' : 's'} no nula${rg === 1 ? '' : 's'} en la RREF de A).`
    body.appendChild(rankBox)

    const latexCode = [
      LATEX_HEADER,
      '',
      '% Matriz de entrada',
      `\\[ A = ${matrixToLatex(a)} \\]`,
      '',
      '% Resultado (transpuesta y rango desde la RREF)',
      `\\[ A^{\\mathsf{T}} = ${matrixToLatex(tr)} \\]`,
      `\\[ \\operatorname{rango}(A) = ${rg} \\]`,
    ].join('\n')

    body.appendChild(buildLatexCopyBlock('transpose-rank', 'Transpuesta y rango', latexCode, track))
    track('linealalrl-solve', { op: 'transpose-rank', size: `${a.length}x${a[0].length}` })
  })

  // 5. Producto matricial A · B
  btnOpMul.addEventListener('click', () => {
    const a = readValidMatrixA({ container: resultMulEl, title: 'Producto A · B' })
    if (!a) return

    const readB = gridB.readMatrix()
    if (!readB.ok) {
      setResultError(resultMulEl, 'Producto A · B', readB.error)
      readB.firstInvalid.focus()
      return
    }
    const b = readB.matriz

    if (a[0].length !== b.length) {
      setResultError(
        resultMulEl,
        'Producto A · B',
        `Dimensiones incompatibles: A es ${a.length}×${a[0].length} y B es ${b.length}×${b[0].length}. Para multiplicar A · B, el número de columnas de A (${a[0].length}) debe ser igual al número de filas de B (${b.length}).`,
      )
      return
    }

    const prod = multiply(a, b)
    const example = buildProductExampleFormula(a, b, prod)
    const body = createResultFrame(
      resultMulEl,
      'Producto A · B',
      `${a.length}×${a[0].length} · ${b.length}×${b[0].length} → ${prod.length}×${prod[0].length}`,
    )

    const mathBox = document.createElement('div')
    mathBox.className = 'la-result-math'
    void renderLatex(`A \\cdot B = ${matrixToLatex(prod)}`, mathBox, {
      displayMode: true,
      fallbackText: `A * B =\n${matrixToString(prod)}`,
    })
    body.appendChild(mathBox)

    const exHeading = document.createElement('h4')
    exHeading.className = 'la-subheading'
    exHeading.textContent = 'Ejemplo de cálculo (entrada c₁₁)'
    body.appendChild(exHeading)

    const exBox = document.createElement('div')
    exBox.className = 'la-result-math la-result-math--compact'
    void renderLatex(example.tex, exBox, {
      displayMode: true,
      fallbackText: example.plain,
    })
    body.appendChild(exBox)

    const latexCode = [
      LATEX_HEADER,
      '',
      '% Matrices de entrada',
      `\\[ A = ${matrixToLatex(a)} \\]`,
      `\\[ B = ${matrixToLatex(b)} \\]`,
      '',
      '% Resultado',
      `\\[ A \\cdot B = ${matrixToLatex(prod)} \\]`,
      '',
      '% Procedimiento (ejemplo de la entrada c_{11})',
      `\\[ ${example.tex} \\]`,
    ].join('\n')

    body.appendChild(buildLatexCopyBlock('multiply', 'Producto A · B', latexCode, track))
    track('linealalrl-solve', {
      op: 'multiply',
      size: `${a.length}x${a[0].length}*${b.length}x${b[0].length}`,
    })
  })

  // 6. Sistemas lineales Ax = b
  btnOpSys.addEventListener('click', () => {
    const a = readValidMatrixA({ container: resultSysEl, title: 'Sistema lineal Ax = b' })
    if (!a) return

    const readVec = readVectorB()
    if (!readVec.ok) {
      setResultError(resultSysEl, 'Sistema lineal Ax = b', readVec.error)
      readVec.firstInvalid.focus()
      return
    }

    const bVec = readVec.vector
    if (bVec.length !== a.length) {
      setResultError(
        resultSysEl,
        'Sistema lineal Ax = b',
        `Dimensiones incompatibles: A tiene ${a.length} filas pero el vector b tiene ${bVec.length} entradas.`,
      )
      return
    }

    const augmented: Matrix = a.map((row, i) => [...row, bVec[i]])
    const res = solveRref(a, bVec)
    const tipoEtiqueta =
      res.tipo === 'unica'
        ? 'Solución única'
        : res.tipo === 'infinitas'
          ? 'Infinitas soluciones'
          : 'Sin solución'

    const body = createResultFrame(
      resultSysEl,
      'Sistema lineal Ax = b',
      `${tipoEtiqueta} · ${res.pasos.length} paso${res.pasos.length === 1 ? '' : 's'}`,
    )

    const badge = document.createElement('div')
    badge.className = `la-sys-badge la-sys-badge--${res.tipo}`
    badge.textContent = tipoEtiqueta
    body.appendChild(badge)

    let latexSolution: string
    if (res.tipo === 'unica' && res.solucion) {
      const colVec: Matrix = res.solucion.map((v) => [v])
      const mathBox = document.createElement('div')
      mathBox.className = 'la-result-math'
      const varsTex = res.solucion.map((_, idx) => `x_${idx + 1}`).join(' \\\\ ')
      void renderLatex(
        `\\mathbf{x} = \\begin{bmatrix} ${varsTex} \\end{bmatrix} = ${matrixToLatex(colVec)}`,
        mathBox,
        {
          displayMode: true,
          fallbackText: res.solucion.map((v, idx) => `x${idx + 1} = ${ratToString(v)}`).join(', '),
        },
      )
      body.appendChild(mathBox)

      const hasFraction = res.solucion.some((v) => v.d !== 1n)
      if (hasFraction) {
        const approx = document.createElement('p')
        approx.className = 'la-result-meta-line'
        approx.textContent =
          'Aproximación decimal: ' +
          res.solucion.map((v, idx) => `x${idx + 1} ≈ ${ratToDecimal(v, 6)}`).join(', ')
        body.appendChild(approx)
      }

      latexSolution = `\\[ \\mathbf{x} = ${matrixToLatex(colVec)} \\]`
    } else if (res.tipo === 'infinitas' && res.libres) {
      const freeVars = formatFreeVariables(res.libres)
      const info = document.createElement('p')
      info.className = 'la-result-meta-line'
      info.textContent = `El sistema es compatible indeterminado. Variables libres: ${freeVars.plain}.`
      body.appendChild(info)

      const freeMath = document.createElement('div')
      freeMath.className = 'la-result-math la-result-math--compact'
      void renderLatex(`\\text{Variables libres: } ${freeVars.tex}`, freeMath, {
        displayMode: false,
        fallbackText: `Variables libres: ${freeVars.plain}`,
      })
      body.appendChild(freeMath)

      latexSolution = `% Infinitas soluciones; variables libres: ${freeVars.tex}`
    } else {
      const info = document.createElement('p')
      info.className = 'la-result-meta-line'
      info.textContent =
        'El sistema es incompatible (ninguna solución): en la RREF aparece una fila con todos los coeficientes en cero y término independiente distinto de cero.'
      body.appendChild(info)
      latexSolution = '% Sistema incompatible (sin solución).'
    }

    body.appendChild(
      renderStepsElement(
        res.pasos,
        'La matriz aumentada [A | b] ya estaba en forma escalonada reducida (0 pasos).',
      ),
    )

    const latexCode = [
      LATEX_HEADER,
      '',
      '% Matriz aumentada [A | b]',
      `\\[ [A \\mid \\mathbf{b}] = ${matrixToLatex(augmented)} \\]`,
      '',
      '% Resultado',
      latexSolution,
      '',
      '% Procedimiento (RREF de la aumentada)',
      formatStepsLatex(res.pasos),
    ].join('\n')

    body.appendChild(buildLatexCopyBlock('solve', 'Sistema lineal Ax = b', latexCode, track))
    track('linealalrl-solve', {
      op: 'solve',
      size: `${a.length}x${a[0].length}`,
      result: res.tipo,
    })
  })
}
