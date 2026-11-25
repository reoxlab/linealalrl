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

import 'katex/dist/katex.min.css'
import './style.css'
import { createThemeController } from 'rl-ui'
import { mountMatrixUI } from './ui.ts'
import { mountDirectUI } from './uiDirect.ts'
import { mountIterativeUI } from './uiIterative.ts'
import { mountEigenUI } from './uiEigen.ts'
import { mountConditionUI } from './uiCondition.ts'

interface LinealAlrlWindow extends Window {
  LINEALALRL_CONFIG?: { brandLink?: number }
  umami?: { track: (name: string, data?: Record<string, string>) => void }
}

export function track(eventName: string, data?: Record<string, string>): void {
  const w = window as LinealAlrlWindow
  try {
    w.umami?.track(eventName, data)
  } catch {
    // Umami no disponible en local o bloqueado: ignorar sin afectar la app.
  }
}

const app = document.querySelector<HTMLDivElement>('#app')
if (!app) throw new Error('Falta #app en index.html')

app.innerHTML = `
  <header class="la-header">
    <div class="la-header-inner">
      <div class="la-header-text">
        <a id="brand-link" class="la-brand" href="https://reoxlab.com" hidden>← Reox Lab</a>
        <h1>LinealAlRL</h1>
        <p class="la-tagline">Álgebra lineal en el navegador: matrices, sistemas, valores propios y condicionamiento</p>
      </div>
      <div class="la-header-controls">
        <label class="la-theme-control" for="theme-select">
          <span>Tema</span>
          <select class="la-select la-select--compact" id="theme-select" aria-label="Tema de la interfaz">
            <option value="system" selected>Sistema</option>
            <option value="light">Claro</option>
            <option value="dark">Oscuro</option>
          </select>
        </label>
        <nav class="la-tabs" role="tablist" aria-label="Módulos de LinealAlRL">
          <button class="la-tab is-active" id="tab-matrix" role="tab" type="button" aria-selected="true" aria-controls="panel-matrix">Matrices y operaciones</button>
          <button class="la-tab" id="tab-direct" role="tab" type="button" aria-selected="false" aria-controls="panel-direct">Sistemas directos</button>
          <button class="la-tab" id="tab-iterative" role="tab" type="button" aria-selected="false" aria-controls="panel-iterative">Sistemas iterativos</button>
          <button class="la-tab" id="tab-eigen" role="tab" type="button" aria-selected="false" aria-controls="panel-eigen">Valores propios</button>
          <button class="la-tab" id="tab-condition" role="tab" type="button" aria-selected="false" aria-controls="panel-condition">Condicionamiento</button>
        </nav>
      </div>
    </div>
  </header>

  <main class="la-main">
    <div id="panel-matrix" role="tabpanel" aria-labelledby="tab-matrix"></div>
    <div id="panel-direct" role="tabpanel" aria-labelledby="tab-direct" hidden></div>
    <div id="panel-iterative" role="tabpanel" aria-labelledby="tab-iterative" hidden></div>
    <div id="panel-eigen" role="tabpanel" aria-labelledby="tab-eigen" hidden></div>
    <div id="panel-condition" role="tabpanel" aria-labelledby="tab-condition" hidden></div>
  </main>

  <footer class="la-footer">
    LinealAlRL es software libre (GPL v3): puedes usarlo, estudiarlo, modificarlo y compartirlo.
    Todo se calcula localmente en tu navegador, sin enviar datos a ningún servidor.
  </footer>
`

const panelMatrix = app.querySelector<HTMLDivElement>('#panel-matrix')
const panelDirect = app.querySelector<HTMLDivElement>('#panel-direct')
const panelIterative = app.querySelector<HTMLDivElement>('#panel-iterative')
const panelEigen = app.querySelector<HTMLDivElement>('#panel-eigen')
const panelCondition = app.querySelector<HTMLDivElement>('#panel-condition')
const tabMatrix = app.querySelector<HTMLButtonElement>('#tab-matrix')
const tabDirect = app.querySelector<HTMLButtonElement>('#tab-direct')
const tabIterative = app.querySelector<HTMLButtonElement>('#tab-iterative')
const tabEigen = app.querySelector<HTMLButtonElement>('#tab-eigen')
const tabCondition = app.querySelector<HTMLButtonElement>('#tab-condition')
if (
  !panelMatrix ||
  !panelDirect ||
  !panelIterative ||
  !panelEigen ||
  !panelCondition ||
  !tabMatrix ||
  !tabDirect ||
  !tabIterative ||
  !tabEigen ||
  !tabCondition
) {
  throw new Error('Faltan pestañas o paneles en la interfaz')
}

mountMatrixUI(panelMatrix, track)
mountDirectUI(panelDirect, track)
mountIterativeUI(panelIterative, track)
mountEigenUI(panelEigen, track)
mountConditionUI(panelCondition, track)

type TabId = 'matrix' | 'direct' | 'iterative' | 'eigen' | 'condition'

function selectTab(active: TabId): void {
  const isMatrix = active === 'matrix'
  const isDirect = active === 'direct'
  const isIterative = active === 'iterative'
  const isEigen = active === 'eigen'
  const isCondition = active === 'condition'

  tabMatrix!.classList.toggle('is-active', isMatrix)
  tabDirect!.classList.toggle('is-active', isDirect)
  tabIterative!.classList.toggle('is-active', isIterative)
  tabEigen!.classList.toggle('is-active', isEigen)
  tabCondition!.classList.toggle('is-active', isCondition)

  tabMatrix!.setAttribute('aria-selected', isMatrix ? 'true' : 'false')
  tabDirect!.setAttribute('aria-selected', isDirect ? 'true' : 'false')
  tabIterative!.setAttribute('aria-selected', isIterative ? 'true' : 'false')
  tabEigen!.setAttribute('aria-selected', isEigen ? 'true' : 'false')
  tabCondition!.setAttribute('aria-selected', isCondition ? 'true' : 'false')

  panelMatrix!.hidden = !isMatrix
  panelDirect!.hidden = !isDirect
  panelIterative!.hidden = !isIterative
  panelEigen!.hidden = !isEigen
  panelCondition!.hidden = !isCondition
}

tabMatrix.addEventListener('click', () => selectTab('matrix'))
tabDirect.addEventListener('click', () => selectTab('direct'))
tabIterative.addEventListener('click', () => selectTab('iterative'))
tabEigen.addEventListener('click', () => selectTab('eigen'))
tabCondition.addEventListener('click', () => selectTab('condition'))

const themeSelect = app.querySelector<HTMLSelectElement>('#theme-select')

createThemeController({
  storageKey: 'linealalrl-theme-v1',
  select: themeSelect,
})

const brandLink = document.querySelector<HTMLAnchorElement>('#brand-link')
if (brandLink && (window as LinealAlrlWindow).LINEALALRL_CONFIG?.brandLink === 1) {
  brandLink.hidden = false
}
