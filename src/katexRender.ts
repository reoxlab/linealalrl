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

// Carga perezosa de KaTeX con respaldo a texto plano si el renderizado falla.

type KatexModule = typeof import('katex')

let katexInstance: KatexModule['default'] | null = null
let katexPromise: Promise<KatexModule['default']> | null = null
const renderSeq = new WeakMap<HTMLElement, number>()

export function loadKatex(): Promise<KatexModule['default']> {
  if (katexInstance) return Promise.resolve(katexInstance)
  if (!katexPromise) {
    katexPromise = import('katex')
      .then((mod) => {
        katexInstance = mod.default
        return katexInstance
      })
      .catch((err: unknown) => {
        katexPromise = null
        throw err
      })
  }
  return katexPromise
}

export interface RenderLatexOptions {
  displayMode?: boolean
  fallbackText?: string
}

/**
 * Renderiza una expresión LaTeX en `el` usando KaTeX cargado de forma perezosa.
 * Si KaTeX falla (error de sintaxis o fallo al cargar), muestra `fallbackText`
 * (o el propio `tex`) en texto plano sin romper la página.
 */
export async function renderLatex(
  tex: string,
  el: HTMLElement,
  options: RenderLatexOptions = {},
): Promise<void> {
  const seq = (renderSeq.get(el) ?? 0) + 1
  renderSeq.set(el, seq)

  const trimmed = tex.trim()
  if (!trimmed) {
    el.textContent = ''
    return
  }

  const fallback = options.fallbackText ?? trimmed

  try {
    const katex = katexInstance ?? (await loadKatex())
    if (renderSeq.get(el) !== seq) return
    katex.render(trimmed, el, {
      throwOnError: true,
      displayMode: options.displayMode ?? false,
    })
  } catch {
    if (renderSeq.get(el) === seq) {
      el.textContent = fallback
    }
  }
}
