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

// Ejemplos predefinidos para el módulo 0 de matrices y sistemas.

import { type Matrix, fromStrings } from './matrix.ts'
import { type Rat, parseRat } from './rational.ts'

export interface MatrixPreset {
  id: string
  nombre: string
  descripcion: string
  a: string[][]
  vector?: string[]
  b?: string[][]
}

export const PRESETS: readonly MatrixPreset[] = [
  {
    id: 'identidad-3x3',
    nombre: 'Identidad 3×3',
    descripcion: 'Matriz identidad I₃: determinante 1, rango 3 e inversa igual a sí misma.',
    a: [
      ['1', '0', '0'],
      ['0', '1', '0'],
      ['0', '0', '1'],
    ],
  },
  {
    id: 'invertible-3x3',
    nombre: 'Invertible 3×3 (det ≠ 0)',
    descripcion: 'Matriz 3×3 con determinante −15 (no singular), ideal para probar cofactores e inversa.',
    a: [
      ['2', '-1', '0'],
      ['1', '3', '2'],
      ['4', '0', '-1'],
    ],
  },
  {
    id: 'singular-3x3',
    nombre: 'Singular 3×3 (det = 0)',
    descripcion: 'Fila 3 = Fila 1 + Fila 2 (det = 0, rango = 2): muestra el atasco al intentar invertir.',
    a: [
      ['1', '2', '3'],
      ['4', '5', '6'],
      ['5', '7', '9'],
    ],
  },
  {
    id: 'sistema-unico',
    nombre: 'Sistema con solución única',
    descripcion: 'Sistema lineal 3×3 con solución exacta única x = (−8, 1, −2).',
    a: [
      ['1', '2', '-1'],
      ['2', '3', '-1'],
      ['-2', '0', '-3'],
    ],
    vector: ['-4', '-11', '22'],
  },
]

export function presetToMatrices(preset: MatrixPreset): {
  a: Matrix
  vector: Rat[] | null
  b: Matrix | null
} {
  const a = fromStrings(preset.a)
  if (!a) {
    throw new Error(`Preset inválido «${preset.id}»: matriz A mal formada`)
  }

  let vector: Rat[] | null = null
  if (preset.vector) {
    const parsedVec: Rat[] = []
    for (const item of preset.vector) {
      const r = parseRat(item)
      if (!r) {
        throw new Error(`Preset inválido «${preset.id}»: vector b mal formado`)
      }
      parsedVec.push(r)
    }
    vector = parsedVec
  }

  let b: Matrix | null = null
  if (preset.b) {
    b = fromStrings(preset.b)
    if (!b) {
      throw new Error(`Preset inválido «${preset.id}»: matriz B mal formada`)
    }
  }

  return { a, vector, b }
}
