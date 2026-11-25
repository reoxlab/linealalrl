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

/** Máximo común divisor no negativo entre dos enteros BigInt. */
function gcd(a: bigint, b: bigint): bigint {
  let x = a < 0n ? -a : a
  let y = b < 0n ? -b : b
  while (y !== 0n) {
    const t = x % y
    x = y
    y = t
  }
  return x
}

function toBigInt(v: bigint | number, etiqueta: string): bigint {
  if (typeof v === 'bigint') return v
  if (!Number.isInteger(v)) {
    throw new Error(`${etiqueta} debe ser un entero: recibido ${String(v)}`)
  }
  return BigInt(v)
}

/**
 * Número racional exacto representado como fracción irreducible n / d,
 * con denominador estrictamente positivo (d > 0) y cero normalizado a 0 / 1.
 */
export class Rat {
  readonly n: bigint
  readonly d: bigint

  static readonly ZERO = new Rat(0n, 1n)
  static readonly ONE = new Rat(1n, 1n)
  static readonly NEG_ONE = new Rat(-1n, 1n)

  constructor(n: bigint | number, d: bigint | number = 1n) {
    let num = toBigInt(n, 'El numerador')
    let den = toBigInt(d, 'El denominador')

    if (den === 0n) {
      throw new Error('El denominador no puede ser cero')
    }
    if (num === 0n) {
      this.n = 0n
      this.d = 1n
      return
    }
    if (den < 0n) {
      num = -num
      den = -den
    }
    const g = gcd(num, den)
    this.n = num / g
    this.d = den / g
  }

  static fromInt(k: bigint | number): Rat {
    return new Rat(k, 1n)
  }

  add(other: Rat): Rat {
    if (this.n === 0n) return other
    if (other.n === 0n) return this
    return new Rat(this.n * other.d + other.n * this.d, this.d * other.d)
  }

  sub(other: Rat): Rat {
    if (other.n === 0n) return this
    if (this.n === 0n) return other.neg()
    return new Rat(this.n * other.d - other.n * this.d, this.d * other.d)
  }

  mul(other: Rat): Rat {
    if (this.n === 0n || other.n === 0n) return Rat.ZERO
    const g1 = gcd(this.n, other.d)
    const g2 = gcd(other.n, this.d)
    return new Rat((this.n / g1) * (other.n / g2), (this.d / g2) * (other.d / g1))
  }

  div(other: Rat): Rat {
    if (other.n === 0n) {
      throw new Error('División por cero')
    }
    if (this.n === 0n) return Rat.ZERO
    return this.mul(other.inv())
  }

  neg(): Rat {
    if (this.n === 0n) return Rat.ZERO
    return new Rat(-this.n, this.d)
  }

  inv(): Rat {
    if (this.n === 0n) {
      throw new Error('El cero no tiene inverso multiplicativo')
    }
    return new Rat(this.d, this.n)
  }

  eq(other: Rat): boolean {
    return this.n === other.n && this.d === other.d
  }

  isZero(): boolean {
    return this.n === 0n
  }

  isOne(): boolean {
    return this.n === 1n && this.d === 1n
  }

  cmp(other: Rat): -1 | 0 | 1 {
    const lhs = this.n * other.d
    const rhs = other.n * this.d
    if (lhs < rhs) return -1
    if (lhs > rhs) return 1
    return 0
  }

  abs(): Rat {
    return this.n < 0n ? new Rat(-this.n, this.d) : this
  }

  /**
   * Conversión aproximada a `number` SOLO para visualización.
   * Nunca debe usarse para realizar cálculos internos.
   */
  toNumber(): number {
    if (this.n === 0n) return 0
    const nf = Number(this.n)
    const df = Number(this.d)
    if (Number.isFinite(nf) && Number.isFinite(df) && df !== 0) {
      return nf / df
    }
    // Respaldo para fracciones con numerador o denominador fuera del rango de float64.
    const neg = this.n < 0n
    const absN = neg ? -this.n : this.n
    const nLen = absN.toString().length
    const dLen = this.d.toString().length
    const exp10 = nLen - dLen
    const shift = 18 - exp10
    let scaled: bigint
    if (shift >= 0) {
      scaled = (absN * 10n ** BigInt(shift)) / this.d
    } else {
      scaled = absN / (this.d * 10n ** BigInt(-shift))
    }
    const approx = Number(scaled) * 10 ** -shift
    return neg ? -approx : approx
  }
}

const FRAC_RE = /^([+-]?\d+)\s*\/\s*([+-]?\d+)$/
const DEC_SCI_RE = /^([+-])?(?:(\d+)(?:\.(\d*))?|\.(\d+))(?:[eE]([+-]?\d+))?$/
const MAX_ABS_EXP = 10000

/**
 * Parsea texto numérico a fracción exacta `Rat`.
 * Acepta enteros (`-3`), fracciones (`1/2`, `-3/4`, con espacios alrededor de `/`),
 * decimales finitos (`0.5`, `-1.25`, `.5`, `2.`) y notación científica finita (`1e-3`, `2.5e2`).
 * Devuelve `null` si el texto no representa un número válido.
 */
export function parseRat(texto: string): Rat | null {
  const limpio = texto.trim().replace(/−/g, '-')
  if (limpio.length === 0) return null

  const fracMatch = FRAC_RE.exec(limpio)
  if (fracMatch) {
    const num = BigInt(fracMatch[1])
    const den = BigInt(fracMatch[2])
    if (den === 0n) return null
    return new Rat(num, den)
  }

  const decMatch = DEC_SCI_RE.exec(limpio)
  if (!decMatch) return null

  const sign = decMatch[1] === '-' ? -1n : 1n
  const intPart = decMatch[2] ?? ''
  const fracPart = decMatch[3] ?? decMatch[4] ?? ''
  const expStr = decMatch[5]

  let exp = 0
  if (expStr !== undefined) {
    exp = Number(expStr)
    if (!Number.isSafeInteger(exp) || Math.abs(exp) > MAX_ABS_EXP) {
      return null
    }
  }

  const digits = intPart + fracPart
  const totalExp = exp - fracPart.length
  if (Math.abs(totalExp) > MAX_ABS_EXP) return null

  const mag = BigInt(digits)
  if (mag === 0n) return Rat.ZERO

  const signedMag = sign * mag
  if (totalExp >= 0) {
    return new Rat(signedMag * 10n ** BigInt(totalExp), 1n)
  }
  return new Rat(signedMag, 10n ** BigInt(-totalExp))
}

/** Representación exacta en texto plano: `"2"` si es entero, `"3/2"` en caso contrario. */
export function ratToString(r: Rat): string {
  if (r.d === 1n) return r.n.toString()
  return `${r.n.toString()}/${r.d.toString()}`
}

/** Representación exacta en LaTeX: enteros tal cual, fracciones como `\frac{a}{b}` con signo fuera. */
export function ratToLatex(r: Rat): string {
  if (r.d === 1n) return r.n.toString()
  if (r.n < 0n) {
    return `-\\frac{${(-r.n).toString()}}{${r.d.toString()}}`
  }
  return `\\frac{${r.n.toString()}}{${r.d.toString()}}`
}

/**
 * Representación decimal aproximada con hasta `cifras` decimales (sin ceros finales),
 * calculada mediante redondeo exacto en BigInt para evitar artefactos de punto flotante.
 */
export function ratToDecimal(r: Rat, cifras = 6): string {
  const digits = Math.max(0, Math.floor(cifras))
  if (r.n === 0n) return '0'
  if (r.d === 1n) return r.n.toString()

  const neg = r.n < 0n
  const absN = neg ? -r.n : r.n
  const scale = 10n ** BigInt(digits)
  const scaled = (absN * scale * 2n + r.d) / (2n * r.d)
  if (scaled === 0n) return '0'

  const sign = neg ? '-' : ''
  if (digits === 0) return `${sign}${scaled.toString()}`

  const intPart = scaled / scale
  const fracPart = (scaled % scale)
    .toString()
    .padStart(digits, '0')
    .replace(/0+$/, '')

  return fracPart.length > 0
    ? `${sign}${intPart.toString()}.${fracPart}`
    : `${sign}${intPart.toString()}`
}
