// Pruebas del módulo de fracciones exactas (src/rational.ts).
import {
  Rat,
  parseRat,
  ratToDecimal,
  ratToLatex,
  ratToString,
} from '../src/rational.ts'

let passed = 0
let failed = 0
function check(name: string, cond: boolean, extra = ''): void {
  if (cond) {
    passed++
    console.log(`ok ${passed + failed} - ${name}`)
  } else {
    failed++
    console.log(`FAIL ${passed + failed} - ${name} ${extra}`)
  }
}

function throws(fn: () => unknown): boolean {
  try {
    fn()
    return false
  } catch {
    return true
  }
}

// Normalización y construcción
const r1 = new Rat(6, 9)
check('reducción por MCD: 6/9 = 2/3', r1.n === 2n && r1.d === 3n)

const rNegDen = new Rat(3, -6)
check('denominador negativo se normaliza: 3/-6 = -1/2', rNegDen.n === -1n && rNegDen.d === 2n)

const rDoubleNeg = new Rat(-4, -8)
check('doble negativo se normaliza: -4/-8 = 1/2', rDoubleNeg.n === 1n && rDoubleNeg.d === 2n)

const rZero = new Rat(0, -7)
check('cero siempre es 0/1', rZero.n === 0n && rZero.d === 1n && rZero.isZero())

check('denominador cero lanza error', throws(() => new Rat(1, 0)))

// Operaciones aritméticas exactas
const half = new Rat(1, 2)
const third = new Rat(1, 3)
check('1/2 + 1/3 = 5/6', half.add(third).eq(new Rat(5, 6)))
check('1/2 - 1/3 = 1/6', half.sub(third).eq(new Rat(1, 6)))
check('1/3 - 1/2 = -1/6', third.sub(half).eq(new Rat(-1, 6)))
check('2/3 * 3/4 = 1/2', new Rat(2, 3).mul(new Rat(3, 4)).eq(half))
check('(3/4) / (9/8) = 2/3', new Rat(3, 4).div(new Rat(9, 8)).eq(new Rat(2, 3)))
check('(2/3) / (4/5) = 5/6', new Rat(2, 3).div(new Rat(4, 5)).eq(new Rat(5, 6)))
check('división por cero lanza error', throws(() => half.div(Rat.ZERO)))

// Inverso, opuesto, absoluto, comparaciones y predicados
check('inverso de 3/4 es 4/3', new Rat(3, 4).inv().eq(new Rat(4, 3)))
check('inverso de -3/4 es -4/3', new Rat(-3, 4).inv().eq(new Rat(-4, 3)))
check('inverso de 0 lanza error', throws(() => Rat.ZERO.inv()))
check('negativo de 3/5 es -3/5', new Rat(3, 5).neg().eq(new Rat(-3, 5)))
check('negativo de 0 es 0', Rat.ZERO.neg().isZero())
check('abs(-7/3) = 7/3', new Rat(-7, 3).abs().eq(new Rat(7, 3)))
check('isOne detecta 1/1 y rechaza 2/1', Rat.ONE.isOne() && !new Rat(2, 1).isOne())
check('cmp: -1/2 < 1/3', new Rat(-1, 2).cmp(third) === -1)
check('cmp: 2/4 == 1/2', new Rat(2, 4).cmp(half) === 0)
check('cmp: 2/3 > 1/2', new Rat(2, 3).cmp(half) === 1)
check('toNumber aproxima 1/4 = 0.25', new Rat(1, 4).toNumber() === 0.25)

// parseRat: casos válidos
const pHalf = parseRat('1/2')
check('parseRat("1/2") = 1/2', pHalf !== null && pHalf.eq(new Rat(1, 2)))

const pHalfSpaces = parseRat('  1 / 2  ')
check('parseRat("  1 / 2  ") = 1/2', pHalfSpaces !== null && pHalfSpaces.eq(new Rat(1, 2)))

const pNegFrac = parseRat('-3/4')
check('parseRat("-3/4") = -3/4', pNegFrac !== null && pNegFrac.eq(new Rat(-3, 4)))

const pNegDen = parseRat('3 / -4')
check('parseRat("3 / -4") = -3/4', pNegDen !== null && pNegDen.eq(new Rat(-3, 4)))

const pDec1 = parseRat('0.5')
check('parseRat("0.5") = 1/2', pDec1 !== null && pDec1.eq(new Rat(1, 2)))

const pDec2 = parseRat('1.25')
check('parseRat("1.25") = 5/4', pDec2 !== null && pDec2.eq(new Rat(5, 4)))

const pDec3 = parseRat('-1.25')
check('parseRat("-1.25") = -5/4', pDec3 !== null && pDec3.eq(new Rat(-5, 4)))

const pDecDotStart = parseRat('.5')
check('parseRat(".5") = 1/2', pDecDotStart !== null && pDecDotStart.eq(new Rat(1, 2)))

const pDecDotEnd = parseRat('2.')
check('parseRat("2.") = 2', pDecDotEnd !== null && pDecDotEnd.eq(new Rat(2, 1)))

const pInt = parseRat('2')
check('parseRat("2") = 2', pInt !== null && pInt.eq(new Rat(2, 1)))

const pNegInt = parseRat('-3')
check('parseRat("-3") = -3', pNegInt !== null && pNegInt.eq(new Rat(-3, 1)))

const pSciNeg = parseRat('1e-3')
check('parseRat("1e-3") = 1/1000', pSciNeg !== null && pSciNeg.eq(new Rat(1, 1000)))

const pSciPos = parseRat('2.5e2')
check('parseRat("2.5e2") = 250', pSciPos !== null && pSciPos.eq(new Rat(250, 1)))

// parseRat: casos inválidos -> null
const invalidos = ['', '   ', 'abc', '1/0', '1/2/3', '.', '1e', '1e+', '--2', 'NaN', 'Infinity']
for (const texto of invalidos) {
  check(`parseRat("${texto}") devuelve null`, parseRat(texto) === null)
}

// Formateo: texto, LaTeX y decimal aproximado
check('ratToString entero: "2"', ratToString(new Rat(2, 1)) === '2')
check('ratToString fracción: "3/2"', ratToString(new Rat(3, 2)) === '3/2')
check('ratToString negativo: "-3/2"', ratToString(new Rat(-3, 2)) === '-3/2')

check('ratToLatex entero: "2"', ratToLatex(new Rat(2, 1)) === '2')
check('ratToLatex fracción positiva: \\frac{3}{2}', ratToLatex(new Rat(3, 2)) === '\\frac{3}{2}')
check(
  'ratToLatex fracción negativa con signo fuera: -\\frac{3}{2}',
  ratToLatex(new Rat(-3, 2)) === '-\\frac{3}{2}',
)

check('ratToDecimal(1/2) = "0.5"', ratToDecimal(new Rat(1, 2)) === '0.5')
check('ratToDecimal(1/3) = "0.333333"', ratToDecimal(new Rat(1, 3)) === '0.333333')
check('ratToDecimal(2/3) = "0.666667"', ratToDecimal(new Rat(2, 3)) === '0.666667')
check('ratToDecimal(-5/4) = "-1.25"', ratToDecimal(new Rat(-5, 4)) === '-1.25')
check('ratToDecimal(-1/10000000, 6) = "0" sin signo negativo', ratToDecimal(new Rat(-1, 10000000), 6) === '0')

if (failed > 0) process.exit(1)
console.log(`\nrational: ${passed} comprobaciones pasaron.`)
