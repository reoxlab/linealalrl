// Pruebas del módulo de matrices exactas y operaciones paso a paso (src/matrix.ts).
import { Rat } from '../src/rational.ts'
import {
  type Matrix,
  clone,
  det,
  dims,
  fromStrings,
  identity,
  inverse,
  isSquare,
  matrixToLatex,
  matrixToString,
  multiply,
  rank,
  rref,
  solveRref,
  transpose,
  zeros,
} from '../src/matrix.ts'

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

function mat(rows: number[][]): Matrix {
  return rows.map((row) => row.map((v) => new Rat(v, 1)))
}

function matEq(a: Matrix, b: Matrix): boolean {
  if (a.length !== b.length) return false
  for (let i = 0; i < a.length; i++) {
    if (a[i].length !== b[i].length) return false
    for (let j = 0; j < a[i].length; j++) {
      if (!a[i][j].eq(b[i][j])) return false
    }
  }
  return true
}

// Constructores y utilidades básicas
const z23 = zeros(2, 3)
check('dims(zeros(2, 3)) = [2, 3]', dims(z23)[0] === 2 && dims(z23)[1] === 3)
check('isSquare(zeros(2, 3)) es false', !isSquare(z23))

const I3 = identity(3)
check('isSquare(identity(3)) es true', isSquare(I3))

const copiaI3 = clone(I3)
copiaI3[0][0] = new Rat(9, 1)
check('clone no muta la matriz original', I3[0][0].isOne())

const parsed = fromStrings([
  ['1/2', '-3'],
  ['0.25', '2'],
])
check(
  'fromStrings parsea fracciones y decimales exactos',
  parsed !== null &&
    parsed[0][0].eq(new Rat(1, 2)) &&
    parsed[0][1].eq(new Rat(-3, 1)) &&
    parsed[1][0].eq(new Rat(1, 4)) &&
    parsed[1][1].eq(new Rat(2, 1)),
)
check(
  'fromStrings devuelve null si una celda es inválida',
  fromStrings([
    ['1', 'abc'],
    ['2', '3'],
  ]) === null,
)
check('fromStrings([]) devuelve null', fromStrings([]) === null)
check(
  'fromStrings no rectangular devuelve null',
  fromStrings([['1', '2'], ['3']]) === null,
)

// Determinante
check('det(I_3) = 1', det(I3).valor.eq(Rat.ONE))
check('det(I_1) = 1', det(identity(1)).valor.eq(Rat.ONE))

// [[2, 1], [3, 4]] tiene det = 2*4 - 1*3 = 5; [[1, 2], [3, 4]] tiene det = 1*4 - 2*3 = -2
const m2a = mat([
  [2, 1],
  [3, 4],
])
check('det([[2, 1], [3, 4]]) = 5', det(m2a).valor.eq(new Rat(5, 1)))

const m2b = mat([
  [1, 2],
  [3, 4],
])
check('det([[1, 2], [3, 4]]) = -2', det(m2b).valor.eq(new Rat(-2, 1)))

// 3×3 conocida con cofactores: det = -15
const m3 = mat([
  [2, -1, 0],
  [1, 3, 2],
  [4, 0, -1],
])
const detM3 = det(m3)
check('det 3×3 conocida con cofactores = -15', detM3.valor.eq(new Rat(-15, 1)))
check(
  'det 3×3 registra pasos de cofactores con menores 2×2',
  detM3.pasos.length >= 2 && detM3.pasos.some((p) => p.matriz.length === 2),
)

// 3×3 con fila/columna abundante en ceros: det = 1
const m3Sparse = mat([
  [1, 0, 2],
  [3, 1, 4],
  [2, 0, 5],
])
const detSparse = det(m3Sparse)
check('det 3×3 dispersa = 1 eligiendo columna 2 con más ceros', detSparse.valor.eq(Rat.ONE))
check(
  'det 3×3 dispersa usa columna 2 (un solo cofactor no nulo)',
  detSparse.pasos.some((p) => p.texto.includes('columna 2')),
)

// Singular 3×3: fila 3 = fila 1 + fila 2 -> det = 0
const mSing3 = mat([
  [1, 2, 3],
  [4, 5, 6],
  [5, 7, 9],
])
check('det de matriz singular = 0', det(mSing3).valor.isZero())
check(
  'det de fila de ceros = 0',
  det(
    mat([
      [1, 2],
      [0, 0],
    ]),
  ).valor.isZero(),
)
check(
  'det de dos filas iguales = 0',
  det(
    mat([
      [1, 2],
      [1, 2],
    ]),
  ).valor.isZero(),
)

// 4×4 verificada a mano (expansión por fila 1 y eliminación gaussiana):
// +1·(-15) - 2·(29) + 0 - (-1)·(-54) = -15 - 58 - 54 = -127
const m4 = mat([
  [1, 2, 0, -1],
  [3, -1, 2, 0],
  [0, 4, 1, 2],
  [2, 0, -3, 1],
])
const detM4 = det(m4)
check(
  'det 4×4 conocida = -127 y fusiona sub-pasos recursivos',
  detM4.valor.eq(new Rat(-127, 1)) &&
    detM4.pasos.some((p) => p.matriz.length === 3) &&
    detM4.pasos.some((p) => p.matriz.length === 2),
)
check('det de matriz no cuadrada lanza error', throws(() => det(z23)))

// RREF (incluyendo intercambio de filas cuando el pivote es 0; det = 10 != 0)
const mSwap = mat([
  [0, 2, 4],
  [1, -1, 3],
  [2, 0, 5],
])
const rrefSwap = rref(mSwap)
check('RREF con intercambio de filas reduce a identidad 3×3', matEq(rrefSwap.resultado, identity(3)))
check(
  'RREF registra el intercambio de filas (↔)',
  rrefSwap.pasos.some((p) => p.texto.includes('↔')),
)
check('RREF no muta la matriz de entrada', mSwap[0][0].isZero())

const mRect = mat([
  [1, 2, -1, -4],
  [2, 3, -1, -11],
  [-2, 0, -3, 22],
])
const rrefRect = rref(mRect)
const esperadoRect = mat([
  [1, 0, 0, -8],
  [0, 1, 0, 1],
  [0, 0, 1, -2],
])
check('RREF de sistema 3×4 coincide exactamente', matEq(rrefRect.resultado, esperadoRect))

const rref1 = rref(mat([[5]]))
check('rref 1×1 de [[5]] = [[1]]', matEq(rref1.resultado, identity(1)))

// Inversa de [[2, 1], [3, 4]] y comprobación A · A⁻¹ = I
const inv2 = inverse(m2a)
const inv2Esperada: Matrix = [
  [new Rat(4, 5), new Rat(-1, 5)],
  [new Rat(-3, 5), new Rat(2, 5)],
]
check(
  'inversa exacta de [[2, 1], [3, 4]]',
  inv2.inversa !== null && matEq(inv2.inversa, inv2Esperada),
)
check(
  'comprobación exacta A · A⁻¹ = I y A⁻¹ · A = I',
  inv2.inversa !== null &&
    matEq(multiply(m2a, inv2.inversa), identity(2)) &&
    matEq(multiply(inv2.inversa, m2a), identity(2)),
)

const inv3 = inverse(mSwap)
check(
  'inversa 3×3 con intercambio verifica A · A⁻¹ = I',
  inv3.inversa !== null && matEq(multiply(mSwap, inv3.inversa), identity(3)),
)

const m1 = mat([[3]])
const inv1 = inverse(m1)
check(
  'inversa 1×1: inv([[3]]) = [[1/3]] y producto = [[1]]',
  inv1.inversa !== null &&
    matEq(inv1.inversa, [[new Rat(1, 3)]]) &&
    matEq(multiply(m1, inv1.inversa), identity(1)),
)

// Inversa de singular -> null con atasco
const mSing2 = mat([
  [1, 2],
  [2, 4],
])
const invSing = inverse(mSing2)
check(
  'inversa de singular devuelve inversa: null con atasco explicando fila y singularidad',
  invSing.inversa === null &&
    invSing.atasco.includes('fila 2') &&
    invSing.atasco.toLowerCase().includes('singular'),
)
check('inversa de matriz no cuadrada lanza error', throws(() => inverse(z23)))

// Transpuesta y producto matricial
const m23 = mat([
  [1, 2, 3],
  [4, 5, 6],
])
const t32 = transpose(m23)
check(
  'transpose(2×3) produce 3×2 correcta',
  matEq(
    t32,
    mat([
      [1, 4],
      [2, 5],
      [3, 6],
    ]),
  ),
)
check(
  'multiply(2×3, 3×2) produce 2×2 exacta',
  matEq(
    multiply(m23, t32),
    mat([
      [14, 32],
      [32, 77],
    ]),
  ),
)
check('multiply con dimensiones incompatibles lanza error', throws(() => multiply(m23, m23)))

// Rango
check('rank(I_3) = 3', rank(I3) === 3)
check('rank de matriz con fila dependiente es 2', rank(mSing3) === 2)
check('rank de matriz nula es 0', rank(zeros(3, 3)) === 0)

// solveRref: solución única
const solUnica = solveRref(m2a, [new Rat(1, 1), new Rat(2, 1)])
// 2x + y = 1, 3x + 4y = 2 => x = 2/5, y = 1/5
check(
  'solveRref solución única: x = [2/5, 1/5]',
  solUnica.tipo === 'unica' &&
    solUnica.solucion !== undefined &&
    solUnica.solucion.length === 2 &&
    solUnica.solucion[0].eq(new Rat(2, 5)) &&
    solUnica.solucion[1].eq(new Rat(1, 5)),
)

// solveRref: infinitas soluciones (variable libre en índice 2)
const aInf = mat([
  [1, 0, 2],
  [0, 1, -1],
  [1, 1, 1],
])
const solInf = solveRref(aInf, [new Rat(3, 1), new Rat(4, 1), new Rat(7, 1)])
check(
  'solveRref infinitas soluciones señala variable libre [2]',
  solInf.tipo === 'infinitas' &&
    solInf.libres !== undefined &&
    solInf.libres.length === 1 &&
    solInf.libres[0] === 2,
)

// solveRref: sin solución
const solNinguna = solveRref(mSing2, [new Rat(1, 1), new Rat(3, 1)])
check('solveRref inconsistente devuelve tipo "ninguna"', solNinguna.tipo === 'ninguna')
check(
  'solveRref con longitud de b incompatible lanza error',
  throws(() => solveRref(m2a, [new Rat(1, 1)])),
)

// LaTeX y texto plano
const latexStr = matrixToLatex(inv2Esperada)
check(
  'matrixToLatex contiene \\begin{bmatrix}, \\end{bmatrix} y fracciones LaTeX',
  latexStr.startsWith('\\begin{bmatrix}') &&
    latexStr.endsWith('\\end{bmatrix}') &&
    latexStr.includes('\\frac{4}{5}') &&
    latexStr.includes('-\\frac{1}{5}') &&
    latexStr.includes('\\\\'),
)

const plainStr = matrixToString(inv2Esperada)
check(
  'matrixToString usa fracciones exactas separadas por espacio y salto de línea',
  plainStr === '4/5 -1/5\n-3/5 2/5',
)

if (failed > 0) process.exit(1)
console.log(`\nmatrix: ${passed} comprobaciones pasaron.`)
