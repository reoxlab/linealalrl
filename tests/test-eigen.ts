// Pruebas del módulo 3: valores y vectores propios (src/eigen.ts).
// Polinomio característico (Faddeev–LeVerrier), raíces racionales, método de la potencia
// (directa, desplazada e inversa vía Gauss), iteración QR, Jacobi simétrico, multiplicidades y defectivas.

import {
  charPoly,
  charPolyLatex,
  eigenResidual,
  esDefectiva,
  evalPoly,
  exactEigenvectors,
  inversePowerMethod,
  jacobiEigen,
  multiplicidades,
  powerMethod,
  qrIteration,
  rationalRoots,
} from '../src/eigen.ts'
import { type Matrix, fromStrings } from '../src/matrix.ts'
import { Rat, ratToString } from '../src/rational.ts'
import {
  NON_RATIONAL_MULT_NOTE,
  describeJacobiBadge,
  formatEigenErrorMessage,
  formatIncompleteRationalNotice,
  formatNumLatex,
  formatNumericMultiplicityCells,
  formatRootWithMultiplicity,
} from '../src/uiEigen.ts'

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

function mat(rows: (number | string)[][]): Matrix {
  const parsed = fromStrings(rows.map((r) => r.map((c) => String(c))))
  if (!parsed) throw new Error('Matriz de prueba inválida')
  return parsed
}

function ratListEq(actual: Rat[], expected: (number | Rat)[]): boolean {
  if (actual.length !== expected.length) return false
  return actual.every((r, i) => {
    const exp = expected[i] instanceof Rat ? expected[i] : Rat.fromInt(expected[i])
    return r.eq(exp)
  })
}

function throwsErrorMatching(fn: () => unknown, pred: (msg: string) => boolean): boolean {
  try {
    fn()
    return false
  } catch (err) {
    return err instanceof Error && pred(err.message)
  }
}

function matMulNum(a: number[][], b: number[][]): number[][] {
  const n = a.length
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => {
      let s = 0
      for (let k = 0; k < n; k++) s += a[i][k] * b[k][j]
      return s
    }),
  )
}

function transposeNum(a: number[][]): number[][] {
  const n = a.length
  return Array.from({ length: n }, (_, j) => Array.from({ length: n }, (_, i) => a[i][j]))
}

function frobeniusDiff(a: number[][], b: number[][]): number {
  let sumSq = 0
  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < a[i].length; j++) {
      const d = a[i][j] - b[i][j]
      sumSq += d * d
    }
  }
  return Math.sqrt(sumSq)
}

function diagNum(vals: number[]): number[][] {
  return Array.from({ length: vals.length }, (_, i) =>
    Array.from({ length: vals.length }, (_, j) => (i === j ? vals[i] : 0)),
  )
}

function identityNum(n: number): number[][] {
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)),
  )
}

function vecCloseUpToSign(actual: number[], expected: number[], tol = 1e-8): boolean {
  if (actual.length !== expected.length) return false
  const sameSign = actual.every((v, i) => Math.abs(v - expected[i]) < tol)
  const oppSign = actual.every((v, i) => Math.abs(v + expected[i]) < tol)
  return sameSign || oppSign
}

// ---------------------------------------------------------------------------
// 1. Polinomio característico y raíces racionales: [[2,1],[1,2]]
// ---------------------------------------------------------------------------
const a2x2 = mat([
  [2, 1],
  [1, 2],
])
const cp2 = charPoly(a2x2)
check(
  'charPoly([[2,1],[1,2]]) = [1, -4, 3] (λ² − 4λ + 3)',
  ratListEq(cp2, [1, -4, 3]),
  `got=[${cp2.map(ratToString).join(', ')}]`,
)

const roots2 = rationalRoots(cp2)
check(
  'rationalRoots([1, -4, 3]) = [1, 3]',
  ratListEq(roots2, [1, 3]),
  `got=[${roots2.map(ratToString).join(', ')}]`,
)

const tex2 = charPolyLatex(cp2)
check(
  'charPolyLatex([[2,1],[1,2]]) genera LaTeX limpio p(\\lambda) = \\lambda^2 - 4\\lambda + 3',
  tex2 === 'p(\\lambda) = \\lambda^2 - 4\\lambda + 3',
  `got=${tex2}`,
)

// ---------------------------------------------------------------------------
// 2. charPoly de 3×3 conocidas verificadas a mano
// ---------------------------------------------------------------------------
// Matriz A = [[1, 2, 0], [0, 2, 1], [1, 0, 3]]:
//   det(λI − A) = (λ−1)(λ−2)(λ−3) + (−2)(−1)(−1)
//               = (λ³ − 6λ² + 11λ − 6) − 2 = λ³ − 6λ² + 11λ − 8.
const a3Manual = mat([
  [1, 2, 0],
  [0, 2, 1],
  [1, 0, 3],
])
const cp3Manual = charPoly(a3Manual)
check(
  'charPoly([[1,2,0],[0,2,1],[1,0,3]]) = [1, -6, 11, -8] (verificado a mano)',
  ratListEq(cp3Manual, [1, -6, 11, -8]),
  `got=[${cp3Manual.map(ratToString).join(', ')}]`,
)

// Matriz simétrica 3×3 preset: A = [[5,0,2],[0,3,2],[2,2,4]]
//   Valores propios exactos: 7, 4, 1
//   p(λ) = (λ−7)(λ−4)(λ−1) = λ³ − 12λ² + 39λ − 28.
const aSym3 = mat([
  [5, 0, 2],
  [0, 3, 2],
  [2, 2, 4],
])
const cpSym3 = charPoly(aSym3)
check(
  'charPoly([[5,0,2],[0,3,2],[2,2,4]]) = [1, -12, 39, -28] (verificado a mano)',
  ratListEq(cpSym3, [1, -12, 39, -28]),
  `got=[${cpSym3.map(ratToString).join(', ')}]`,
)

const rootsSym3 = rationalRoots(cpSym3)
check(
  'rationalRoots de la simétrica 3×3 devuelve [1, 4, 7] verificando p(r) = 0 exacto',
  ratListEq(rootsSym3, [1, 4, 7]) && rootsSym3.every((r) => evalPoly(cpSym3, r).isZero()),
  `got=[${rootsSym3.map(ratToString).join(', ')}]`,
)

// Raíces racionales fraccionarias: diag(1/2, -3/4) -> λ² + (1/4)λ − 3/8
const aFrac = mat([
  ['1/2', '0'],
  ['0', '-3/4'],
])
const cpFrac = charPoly(aFrac)
const rootsFrac = rationalRoots(cpFrac)
check(
  'charPoly y rationalRoots con fracciones exactas: diag(1/2, -3/4) -> [-3/4, 1/2]',
  ratListEq(cpFrac, [Rat.ONE, new Rat(1, 4), new Rat(-3, 8)]) &&
    ratListEq(rootsFrac, [new Rat(-3, 4), new Rat(1, 2)]),
  `roots=[${rootsFrac.map(ratToString).join(', ')}]`,
)

// Raíz cero y polinomio sin raíces racionales
const aZeroEig = mat([
  [0, 1],
  [0, 2],
])
check(
  'rationalRoots extrae λ = 0 correctamente en [[0,1],[0,2]] -> [0, 2]',
  ratListEq(rationalRoots(charPoly(aZeroEig)), [0, 2]),
)

const aIrr = mat([
  [1, 1],
  [1, 0],
])
check(
  'rationalRoots devuelve [] cuando no hay raíces racionales ([[1,1],[1,0]], λ² − λ − 1)',
  ratListEq(charPoly(aIrr), [1, -1, -1]) && rationalRoots(charPoly(aIrr)).length === 0,
)

// ---------------------------------------------------------------------------
// 3. Método de la potencia (directo, desplazado e inverso vía Gauss)
// ---------------------------------------------------------------------------
const a2Num = [
  [2, 1],
  [1, 2],
]
const invSqrt2 = 1 / Math.sqrt(2)
const pRes = powerMethod(a2Num, [1, 0], 1e-10, 200)
check(
  'Método de la potencia en [[2,1],[1,2]] converge a λ = 3 (error < 1e-8) y vector ≈ [1,1]/√2',
  pRes.convergio &&
    Math.abs(pRes.lambda - 3) < 1e-8 &&
    vecCloseUpToSign(pRes.vector, [invSqrt2, invSqrt2], 1e-8) &&
    pRes.iteraciones.length > 0 &&
    eigenResidual(a2Num, pRes.vector, pRes.lambda) < 1e-8,
  `lambda=${pRes.lambda}, v=[${pRes.vector.join(', ')}]`,
)

// Potencia con desplazamiento shift = 3 en [[2,1],[1,2]] -> converge a λ = 1
const pShift = powerMethod(a2Num, [1, 0], 1e-10, 200, 3)
check(
  'Potencia con desplazamiento (shift = 3) en [[2,1],[1,2]] converge a λ = 1 y vector ≈ [1,-1]/√2',
  pShift.convergio &&
    Math.abs(pShift.lambda - 1) < 1e-8 &&
    vecCloseUpToSign(pShift.vector, [invSqrt2, -invSqrt2], 1e-8),
  `lambda=${pShift.lambda}`,
)

// Potencia inversa con desplazamiento (reutilizando gaussSolve de src/direct.ts)
const aSym3Num = [
  [5, 0, 2],
  [0, 3, 2],
  [2, 2, 4],
]
const pInvMid = inversePowerMethod(aSym3Num, [1, 2, 3], 1e-10, 100, 3.8)
check(
  'Potencia inversa (shift = 3.8) en simétrica 3×3 converge al valor propio intermedio λ = 4 (error < 1e-8)',
  pInvMid.convergio &&
    Math.abs(pInvMid.lambda - 4) < 1e-8 &&
    eigenResidual(aSym3Num, pInvMid.vector, pInvMid.lambda) < 1e-8,
  `lambda=${pInvMid.lambda}, iters=${pInvMid.iteraciones.length}`,
)

const pInvMin = inversePowerMethod(aSym3Num, [1, 2, 3], 1e-10, 100, 0.2)
check(
  'Potencia inversa (shift = 0.2) en simétrica 3×3 converge al menor valor propio λ = 1 (error < 1e-8)',
  pInvMin.convergio &&
    Math.abs(pInvMin.lambda - 1) < 1e-8 &&
    eigenResidual(aSym3Num, pInvMin.vector, pInvMin.lambda) < 1e-8,
  `lambda=${pInvMin.lambda}`,
)

// Potencia inversa cuando shift coincide exactamente con un valor propio -> singular manejado sin excepción
const pInvSing = inversePowerMethod(a2Num, [1, 0], 1e-8, 50, 3)
check(
  'Potencia inversa reporta convergio: false con motivo claro cuando (A − shift·I) es singular',
  !pInvSing.convergio && typeof pInvSing.motivo === 'string' && pInvSing.motivo.includes('singular'),
  `motivo=${String(pInvSing.motivo)}`,
)

// Potencia sin convergencia en maxIter (valores propios ±1 de igual módulo)
const pNoConv = powerMethod(
  [
    [0, 1],
    [1, 0],
  ],
  [1, 0],
  1e-8,
  25,
)
check(
  'Método de la potencia devuelve convergio: false en [[0,1],[1,0]] con x₀ = [1,0]',
  !pNoConv.convergio && pNoConv.iteraciones.length === 25 && Boolean(pNoConv.motivo),
)

// ---------------------------------------------------------------------------
// 4. Iteración QR (reutilizando Householder de src/direct.ts)
// ---------------------------------------------------------------------------
const aUpper3 = [
  [4, 2, -1],
  [0, -3, 5],
  [0, 0, 7],
]
const qrUpper = qrIteration(aUpper3, 1e-10, 100)
check(
  'qrIteration en 3×3 triangular superior devuelve exactamente los valores de la diagonal [4, -3, 7]',
  qrUpper.convergio &&
    qrUpper.iteraciones === 0 &&
    qrUpper.valores.length === 3 &&
    qrUpper.valores[0] === 4 &&
    qrUpper.valores[1] === -3 &&
    qrUpper.valores[2] === 7,
  `valores=[${qrUpper.valores.join(', ')}]`,
)

const qrSym = qrIteration(aSym3Num, 1e-10, 300)
const qrSymSorted = qrSym.valores.slice().sort((x, y) => y - x)
check(
  'qrIteration en simétrica 3×3 [[5,0,2],[0,3,2],[2,2,4]] converge a [7, 4, 1] con error < 1e-8',
  qrSym.convergio &&
    qrSym.iteraciones > 0 &&
    Math.abs(qrSymSorted[0] - 7) < 1e-8 &&
    Math.abs(qrSymSorted[1] - 4) < 1e-8 &&
    Math.abs(qrSymSorted[2] - 1) < 1e-8,
  `valores=[${qrSymSorted.join(', ')}], iters=${qrSym.iteraciones}`,
)

// QR sin convergencia en matriz de rotación (valores propios complejos ±i)
const qrRot = qrIteration(
  [
    [0, -1],
    [1, 0],
  ],
  1e-8,
  40,
)
check(
  'qrIteration devuelve convergio: false con motivo claro cuando no converge en maxIter (rotación 90°)',
  !qrRot.convergio && qrRot.iteraciones === 40 && qrRot.valores.length === 2 && Boolean(qrRot.motivo),
)

// ---------------------------------------------------------------------------
// 5. Método de Jacobi para matrices simétricas (reconstrucción Q·D·Qᵀ y QᵀQ ≈ I)
// ---------------------------------------------------------------------------
const jacSym3 = jacobiEigen(aSym3Num)
const q3 = jacSym3.vectores
const d3 = diagNum(jacSym3.valores)
const recon3 = matMulNum(matMulNum(q3, d3), transposeNum(q3))
const qtq3 = matMulNum(transposeNum(q3), q3)
const residRecon3 = frobeniusDiff(aSym3Num, recon3)
const residOrtho3 = frobeniusDiff(qtq3, identityNum(3))

check(
  'jacobiEigen en simétrica 3×3 halla valores [7, 4, 1] con error < 1e-10',
  Math.abs(jacSym3.valores[0] - 7) < 1e-10 &&
    Math.abs(jacSym3.valores[1] - 4) < 1e-10 &&
    Math.abs(jacSym3.valores[2] - 1) < 1e-10 &&
    jacSym3.iteraciones > 0,
  `valores=[${jacSym3.valores.join(', ')}]`,
)
check(
  'jacobiEigen en simétrica 3×3: reconstrucción Q·D·Qᵀ con residuo < 1e-10 y QᵀQ ≈ I',
  residRecon3 < 1e-10 &&
    residOrtho3 < 1e-10 &&
    jacSym3.residuo < 1e-10 &&
    jacSym3.residuoOrtogonalidad < 1e-10,
  `recon=${residRecon3.toExponential(2)}, ortho=${residOrtho3.toExponential(2)}`,
)

// Simétrica 3×3 con raíces irracionales: [[4,1,0],[1,4,1],[0,1,4]] -> 4+√2, 4, 4−√2
const aSymIrr3 = [
  [4, 1, 0],
  [1, 4, 1],
  [0, 1, 4],
]
const jacIrr = jacobiEigen(aSymIrr3)
const expIrr = [4 + Math.SQRT2, 4, 4 - Math.SQRT2]
check(
  'jacobiEigen en simétrica 3×3 irracional halla 4±√2 y 4 con Q·D·Qᵀ y QᵀQ < 1e-10',
  jacIrr.valores.every((v, i) => Math.abs(v - expIrr[i]) < 1e-10) &&
    jacIrr.residuo < 1e-10 &&
    jacIrr.residuoOrtogonalidad < 1e-10,
  `valores=[${jacIrr.valores.join(', ')}]`,
)

check(
  'jacobiEigen lanza Error en español para matriz no simétrica',
  throwsErrorMatching(
    () =>
      jacobiEigen([
        [1, 2],
        [0, 1],
      ]),
    (msg) => msg.toLowerCase().includes('simétrica'),
  ),
)

// ---------------------------------------------------------------------------
// 6. Multiplicidades y esDefectiva
// ---------------------------------------------------------------------------
const aDef = mat([
  [1, 1],
  [0, 1],
])
const mDef = multiplicidades(aDef, Rat.ONE)
check(
  'esDefectiva([[1,1],[0,1]]) = true con algebraica = 2 y geométrica = 1',
  esDefectiva(aDef) === true && mDef.algebraica === 2 && mDef.geometrica === 1,
  `alg=${mDef.algebraica}, geom=${mDef.geometrica}`,
)

const m2_3 = multiplicidades(a2x2, Rat.fromInt(3))
const m2_1 = multiplicidades(a2x2, Rat.ONE)
check(
  'esDefectiva([[2,1],[1,2]]) = false con multiplicidades (1, 1) para λ = 1 y λ = 3',
  esDefectiva(a2x2) === false &&
    m2_3.algebraica === 1 &&
    m2_3.geometrica === 1 &&
    m2_1.algebraica === 1 &&
    m2_1.geometrica === 1,
)

// Matriz escalar 2I: raíz doble pero NO defectiva (alg = 2, geom = 2)
const aScalar = mat([
  [2, 0],
  [0, 2],
])
const mScalar = multiplicidades(aScalar, Rat.fromInt(2))
check(
  'esDefectiva([[2,0],[0,2]]) = false (raíz doble con alg = 2, geom = 2)',
  esDefectiva(aScalar) === false && mScalar.algebraica === 2 && mScalar.geometrica === 2,
)

// Simétrica 3×3 con valor propio doble [[2,1,1],[1,2,1],[1,1,2]]: λ=4 (1,1), λ=1 (2,2) -> no defectiva
const aSymDouble = mat([
  [2, 1, 1],
  [1, 2, 1],
  [1, 1, 2],
])
const mDouble1 = multiplicidades(aSymDouble, Rat.ONE)
const basis1 = exactEigenvectors(aSymDouble, Rat.ONE)
check(
  'Simétrica con raíz doble [[2,1,1],[1,2,1],[1,1,2]]: no defectiva, alg=2, geom=2 y base exacta de 2 vectores',
  esDefectiva(aSymDouble) === false &&
    mDouble1.algebraica === 2 &&
    mDouble1.geometrica === 2 &&
    basis1.length === 2,
)

// Bloque de Jordan 3×3 [[2,1,0],[0,2,1],[0,0,2]]: defectiva (alg=3, geom=1)
const aJordan3 = mat([
  [2, 1, 0],
  [0, 2, 1],
  [0, 0, 2],
])
const mJordan3 = multiplicidades(aJordan3, Rat.fromInt(2))
check(
  'Bloque de Jordan 3×3 es defectivo con algebraica = 3 y geométrica = 1',
  esDefectiva(aJordan3) === true && mJordan3.algebraica === 3 && mJordan3.geometrica === 1,
)

// Multiplicidades con raíz irracional (number): algebraica null + aviso + geométrica numérica = 1
const phi = (1 + Math.sqrt(5)) / 2
const mIrr = multiplicidades(aIrr, phi)
check(
  'multiplicidades con raíz irracional φ devuelve algebraica: null con aviso y geométrica = 1',
  mIrr.algebraica === null && mIrr.geometrica === 1 && typeof mIrr.aviso === 'string',
  `alg=${String(mIrr.algebraica)}, geom=${mIrr.geometrica}`,
)

// ---------------------------------------------------------------------------
// 7. Fronteras: 1×1, matriz no cuadrada, no mutación
// ---------------------------------------------------------------------------
const a1x1 = mat([[7]])
check(
  'Caso 1×1 funciona en charPoly, rationalRoots, esDefectiva, powerMethod, qrIteration y jacobiEigen',
  ratListEq(charPoly(a1x1), [1, -7]) &&
    ratListEq(rationalRoots(charPoly(a1x1)), [7]) &&
    esDefectiva(a1x1) === false &&
    powerMethod([[7]], [2], 1e-10, 10).lambda === 7 &&
    qrIteration([[7]], 1e-10, 10).valores[0] === 7 &&
    jacobiEigen([[7]]).valores[0] === 7,
)

const rectExact = mat([
  [1, 2, 3],
  [4, 5, 6],
])
const rectNum = [
  [1, 2, 3],
  [4, 5, 6],
]
check(
  'Matriz no cuadrada lanza Error claro en charPoly, esDefectiva, powerMethod, qrIteration y jacobiEigen',
  throwsErrorMatching(() => charPoly(rectExact), (m) => m.includes('cuadrada')) &&
    throwsErrorMatching(() => esDefectiva(rectExact), (m) => m.includes('cuadrada')) &&
    throwsErrorMatching(() => powerMethod(rectNum, [1, 1], 1e-6, 10), (m) => m.includes('cuadrada')) &&
    throwsErrorMatching(() => qrIteration(rectNum, 1e-6, 10), (m) => m.includes('cuadrada')) &&
    throwsErrorMatching(() => jacobiEigen(rectNum), (m) => m.includes('cuadrada')),
)

// Verificar que ninguna función muta la matriz de entrada
const frozenNum = [
  Object.freeze([5, 0, 2]),
  Object.freeze([0, 3, 2]),
  Object.freeze([2, 2, 4]),
] as unknown as number[][]
Object.freeze(frozenNum)
const frozenX0 = Object.freeze([1, 2, 3]) as unknown as number[]
let noMutation = true
try {
  powerMethod(frozenNum, frozenX0, 1e-8, 50)
  qrIteration(frozenNum, 1e-8, 50)
  jacobiEigen(frozenNum)
} catch {
  noMutation = false
}
check('powerMethod, qrIteration y jacobiEigen no mutan las matrices ni vectores de entrada', noMutation)

// 6×6 simétrica tridiagonal-bloque para validar el límite máximo de la rejilla (6×6)
const a6x6Exact = mat([
  [2, 1, 0, 0, 0, 0],
  [1, 2, 0, 0, 0, 0],
  [0, 0, 5, 0, 2, 0],
  [0, 0, 0, 3, 2, 0],
  [0, 0, 2, 2, 4, 0],
  [0, 0, 0, 0, 0, -2],
])
// Bloques: [[2,1],[1,2]] (λ=1,3), [[5,0,2],[0,3,2],[2,2,4]] (λ=1,4,7), [-2] (λ=-2)
// Raíces distintas: [-2, 1, 3, 4, 7] (con λ=1 doble)
const cp6 = charPoly(a6x6Exact)
const roots6 = rationalRoots(cp6)
const a6Num = a6x6Exact.map((r) => r.map((v) => v.toNumber()))
const jac6 = jacobiEigen(a6Num)
check(
  'Caso 6×6 exacto: charPoly (grado 6), rationalRoots = [-2, 1, 3, 4, 7], no defectiva y Jacobi con residuo < 1e-10',
  cp6.length === 7 &&
    ratListEq(roots6, [-2, 1, 3, 4, 7]) &&
    multiplicidades(a6x6Exact, Rat.ONE).algebraica === 2 &&
    multiplicidades(a6x6Exact, Rat.ONE).geometrica === 2 &&
    esDefectiva(a6x6Exact) === false &&
    jac6.residuo < 1e-10 &&
    jac6.residuoOrtogonalidad < 1e-10,
  `roots=[${roots6.map(ratToString).join(', ')}], resid=${jac6.residuo.toExponential(2)}`,
)

// ---------------------------------------------------------------------------
// 8. Formato y mensajes de UI (src/uiEigen.ts)
// ---------------------------------------------------------------------------
const irrCells = formatNumericMultiplicityCells({ algebraica: null, geometrica: 1 })
check(
  'UI multiplicidades no racionales: algebraica «—» (sin null) y geométrica estimada',
  irrCells.algText === '—' &&
    !irrCells.algText.toLowerCase().includes('null') &&
    irrCells.geomText.includes('≈ 1') &&
    irrCells.geomText.includes('estimada') &&
    !NON_RATIONAL_MULT_NOTE.toLowerCase().includes('null'),
  `alg="${irrCells.algText}", geom="${irrCells.geomText}"`,
)

const missingMultRoot = formatRootWithMultiplicity(Rat.fromInt(3), undefined)
const doubleMultRoot = formatRootWithMultiplicity(Rat.ONE, 2)
const singleMultRoot = formatRootWithMultiplicity(Rat.fromInt(5), 1)
check(
  'UI raíces racionales: clave faltante muestra «—» en lugar de inventar multiplicidad 1',
  missingMultRoot.plain === '3 (mult. —)' &&
    missingMultRoot.latex.includes('—') &&
    doubleMultRoot.plain === '1 (mult. 2)' &&
    singleMultRoot.plain === '5',
  `missing="${missingMultRoot.plain}", double="${doubleMultRoot.plain}", single="${singleMultRoot.plain}"`,
)

check(
  'UI formatEigenErrorMessage conserva errores en español y traduce fallos internos a español',
  formatEigenErrorMessage(new Error('La matriz es singular en el paso 2.')) ===
    'La matriz es singular en el paso 2.' &&
    formatEigenErrorMessage(new TypeError('Cannot read properties of undefined')) ===
      'No se pudo completar el cálculo.' &&
    formatEigenErrorMessage(new Error('Division by zero')) === 'No se pudo completar el cálculo.' &&
    formatEigenErrorMessage('fallo sin Error') === 'No se pudo completar el cálculo.',
)

const partialNotice = formatIncompleteRationalNotice(1, 3)
const emptyNotice = formatIncompleteRationalNotice(0, 2)
check(
  'UI aviso de polinomio aclara que las secciones numéricas aproximan reales y no separan pares complejos',
  partialNotice.includes('no separan pares complejos') &&
    partialNotice.includes('valores propios reales') &&
    emptyNotice.includes('no separan pares complejos') &&
    !partialNotice.includes('se muestran de forma numérica abajo'),
)

const jacBadgeOk = describeJacobiBadge({
  valores: [7, 4, 1],
  iteraciones: 6,
  residuo: 1e-14,
  residuoOrtogonalidad: 1e-15,
})
const jacBadgeHigh = describeJacobiBadge({
  valores: [7, 4, 1],
  iteraciones: 200,
  residuo: 1e-2,
  residuoOrtogonalidad: 1e-12,
})
check(
  'UI badge de Jacobi refleja el residuo y avisa si es elevado',
  jacBadgeOk.ok === true &&
    jacBadgeOk.text.includes('completada') &&
    jacBadgeHigh.ok === false &&
    jacBadgeHigh.text.includes('residuo elevado'),
  `ok="${jacBadgeOk.text}", high="${jacBadgeHigh.text}"`,
)

check(
  'UI formatNumLatex usa \\times 10^{n} para magnitudes >= 1e5 o < 1e-4 sin dejar la e en LaTeX',
  formatNumLatex(123456.7) === '1.2346 \\times 10^{5}' &&
    formatNumLatex(2.5e-5) === '2.5 \\times 10^{-5}' &&
    formatNumLatex(-3e-6) === '-3 \\times 10^{-6}' &&
    formatNumLatex(1e10) === '1 \\times 10^{10}' &&
    formatNumLatex(3.5) === '3.5' &&
    formatNumLatex(0) === '0',
  `big="${formatNumLatex(123456.7)}", small="${formatNumLatex(2.5e-5)}"`,
)

console.log(`\neigen: ${passed} comprobaciones pasaron${failed ? `, ${failed} fallaron` : ''}.`)
if (failed > 0) process.exit(1)
