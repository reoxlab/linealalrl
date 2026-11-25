// Pruebas del módulo 4: normas vectoriales y matriciales, número de condición κ(A),
// veredicto didáctico, estimación de error, equilibrado de filas y matrices de Hilbert (src/condition.ts).

import {
  type MatrixNorm,
  type VectorNorm,
  condNumber,
  equilibrarFilas,
  estimaErrorSolucion,
  hilbert,
  normMatrix,
  normVector,
  veredictoCondicion,
} from '../src/condition.ts'
import { gaussInverseNum } from '../src/direct.ts'
import { fromStrings, det } from '../src/matrix.ts'
import { formatCondNum, formatCondNumLatex, verdictBadgeLabel } from '../src/uiCondition.ts'

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

function close(actual: number, expected: number, tol: number): boolean {
  return Number.isFinite(actual) && Math.abs(actual - expected) <= tol
}

function relClose(actual: number, expected: number, relTol: number): boolean {
  return (
    Number.isFinite(actual) &&
    Math.abs(actual - expected) <= relTol * Math.max(1, Math.abs(expected))
  )
}

function throwsMatching(fn: () => unknown, pred: (msg: string) => boolean): boolean {
  try {
    fn()
    return false
  } catch (err) {
    return err instanceof Error && pred(err.message)
  }
}

function identityNum(n: number): number[][] {
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)),
  )
}

// ---------------------------------------------------------------------------
// 1. Normas vectoriales
// ---------------------------------------------------------------------------
check('normVector([3, -4], 1) = 7', close(normVector([3, -4], 1), 7, 1e-12))
check('normVector([3, -4], 2) = 5', close(normVector([3, -4], 2), 5, 1e-12))
check('normVector([3, -4], inf) = 4', close(normVector([3, -4], 'inf'), 4, 1e-12))
check('normVector([0, 0, 0], 2) = 0', normVector([0, 0, 0], 2) === 0)
check('normVector([], 1) = 0', normVector([], 1) === 0)
check(
  'normVector rechaza valores no finitos (NaN / Infinity)',
  throwsMatching(() => normVector([1, Number.NaN], 2), (m) => /no finito/i.test(m)) &&
    throwsMatching(() => normVector([1, Number.POSITIVE_INFINITY], 'inf'), (m) => /no finito/i.test(m)),
)
check(
  'normVector rechaza valores inválidos de p con error en español (0, -1, 3, "fro", vector vacío)',
  ([0, -1, 3, 'fro', 'invalido'] as unknown[]).every((badP) =>
    throwsMatching(() => normVector([3, -4], badP as VectorNorm), (m) => /norma vectorial no válida/i.test(m)),
  ) &&
    throwsMatching(() => normVector([], 3 as unknown as VectorNorm), (m) => /norma vectorial no válida/i.test(m)),
)

// ---------------------------------------------------------------------------
// 2. Normas matriciales de [[1,2],[3,4]]
// ---------------------------------------------------------------------------
const a2 = [
  [1, 2],
  [3, 4],
]
const n1 = normMatrix(a2, 1)
const nInf = normMatrix(a2, 'inf')
const nFro = normMatrix(a2, 'fro')
const n2 = normMatrix(a2, 2)
// Valor singular exacto de [[1,2],[3,4]]: sqrt(15 + sqrt(221)) ≈ 5.464985704219043
const expectedSigma2 = Math.sqrt(15 + Math.sqrt(221))

check(`‖[[1,2],[3,4]]‖₁ = 6`, close(n1, 6, 1e-9), `(obtenido ${n1})`)
check(`‖[[1,2],[3,4]]‖∞ = 7`, close(nInf, 7, 1e-9), `(obtenido ${nInf})`)
check(
  `‖[[1,2],[3,4]]‖_F = √30 ≈ 5.477`,
  close(nFro, Math.sqrt(30), 1e-9),
  `(obtenido ${nFro})`,
)
check(
  `‖[[1,2],[3,4]]‖₂ ≈ 5.465 (tol 1e-6)`,
  close(n2, expectedSigma2, 1e-6) && close(n2, 5.4649857, 1e-6),
  `(obtenido ${n2}, esperado ${expectedSigma2})`,
)

// Matriz rectangular y matriz nula
const rect = [
  [1, -2, 3],
  [4, 0, -1],
]
check('normMatrix rectangular 2×3: ‖A‖₁ = 5', close(normMatrix(rect, 1), 5, 1e-12))
check('normMatrix rectangular 2×3: ‖A‖∞ = 6', close(normMatrix(rect, 'inf'), 6, 1e-12))
check(
  'normMatrix rectangular 2×3: ‖A‖_F = √31',
  close(normMatrix(rect, 'fro'), Math.sqrt(31), 1e-12),
)
const expectedSigmaRect = Math.sqrt((31 + Math.sqrt(13)) / 2)
const rectNorm2 = normMatrix(rect, 2)
check(
  'normMatrix rectangular 2×3: ‖A‖₂ = √((31+√13)/2) ≈ 4.15966 (tol 1e-6)',
  close(rectNorm2, expectedSigmaRect, 1e-6) &&
    rectNorm2 <= normMatrix(rect, 'fro') + 1e-12,
  `(obtenido ${rectNorm2}, esperado ${expectedSigmaRect})`,
)
check(
  'normMatrix rechaza valores inválidos de p con error en español (0, -1, 3, "invalido")',
  ([0, -1, 3, 'invalido'] as unknown[]).every((badP) =>
    throwsMatching(() => normMatrix(a2, badP as MatrixNorm), (m) => /norma matricial no válida/i.test(m)),
  ) &&
    throwsMatching(() => condNumber(a2, 3 as unknown as MatrixNorm), (m) => /norma matricial no válida/i.test(m)),
)

// Respaldo determinista con Jacobi en normMatrix(A, 2):
// Matriz 2×2 con valores singulares muy próximos (σ₁ = 1, σ₂ = 0.995) rotada θ = 0.35 rad
// (ninguna semilla del método de la potencia converge en 100 iteraciones).
// Sin Jacobi (cayendo a Frobenius), devolvería √(1 + 0.995²) ≈ 1.41068 en lugar de 1.
const thetaClose = 0.35
const cosT = Math.cos(thetaClose)
const sinT = Math.sin(thetaClose)
const s2Close = 0.995
const closeSingular2 = [
  [cosT * cosT + s2Close * sinT * sinT, cosT * sinT * (1 - s2Close)],
  [cosT * sinT * (1 - s2Close), sinT * sinT + s2Close * cosT * cosT],
]
const closeNorm2 = normMatrix(closeSingular2, 2)
check(
  'normMatrix(A, 2) usa respaldo Jacobi exacto cuando la potencia no converge (σ_max = 1 ≠ ‖A‖_F ≈ 1.411)',
  close(closeNorm2, 1, 1e-10) && Math.abs(closeNorm2 - normMatrix(closeSingular2, 'fro')) > 0.3,
  `(obtenido ${closeNorm2}, fro=${normMatrix(closeSingular2, 'fro')})`,
)
// Bloque 3×3: subespacio dominante 2×2 sin convergencia en 100 iter + eje canónico con σ₃ = 0.5
const closeBlock3 = [
  [closeSingular2[0][0], closeSingular2[0][1], 0],
  [closeSingular2[1][0], closeSingular2[1][1], 0],
  [0, 0, 0.5],
]
const closeBlockNorm2 = normMatrix(closeBlock3, 2)
check(
  'normMatrix(A, 2) en bloque 3×3 no devuelve autovalor subdominante (σ_max = 1, no 0.5 ni ‖A‖_F)',
  close(closeBlockNorm2, 1, 1e-10),
  `(obtenido ${closeBlockNorm2})`,
)
check(
  'normMatrix de matriz nula devuelve 0 en todas las normas',
  normMatrix([[0, 0], [0, 0]], 1) === 0 &&
    normMatrix([[0, 0], [0, 0]], 2) === 0 &&
    normMatrix([[0, 0], [0, 0]], 'inf') === 0 &&
    normMatrix([[0, 0], [0, 0]], 'fro') === 0,
)
check(
  'normMatrix de 1×1 [[-7]] devuelve 7 en todas las normas',
  normMatrix([[-7]], 1) === 7 &&
    normMatrix([[-7]], 2) === 7 &&
    normMatrix([[-7]], 'inf') === 7 &&
    normMatrix([[-7]], 'fro') === 7,
)

// ---------------------------------------------------------------------------
// 3. condNumber: identidad y diagonal [[1,0],[0,2]]
// ---------------------------------------------------------------------------
const id3 = identityNum(3)
check('condNumber(I₃, 1).kappa = 1', close(condNumber(id3, 1).kappa, 1, 1e-12))
check('condNumber(I₃, 2).kappa = 1', close(condNumber(id3, 2).kappa, 1, 1e-12))
check('condNumber(I₃, inf).kappa = 1', close(condNumber(id3, 'inf').kappa, 1, 1e-12))
// Para Frobenius: ‖Iₙ‖_F = √n ⇒ κ_F(Iₙ) = n (y para I₁ es 1 en las 4 normas)
check(
  'condNumber(I₁, fro).kappa = 1 y condNumber(I₃, fro).kappa = 3 (‖I₃‖_F = √3)',
  close(condNumber([[1]], 'fro').kappa, 1, 1e-12) &&
    close(condNumber(id3, 'fro').kappa, 3, 1e-12),
)

const diag12 = [
  [1, 0],
  [0, 2],
]
const cDiagInf = condNumber(diag12, 'inf')
check(
  'condNumber([[1,0],[0,2]], inf).kappa = 2 e inversaNorma = 1',
  close(cDiagInf.kappa, 2, 1e-12) && close(cDiagInf.inversaNorma, 1, 1e-12),
  `(obtenido kappa=${cDiagInf.kappa}, inv=${cDiagInf.inversaNorma})`,
)
check('condNumber([[1,0],[0,2]], 1).kappa = 2', close(condNumber(diag12, 1).kappa, 2, 1e-12))
check('condNumber([[1,0],[0,2]], 2).kappa = 2', close(condNumber(diag12, 2).kappa, 2, 1e-9))

// ---------------------------------------------------------------------------
// 4. veredictoCondicion
// ---------------------------------------------------------------------------
const vBien = veredictoCondicion(1)
const vBienBorde = veredictoCondicion(99.9)
const vModBorde = veredictoCondicion(100)
const vMod = veredictoCondicion(500)
const vModAlto = veredictoCondicion(9999)
const vMalBorde = veredictoCondicion(1e4)
const vMal = veredictoCondicion(1e6)
const vInf = veredictoCondicion(Number.POSITIVE_INFINITY)

check('veredictoCondicion(1).nivel === "bien"', vBien.nivel === 'bien')
check('veredictoCondicion(99.9).nivel === "bien"', vBienBorde.nivel === 'bien')
check('veredictoCondicion(100).nivel === "moderada"', vModBorde.nivel === 'moderada')
check('veredictoCondicion(500).nivel === "moderada"', vMod.nivel === 'moderada')
check('veredictoCondicion(9999).nivel === "moderada"', vModAlto.nivel === 'moderada')
check('veredictoCondicion(1e4).nivel === "mal"', vMalBorde.nivel === 'mal')
check('veredictoCondicion(1e6).nivel === "mal"', vMal.nivel === 'mal')
check('veredictoCondicion(Infinity).nivel === "mal"', vInf.nivel === 'mal')
check(
  'veredictoCondicion explica en palabras llanas la pérdida de dígitos decimales (κ ~ 10ᵏ)',
  /10ᵏ/.test(vBien.texto) &&
    /dígitos/i.test(vMod.texto) &&
    /dígitos/i.test(vMal.texto) &&
    /singular/i.test(vInf.texto),
)

// ---------------------------------------------------------------------------
// 5. estimaErrorSolucion
// ---------------------------------------------------------------------------
const est = estimaErrorSolucion(100, 1e-6)
check(
  'estimaErrorSolucion(100, 1e-6) = 1e-4',
  close(est, 1e-4, 1e-15),
  `(obtenido ${est})`,
)
check(
  'estimaErrorSolucion(Infinity, 1e-6) = Infinity',
  estimaErrorSolucion(Number.POSITIVE_INFINITY, 1e-6) === Number.POSITIVE_INFINITY,
)
check(
  'estimaErrorSolucion(Infinity, 0) = Infinity (sin NaN)',
  estimaErrorSolucion(Number.POSITIVE_INFINITY, 0) === Number.POSITIVE_INFINITY,
)
check(
  'estimaErrorSolucion rechaza error relativo negativo',
  throwsMatching(() => estimaErrorSolucion(10, -1e-6), (m) => /no negativo/i.test(m)),
)

// ---------------------------------------------------------------------------
// 6. Matrices de Hilbert H₃ y H₄
// ---------------------------------------------------------------------------
const h3 = hilbert(3)
const h4 = hilbert(4)
check(
  'hilbert(3) cumple H[i][j] = 1/(i+j+1)',
  h3.length === 3 &&
    close(h3[0][0], 1, 1e-15) &&
    close(h3[0][2], 1 / 3, 1e-15) &&
    close(h3[2][2], 1 / 5, 1e-15),
)
check(
  'hilbert rechaza n < 1 o no entero',
  throwsMatching(() => hilbert(0), (m) => /entero/i.test(m)) &&
    throwsMatching(() => hilbert(2.5), (m) => /entero/i.test(m)),
)

const kH3 = condNumber(h3, 2).kappa
const kH4 = condNumber(h4, 2).kappa
check(
  `Hilbert 3×3: κ₂ ≈ 524.057 (tolerancia 1e-4)`,
  relClose(kH3, 524.0567775861, 1e-4),
  `(obtenido ${kH3})`,
)
check(
  `Hilbert 4×4: κ₂ ≈ 15513.74 (tolerancia 1e-4)`,
  relClose(kH4, 15513.7387389281, 1e-4),
  `(obtenido ${kH4})`,
)
check(
  'Veredictos de Hilbert: H₃ es moderada y H₄ es mal condicionada',
  veredictoCondicion(kH3).nivel === 'moderada' && veredictoCondicion(kH4).nivel === 'mal',
)

// ---------------------------------------------------------------------------
// 7. Matrices singulares -> { kappa: Infinity }
// ---------------------------------------------------------------------------
const sing2 = [
  [1, 2],
  [2, 4],
]
const sing3 = [
  [1, 2, 3],
  [4, 5, 6],
  [7, 8, 9],
]
check(
  'Singular 2×2 devuelve { kappa: Infinity, inversaNorma: Infinity } en las 4 normas',
  ([1, 2, 'inf', 'fro'] as const).every((p) => {
    const res = condNumber(sing2, p)
    return res.kappa === Number.POSITIVE_INFINITY && res.inversaNorma === Number.POSITIVE_INFINITY
  }),
)
check(
  'Singular 3×3 (rango 2) devuelve { kappa: Infinity, inversaNorma: Infinity }',
  ([2, 'inf'] as const).every((p) => {
    const res = condNumber(sing3, p)
    return res.kappa === Number.POSITIVE_INFINITY && res.inversaNorma === Number.POSITIVE_INFINITY
  }),
)
const resZero = condNumber([[0, 0], [0, 0]], 2)
check(
  'Matriz nula devuelve { kappa: Infinity, inversaNorma: Infinity }',
  resZero.kappa === Number.POSITIVE_INFINITY && resZero.inversaNorma === Number.POSITIVE_INFINITY,
)
check('gaussInverseNum devuelve null en matriz singular', gaussInverseNum(sing2) === null)

// ---------------------------------------------------------------------------
// 8. Aviso didáctico (det = 1e-6 pero κ = 1), equilibrado de filas y no mutación
// ---------------------------------------------------------------------------
const smallDetMat = [
  [0.01, 0, 0],
  [0, 0.01, 0],
  [0, 0, 0.01],
]
const smallDetExact = fromStrings([
  ['0.01', '0', '0'],
  ['0', '0.01', '0'],
  ['0', '0', '0.01'],
])!
check(
  'det(0.01·I₃) = 1e-6 exacto pero κ₁ = κ₂ = κ∞ = 1',
  close(det(smallDetExact).valor.toNumber(), 1e-6, 1e-18) &&
    close(condNumber(smallDetMat, 1).kappa, 1, 1e-12) &&
    close(condNumber(smallDetMat, 2).kappa, 1, 1e-12) &&
    close(condNumber(smallDetMat, 'inf').kappa, 1, 1e-12),
)

// Equilibrado de filas en matriz mal escalada
const badlyScaled = [
  [1000, 1000, 0],
  [0, 1, 1],
  [0, 0, 0.001],
]
const kBefore = condNumber(badlyScaled, 'inf').kappa
const eq = equilibrarFilas(badlyScaled)
const kAfter = condNumber(eq.matriz, 'inf').kappa
check(
  'equilibrarFilas divide cada fila por su máximo absoluto y reduce κ drásticamente',
  close(eq.escalas[0], 1000, 1e-12) &&
    close(eq.escalas[1], 1, 1e-12) &&
    close(eq.escalas[2], 0.001, 1e-12) &&
    kBefore > 1e5 &&
    kAfter < 10,
  `(κ antes=${kBefore}, κ después=${kAfter})`,
)

// Fila de ceros en equilibrarFilas no divide entre 0 ni genera NaN
const withZeroRow = [
  [0, 0],
  [2, -4],
]
const eqZero = equilibrarFilas(withZeroRow)
check(
  'equilibrarFilas mantiene intacta una fila nula sin producir NaN',
  eqZero.escalas[0] === 1 &&
    eqZero.escalas[1] === 4 &&
    eqZero.matriz[0][0] === 0 &&
    eqZero.matriz[0][1] === 0 &&
    close(eqZero.matriz[1][0], 0.5, 1e-12) &&
    close(eqZero.matriz[1][1], -1, 1e-12),
)

// Errores claros en matriz no cuadrada y no mutación
check(
  'condNumber rechaza matriz no cuadrada con error en español',
  throwsMatching(() => condNumber(rect, 2), (m) => /cuadrada/i.test(m)),
)

const snap = a2.map((r) => r.slice())
normMatrix(a2, 2)
condNumber(a2, 2)
equilibrarFilas(a2)
check(
  'normMatrix, condNumber y equilibrarFilas no mutan la matriz de entrada',
  a2.every((row, i) => row.every((val, j) => val === snap[i][j])),
)

// Utilidades de formato de uiCondition.ts
check(
  'formatCondNum y formatCondNumLatex manejan finitos e infinito',
  formatCondNum(Number.POSITIVE_INFINITY) === '∞' &&
    formatCondNumLatex(Number.POSITIVE_INFINITY) === '\\infty' &&
    formatCondNumLatex(1e-6) === '1 \\times 10^{-6}' &&
    verdictBadgeLabel('bien').includes('Bien'),
)

console.log(`\ncondition: ${passed} comprobaciones pasaron${failed ? `, ${failed} fallaron` : ''}.`)
if (failed > 0) process.exit(1)
