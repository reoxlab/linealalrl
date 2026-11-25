// Pruebas de los solucionadores directos (src/direct.ts): Gauss, LU, Cholesky, QR y comparador.
import {
  choleskyFactor,
  choleskySolve,
  directCompare,
  gaussSolve,
  luFactor,
  luSolve,
  qrHouseholder,
  qrSolve,
} from '../src/direct.ts'
import { type Matrix, identity, multiply } from '../src/matrix.ts'
import { Rat } from '../src/rational.ts'

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

function throwsErrorMatching(fn: () => unknown, pred: (msg: string) => boolean): boolean {
  try {
    fn()
    return false
  } catch (err) {
    return err instanceof Error && pred(err.message)
  }
}

function isSpanishSingularError(msg: string): boolean {
  const lower = msg.toLowerCase()
  return lower.includes('singular') && lower.includes('matriz') && lower.includes('solución única')
}

function isSpanishDimsError(msg: string, bLen: number): boolean {
  const lower = msg.toLowerCase()
  return (
    lower.includes('dimensiones incompatibles') &&
    msg.includes('2×2') &&
    lower.includes(`vector b tiene ${bLen} entradas`)
  )
}

function freezeMatrix(m: Matrix): void {
  for (const row of m) {
    Object.freeze(row)
  }
  Object.freeze(m)
}

function mat(rows: number[][]): Matrix {
  return rows.map((row) => row.map((v) => new Rat(v, 1)))
}

function vec(vals: number[]): Rat[] {
  return vals.map((v) => new Rat(v, 1))
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

function vecEq(a: Rat[], b: Rat[]): boolean {
  if (a.length !== b.length) return false
  return a.every((v, i) => v.eq(b[i]))
}

function numVecClose(a: number[], b: number[], tol = 1e-10): boolean {
  if (a.length !== b.length) return false
  return a.every((v, i) => Math.abs(v - b[i]) < tol)
}

function numMatMul(a: number[][], b: number[][]): number[][] {
  const m = a.length
  const k = a[0].length
  const n = b[0].length
  const out = Array.from({ length: m }, () => Array.from({ length: n }, () => 0))
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      let s = 0
      for (let t = 0; t < k; t++) s += a[i][t] * b[t][j]
      out[i][j] = s
    }
  }
  return out
}

function numTranspose(a: number[][]): number[][] {
  return Array.from({ length: a[0].length }, (_, j) =>
    Array.from({ length: a.length }, (_, i) => a[i][j]),
  )
}

function frobeniusDiff(a: number[][], b: number[][]): number {
  let s = 0
  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < a[i].length; j++) {
      const d = a[i][j] - b[i][j]
      s += d * d
    }
  }
  return Math.sqrt(s)
}

// ---------------------------------------------------------------------------
// 1. Eliminación de Gauss exacta
// ---------------------------------------------------------------------------
const a2 = mat([
  [2, 1],
  [1, 3],
])
const b2 = vec([5, 10])
const g2 = gaussSolve(a2, b2)
check(
  'Gauss resuelve [[2,1],[1,3]] x = [5,10] exactamente -> x = [1, 3]',
  vecEq(g2.x, vec([1, 3])) && g2.pasos.length > 0,
)

// Intercambio de filas necesario (pivote cero en a_11)
const aSwap = mat([
  [0, 2, 1],
  [1, -2, -3],
  [-1, 1, 2],
])
const bSwap = vec([-8, 0, 3])
const gSwap = gaussSolve(aSwap, bSwap)
check(
  'Gauss con pivote cero en a11 intercambia filas y resuelve exacto x = [-4, -5, 2]',
  vecEq(gSwap.x, vec([-4, -5, 2])) && gSwap.pasos.some((p) => p.texto.includes('↔')),
)

// Preset 3×3: A = [[2,1,-1],[-3,-1,2],[-2,1,2]], b = [8,-11,-3] -> x = [2, 3, -1]
const aPreset = mat([
  [2, 1, -1],
  [-3, -1, 2],
  [-2, 1, 2],
])
const bPreset = vec([8, -11, -3])
const gPreset = gaussSolve(aPreset, bPreset)
check(
  'Gauss resuelve el preset 3×3 exactamente -> x = [2, 3, -1]',
  vecEq(gPreset.x, vec([2, 3, -1])),
)

// Singular -> lanza Error
const aSing = mat([
  [1, 2, 3],
  [4, 5, 6],
  [5, 7, 9],
])
check(
  'Gauss lanza Error en español indicando matriz singular',
  throwsErrorMatching(() => gaussSolve(aSing, vec([1, 2, 3])), isSpanishSingularError),
)
check(
  'Gauss lanza error en matriz no cuadrada',
  throws(() =>
    gaussSolve(
      mat([
        [1, 2, 3],
        [4, 5, 6],
      ]),
      vec([1, 2]),
    ),
  ),
)

// ---------------------------------------------------------------------------
// 2. Factorización LU (PA = LU) y luSolve
// ---------------------------------------------------------------------------
const lu3 = luFactor(aPreset)
const pa = multiply(lu3.P, aPreset)
const luProd = multiply(lu3.L, lu3.U)
const isLowerUnit = lu3.L.every((row, i) =>
  row.every((v, j) => (i === j ? v.isOne() : j > i ? v.isZero() : true)),
)
const isUpper = lu3.U.every((row, i) => row.every((v, j) => (i > j ? v.isZero() : true)))
check('LU en 3×3 verifica P·A = L·U exactamente', matEq(pa, luProd))
check('LU produce L triangular inferior unitaria y U triangular superior', isLowerUnit && isUpper)

// LU con pivote cero en a11 también cumple P·A = L·U exacto
const luSwap = luFactor(aSwap)
check(
  'LU con a11 = 0 verifica P·A = L·U exactamente',
  matEq(multiply(luSwap.P, aSwap), multiply(luSwap.L, luSwap.U)),
)

const luSolPreset = luSolve(aPreset, bPreset)
check('luSolve coincide exactamente con Gauss en el preset 3×3', vecEq(luSolPreset.x, gPreset.x))

const luSolSwap = luSolve(aSwap, bSwap)
check('luSolve coincide exactamente con Gauss cuando a11 = 0', vecEq(luSolSwap.x, gSwap.x))

// Verificar Ly = Pb y Ux = y exactamente
const ly = multiply(
  luSolPreset.L,
  luSolPreset.y.map((v) => [v]),
).map((r) => r[0])
const ux = multiply(
  luSolPreset.U,
  luSolPreset.x.map((v) => [v]),
).map((r) => r[0])
check(
  'luSolve verifica exactamente L·y = P·b y U·x = y',
  vecEq(ly, luSolPreset.pb) && vecEq(ux, luSolPreset.y),
)

check(
  'luSolve lanza Error en español indicando matriz singular',
  throwsErrorMatching(() => luSolve(aSing, vec([1, 2, 3])), isSpanishSingularError),
)

// ---------------------------------------------------------------------------
// 3. Cholesky (numérico, solo simétricas definidas positivas)
// ---------------------------------------------------------------------------
const aSpd2 = mat([
  [4, 2],
  [2, 3],
])
const chol2 = choleskyFactor(aSpd2)
check(
  'Cholesky factoriza [[4,2],[2,3]] con l11 = 2, l21 = 1, l22 = √2 y residuo < 1e-12',
  !('error' in chol2) &&
    Math.abs(chol2.L[0][0] - 2) < 1e-12 &&
    Math.abs(chol2.L[0][1]) < 1e-12 &&
    Math.abs(chol2.L[1][0] - 1) < 1e-12 &&
    Math.abs(chol2.L[1][1] - Math.SQRT2) < 1e-12 &&
    chol2.residuo < 1e-12,
)

const cholSol2 = choleskySolve(aSpd2, vec([8, 7])) // 4(5/4) + 2(3/2) = 5 + 3 = 8, 2(5/4) + 3(3/2) = 5/2 + 9/2 = 7 => x = [5/4, 3/2] = [1.25, 1.5]
check(
  'choleskySolve resuelve [[4,2],[2,3]] x = [8,7] -> [1.25, 1.5] con residuo < 1e-10',
  !('error' in cholSol2) &&
    numVecClose(cholSol2.x, [1.25, 1.5], 1e-10) &&
    cholSol2.residuo < 1e-10,
)

// No simétrica -> error explicando simetría y sugiriendo LU
const cholNoSym = choleskyFactor(aPreset)
check(
  'Cholesky rechaza matriz no simétrica con error claro sugiriendo LU',
  'error' in cholNoSym &&
    cholNoSym.error.toLowerCase().includes('simétrica') &&
    cholNoSym.error.includes('LU'),
)

// Simétrica no definida positiva [[1,2],[2,1]] -> error explicando pivote / definida positiva y LU
const aIndef = mat([
  [1, 2],
  [2, 1],
])
const cholIndef = choleskyFactor(aIndef)
check(
  'Cholesky rechaza [[1,2],[2,1]] (simétrica indefinida) indicando pivote 2 ≤ 0 y sugiriendo LU',
  'error' in cholIndef &&
    cholIndef.error.includes('pivote 2') &&
    cholIndef.error.toLowerCase().includes('definida positiva') &&
    cholIndef.error.includes('LU'),
)

// Simétrica semidefinida positiva [[1,2],[2,4]] -> error
const cholSemi = choleskyFactor(
  mat([
    [1, 2],
    [2, 4],
  ]),
)
check('Cholesky rechaza matriz simétrica semidefinida positiva', 'error' in cholSemi)

// ---------------------------------------------------------------------------
// 4. QR por reflexiones de Householder
// ---------------------------------------------------------------------------
const qr3 = qrHouseholder(aPreset)
const qtq = numMatMul(numTranspose(qr3.Q), qr3.Q)
const id3Num = identity(3).map((r) => r.map((v) => v.toNumber()))
const aPresetNum = aPreset.map((r) => r.map((v) => v.toNumber()))
const qrProd = numMatMul(qr3.Q, qr3.R)

check('QR verifica QᵀQ ≈ I con error < 1e-10', frobeniusDiff(qtq, id3Num) < 1e-10)
check(
  'QR verifica A ≈ Q·R con residuo < 1e-10',
  qr3.residuoQR < 1e-10 && frobeniusDiff(aPresetNum, qrProd) < 1e-10,
)
const rIsUpper = qr3.R.every((row, i) =>
  row.every((v, j) => (i > j ? Math.abs(v) < 1e-12 : true)),
)
check('QR produce R triangular superior', rIsUpper)

const qrSolPreset = qrSolve(aPreset, bPreset)
check(
  'qrSolve resuelve el preset 3×3 -> [2, 3, -1] con residuo < 1e-10',
  numVecClose(qrSolPreset.x, [2, 3, -1], 1e-10) && qrSolPreset.residuo < 1e-10,
)

const qrSolSwap = qrSolve(aSwap, bSwap)
check(
  'qrSolve resuelve sistema con a11 = 0 -> [-4, -5, 2] con residuo < 1e-10',
  numVecClose(qrSolSwap.x, [-4, -5, 2], 1e-10) && qrSolSwap.residuo < 1e-10,
)

check(
  'qrSolve lanza Error en español indicando matriz singular',
  throwsErrorMatching(() => qrSolve(aSing, vec([1, 2, 3])), isSpanishSingularError),
)

// ---------------------------------------------------------------------------
// 5. Comparador directo (directCompare)
// ---------------------------------------------------------------------------
// Caso A: Matriz SPD 3×3 -> los 4 métodos aplican y coinciden
const aSpd3 = mat([
  [4, 2, -2],
  [2, 10, 2],
  [-2, 2, 6],
])
const bSpd3 = vec([10, 20, -4]) // x = [1, 2, -1]
const cmpSpd = directCompare(aSpd3, bSpd3)
const expectedSpd = [1, 2, -1]
check(
  'directCompare en SPD 3×3 devuelve las 4 filas todas aplicables',
  cmpSpd.length === 4 && cmpSpd.every((r) => r.aplicable),
)
check(
  'directCompare en SPD 3×3: las 4 soluciones coinciden dentro de tolerancia 1e-10',
  cmpSpd.every((r) => {
    const asNums = r.exacto ? (r.x as Rat[]).map((v) => v.toNumber()) : (r.x as number[])
    return numVecClose(asNums, expectedSpd, 1e-10) && r.residuo < 1e-10
  }),
)

// Caso B: Preset 3×3 no simétrico -> 3 aplican + fila de Cholesky explica motivo sin fallar
const cmpPreset = directCompare(aPreset, bPreset)
const cholRow = cmpPreset.find((r) => r.clave === 'cholesky')
const applicablePreset = cmpPreset.filter((r) => r.aplicable)
check(
  'directCompare en matriz no simétrica devuelve 4 filas (3 aplicables + Cholesky con motivo)',
  cmpPreset.length === 4 &&
    applicablePreset.length === 3 &&
    cholRow !== undefined &&
    !cholRow.aplicable &&
    typeof cholRow.motivo === 'string' &&
    cholRow.motivo.toLowerCase().includes('simétrica'),
)
check(
  'directCompare en preset 3×3: Gauss, LU y QR coinciden en [2, 3, -1]',
  applicablePreset.every((r) => {
    const asNums = r.exacto ? (r.x as Rat[]).map((v) => v.toNumber()) : (r.x as number[])
    return numVecClose(asNums, [2, 3, -1], 1e-10)
  }),
)
check(
  'directCompare incluye costo aproximado (~n³/3, ~n³/6, ~2n³/3) y cuándo usar en todas las filas',
  cmpPreset.every((r) => r.operaciones.includes('n³') && r.cuandoUsar.length > 10),
)

// ---------------------------------------------------------------------------
// 6. Casos borde: sistemas 1×1, no mutación de A y vector b con longitud inválida
// ---------------------------------------------------------------------------
// 6.1 Sistemas 1×1 con A = [[4]], b = [8]
const a1 = mat([[4]])
const b1 = vec([8])

const g1 = gaussSolve(a1, b1)
check('Caso 1×1 (Gauss): [[4]] x = [8] da exactamente x = [2]', vecEq(g1.x, vec([2])))

const lu1 = luFactor(a1)
const luSol1 = luSolve(a1, b1)
check(
  'Caso 1×1 (LU): [[4]] x = [8] verifica P·A = L·U y da exactamente x = [2]',
  matEq(multiply(lu1.P, a1), multiply(lu1.L, lu1.U)) && vecEq(luSol1.x, vec([2])),
)

const cholFac1 = choleskyFactor(a1)
const cholSol1 = choleskySolve(a1, b1)
check(
  'Caso 1×1 (Cholesky): [[4]] da L = [[2]] y x ≈ [2]',
  !('error' in cholFac1) &&
    cholFac1.L.length === 1 &&
    cholFac1.L[0].length === 1 &&
    Math.abs(cholFac1.L[0][0] - 2) < 1e-12 &&
    cholFac1.residuo < 1e-12 &&
    !('error' in cholSol1) &&
    numVecClose(cholSol1.x, [2], 1e-10) &&
    cholSol1.residuo < 1e-10,
)

const qr1 = qrHouseholder(a1)
const qtq1 = numMatMul(numTranspose(qr1.Q), qr1.Q)
const qrProd1 = numMatMul(qr1.Q, qr1.R)
const qrSol1 = qrSolve(a1, b1)
check(
  'Caso 1×1 (QR): [[4]] cumple QᵀQ ≈ [[1]], A ≈ QR y x ≈ [2]',
  frobeniusDiff(qtq1, [[1]]) < 1e-10 &&
    frobeniusDiff([[4]], qrProd1) < 1e-10 &&
    qr1.residuoQR < 1e-10 &&
    numVecClose(qrSol1.x, [2], 1e-10) &&
    qrSol1.residuo < 1e-10,
)

// 6.2 No mutación de A en gaussSolve y luFactor (con pivoteo y eliminación)
const aGaussImmutable = mat([
  [0, 2, 1],
  [1, -2, -3],
  [-1, 1, 2],
])
const aGaussSnapshot = mat([
  [0, 2, 1],
  [1, -2, -3],
  [-1, 1, 2],
])
freezeMatrix(aGaussImmutable)
gaussSolve(aGaussImmutable, vec([-8, 0, 3]))
check('gaussSolve no muta la matriz A suministrada', matEq(aGaussImmutable, aGaussSnapshot))

const aLuImmutable = mat([
  [0, 2, 1],
  [1, -2, -3],
  [-1, 1, 2],
])
const aLuSnapshot = mat([
  [0, 2, 1],
  [1, -2, -3],
  [-1, 1, 2],
])
freezeMatrix(aLuImmutable)
luFactor(aLuImmutable)
check('luFactor no muta la matriz A suministrada', matEq(aLuImmutable, aLuSnapshot))

// 6.3 Vector b de longitud incorrecta (1 y 3) para A = [[2,1],[1,3]]
const bShort = vec([5])
const bLong = vec([5, 10, 1])

check(
  'gaussSolve rechaza b de longitud 1 con error español de dimensiones',
  throwsErrorMatching(() => gaussSolve(a2, bShort), (msg) => isSpanishDimsError(msg, 1)),
)
check(
  'gaussSolve rechaza b de longitud 3 con error español de dimensiones',
  throwsErrorMatching(() => gaussSolve(a2, bLong), (msg) => isSpanishDimsError(msg, 3)),
)

check(
  'luSolve rechaza b de longitud 1 con error español de dimensiones',
  throwsErrorMatching(() => luSolve(a2, bShort), (msg) => isSpanishDimsError(msg, 1)),
)
check(
  'luSolve rechaza b de longitud 3 con error español de dimensiones',
  throwsErrorMatching(() => luSolve(a2, bLong), (msg) => isSpanishDimsError(msg, 3)),
)

const cholShort = choleskySolve(a2, bShort)
check(
  'choleskySolve rechaza b de longitud 1 con error español de dimensiones',
  'error' in cholShort && isSpanishDimsError(cholShort.error, 1),
)
const cholLong = choleskySolve(a2, bLong)
check(
  'choleskySolve rechaza b de longitud 3 con error español de dimensiones',
  'error' in cholLong && isSpanishDimsError(cholLong.error, 3),
)

check(
  'qrSolve rechaza b de longitud 1 con error español de dimensiones',
  throwsErrorMatching(() => qrSolve(a2, bShort), (msg) => isSpanishDimsError(msg, 1)),
)
check(
  'qrSolve rechaza b de longitud 3 con error español de dimensiones',
  throwsErrorMatching(() => qrSolve(a2, bLong), (msg) => isSpanishDimsError(msg, 3)),
)

check(
  'directCompare rechaza b de longitud 1 con error español de dimensiones',
  throwsErrorMatching(() => directCompare(a2, bShort), (msg) => isSpanishDimsError(msg, 1)),
)
check(
  'directCompare rechaza b de longitud 3 con error español de dimensiones',
  throwsErrorMatching(() => directCompare(a2, bLong), (msg) => isSpanishDimsError(msg, 3)),
)

if (failed > 0) process.exit(1)
console.log(`\ndirect: ${passed} comprobaciones pasaron.`)
