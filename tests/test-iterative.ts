// Pruebas de los solucionadores iterativos (src/iterative.ts):
// Jacobi, Gauss-Seidel, SOR, Gradiente conjugado, dominancia diagonal, radio espectral y ω óptimo.
import {
  conjugateGradient,
  esDiagonalDominante,
  estimaRadioEspectralJacobi,
  gaussSeidel,
  jacobi,
  omegaOptimo,
  residualNorm2,
  sor,
} from '../src/iterative.ts'

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

function throwsErrorMatching(fn: () => unknown, pred: (msg: string) => boolean): boolean {
  try {
    fn()
    return false
  } catch (err) {
    return err instanceof Error && pred(err.message)
  }
}

function vecClose(a: number[], b: number[], tol = 1e-8): boolean {
  if (a.length !== b.length) return false
  return a.every((v, i) => Math.abs(v - b[i]) < tol)
}

function freezeMatrix(m: number[][]): void {
  for (const row of m) Object.freeze(row)
  Object.freeze(m)
}

// ---------------------------------------------------------------------------
// 1. Preset diagonalmente dominante 3×3: A = [[4,1,1],[1,5,1],[1,1,6]], b = [6,7,8] -> x = [1,1,1]
// ---------------------------------------------------------------------------
const aPreset = [
  [4, 1, 1],
  [1, 5, 1],
  [1, 1, 6],
]
const bPreset = [6, 7, 8]
const x0Preset = [0, 0, 0]
const exactPreset = [1, 1, 1]

const rJacobi = jacobi(aPreset, bPreset, x0Preset, 1e-10, 200)
check(
  'Jacobi converge en el preset diagonalmente dominante a [1, 1, 1] con error < 1e-8',
  rJacobi.convergio &&
    vecClose(rJacobi.x, exactPreset, 1e-8) &&
    residualNorm2(aPreset, rJacobi.x, bPreset) < 1e-8 &&
    rJacobi.iteraciones.length > 0,
)

const rGs = gaussSeidel(aPreset, bPreset, x0Preset, 1e-10, 200)
check(
  'Gauss-Seidel converge en el preset diagonalmente dominante a [1, 1, 1] con error < 1e-8',
  rGs.convergio &&
    vecClose(rGs.x, exactPreset, 1e-8) &&
    residualNorm2(aPreset, rGs.x, bPreset) < 1e-8 &&
    rGs.iteraciones.length < rJacobi.iteraciones.length,
)

const wOptPreset = omegaOptimo(aPreset)
check(
  'omegaOptimo devuelve un valor en (1, 2) para el preset diagonalmente dominante',
  wOptPreset !== null && wOptPreset > 1 && wOptPreset < 2,
  `wOpt=${String(wOptPreset)}`,
)

const rSorOpt = sor(aPreset, bPreset, x0Preset, wOptPreset ?? 1.05, 1e-10, 200)
check(
  'SOR converge en el preset diagonalmente dominante a [1, 1, 1] con error < 1e-8',
  rSorOpt.convergio &&
    vecClose(rSorOpt.x, exactPreset, 1e-8) &&
    residualNorm2(aPreset, rSorOpt.x, bPreset) < 1e-8,
)

// ---------------------------------------------------------------------------
// 2. SOR con ω = 1 equivale a Gauss-Seidel (misma trayectoria e iteraciones)
// ---------------------------------------------------------------------------
const rSor1 = sor(aPreset, bPreset, x0Preset, 1, 1e-10, 200)
const sameTrajectory =
  rSor1.convergio === rGs.convergio &&
  rSor1.iteraciones.length === rGs.iteraciones.length &&
  rSor1.iteraciones.every(
    (step, idx) =>
      step.k === rGs.iteraciones[idx].k &&
      Math.abs(step.error - rGs.iteraciones[idx].error) < 1e-14 &&
      vecClose(step.x, rGs.iteraciones[idx].x, 1e-14),
  )
check(
  'SOR con ω = 1 produce la misma trayectoria y número de iteraciones que Gauss-Seidel',
  sameTrajectory,
)

// Validación de rango de ω en SOR: 0 < ω < 2
check(
  'SOR lanza Error en español cuando ω = 0',
  throwsErrorMatching(
    () => sor(aPreset, bPreset, x0Preset, 0, 1e-6, 100),
    (msg) => msg.includes('0 < ω < 2'),
  ),
)
check(
  'SOR lanza Error en español cuando ω < 0',
  throwsErrorMatching(
    () => sor(aPreset, bPreset, x0Preset, -0.5, 1e-6, 100),
    (msg) => msg.includes('0 < ω < 2'),
  ),
)
check(
  'SOR lanza Error en español cuando ω = 2',
  throwsErrorMatching(
    () => sor(aPreset, bPreset, x0Preset, 2, 1e-6, 100),
    (msg) => msg.includes('0 < ω < 2'),
  ),
)
check(
  'SOR lanza Error en español cuando ω > 2 o NaN',
  throwsErrorMatching(
    () => sor(aPreset, bPreset, x0Preset, 2.3, 1e-6, 100),
    (msg) => msg.includes('0 < ω < 2'),
  ) &&
    throwsErrorMatching(
      () => sor(aPreset, bPreset, x0Preset, Number.NaN, 1e-6, 100),
      (msg) => msg.includes('0 < ω < 2'),
    ),
)

// ---------------------------------------------------------------------------
// 3. Gradiente conjugado (SPD y no SPD)
// ---------------------------------------------------------------------------
const aSpd3 = [
  [4, 2, -2],
  [2, 10, 2],
  [-2, 2, 6],
]
const bSpd3 = [10, 20, -4] // x = [1, 2, -1]
const rCgSpd = conjugateGradient(aSpd3, bSpd3, [0, 0, 0], 1e-10, 50)
check(
  'Gradiente conjugado converge en SPD 3×3 a [1, 2, -1] con error < 1e-8 en ≤ 3 iteraciones',
  !('error' in rCgSpd) &&
    rCgSpd.convergio &&
    vecClose(rCgSpd.x, [1, 2, -1], 1e-8) &&
    residualNorm2(aSpd3, rCgSpd.x, bSpd3) < 1e-8 &&
    rCgSpd.iteraciones.length <= 3,
)

const rCgPreset = conjugateGradient(aPreset, bPreset, x0Preset, 1e-10, 50)
check(
  'Gradiente conjugado converge en el preset SPD 3×3 a [1, 1, 1] con error < 1e-8',
  !('error' in rCgPreset) && rCgPreset.convergio && vecClose(rCgPreset.x, exactPreset, 1e-8),
)

// No simétrica -> devuelve { error }
const rCgNoSym = conjugateGradient(
  [
    [5, 1, 1],
    [0, 4, 1],
    [1, 0, 3],
  ],
  [7, 5, 4],
  [0, 0, 0],
  1e-6,
  50,
)
check(
  'Gradiente conjugado devuelve { error } explicando asimetría en matriz no simétrica',
  'error' in rCgNoSym && rCgNoSym.error.toLowerCase().includes('simétrica'),
)

// Simétrica indefinida [[1, 2], [2, 1]] -> devuelve { error }
const rCgIndef = conjugateGradient(
  [
    [1, 2],
    [2, 1],
  ],
  [5, 4],
  [0, 0],
  1e-6,
  50,
)
check(
  'Gradiente conjugado devuelve { error } en matriz simétrica indefinida (pivote ≤ 0)',
  'error' in rCgIndef && rCgIndef.error.toLowerCase().includes('definida positiva'),
)

// Simétrica semidefinida [[1, 2], [2, 4]] -> devuelve { error }
const rCgSemi = conjugateGradient(
  [
    [1, 2],
    [2, 4],
  ],
  [3, 6],
  [0, 0],
  1e-6,
  50,
)
check(
  'Gradiente conjugado devuelve { error } en matriz simétrica semidefinida positiva',
  'error' in rCgSemi && rCgSemi.error.toLowerCase().includes('definida positiva'),
)

// ---------------------------------------------------------------------------
// 4. Sistema NO diagonalmente dominante: [[1, 2], [3, 4]]
// ---------------------------------------------------------------------------
const aNoDom = [
  [1, 2],
  [3, 4],
]
const bNoDom = [3, 7] // solución exacta [1, 1]

const rJacobiNoDom = jacobi(aNoDom, bNoDom, [0, 0], 1e-6, 100)
check(
  'Jacobi no converge en 100 iteraciones en [[1,2],[3,4]] (convergio: false con motivo)',
  !rJacobiNoDom.convergio &&
    typeof rJacobiNoDom.motivo === 'string' &&
    rJacobiNoDom.motivo.length > 10,
)

const rhoNoDom = estimaRadioEspectralJacobi(aNoDom)
check(
  'estimaRadioEspectralJacobi da ρ ≥ 1 (≈ √1.5 ≈ 1.2247) para [[1,2],[3,4]]',
  rhoNoDom >= 1 && Math.abs(rhoNoDom - Math.sqrt(1.5)) < 1e-6,
  `rho=${rhoNoDom}`,
)

check(
  'omegaOptimo devuelve null cuando ρ ≥ 1 ([[1,2],[3,4]])',
  omegaOptimo(aNoDom) === null,
)

// ---------------------------------------------------------------------------
// 5. esDiagonalDominante y diagnósticos
// ---------------------------------------------------------------------------
check('esDiagonalDominante devuelve true para el preset [[4,1,1],[1,5,1],[1,1,6]]', esDiagonalDominante(aPreset))
check('esDiagonalDominante devuelve false para [[1,2],[3,4]]', !esDiagonalDominante(aNoDom))
check(
  'esDiagonalDominante exige dominancia estricta (empate |a_ii| == suma -> false)',
  !esDiagonalDominante([
    [2, 2],
    [1, 3],
  ]),
)
check(
  'esDiagonalDominante devuelve false para matriz no cuadrada',
  !esDiagonalDominante([
    [5, 1, 1],
    [1, 5, 1],
  ]),
)

const rhoPreset = estimaRadioEspectralJacobi(aPreset)
check(
  'estimaRadioEspectralJacobi da 0 < ρ < 1 para el preset diagonalmente dominante',
  rhoPreset > 0 && rhoPreset < 1,
  `rho=${rhoPreset}`,
)
check(
  'estimaRadioEspectralJacobi da 0 para una matriz diagonal',
  estimaRadioEspectralJacobi([
    [3, 0],
    [0, 5],
  ]) === 0,
)

// ---------------------------------------------------------------------------
// 6. Poisson tridiagonal 4×4: SOR óptimo acelera frente a Jacobi y Gauss-Seidel
// ---------------------------------------------------------------------------
const aPoisson4 = [
  [2, -1, 0, 0],
  [-1, 2, -1, 0],
  [0, -1, 2, -1],
  [0, 0, -1, 2],
]
const bPoisson4 = [1, 0, 0, 1] // x = [1, 1, 1, 1]
const x0Poisson4 = [0, 0, 0, 0]
const wOptPoisson = omegaOptimo(aPoisson4)
const jP4 = jacobi(aPoisson4, bPoisson4, x0Poisson4, 1e-8, 300)
const gsP4 = gaussSeidel(aPoisson4, bPoisson4, x0Poisson4, 1e-8, 300)
const sorP4 = sor(aPoisson4, bPoisson4, x0Poisson4, wOptPoisson ?? 1.26, 1e-8, 300)
const cgP4 = conjugateGradient(aPoisson4, bPoisson4, x0Poisson4, 1e-8, 50)

check(
  'Poisson 4×4: ω_opt en (1, 2) y SOR óptimo converge en menos iteraciones que Gauss-Seidel y Jacobi',
  wOptPoisson !== null &&
    wOptPoisson > 1 &&
    wOptPoisson < 2 &&
    jP4.convergio &&
    gsP4.convergio &&
    sorP4.convergio &&
    sorP4.iteraciones.length < gsP4.iteraciones.length &&
    gsP4.iteraciones.length < jP4.iteraciones.length &&
    vecClose(sorP4.x, [1, 1, 1, 1], 1e-7),
)
check(
  'Poisson 4×4: Gradiente conjugado converge en ≤ 4 iteraciones a [1, 1, 1, 1]',
  !('error' in cgP4) &&
    cgP4.convergio &&
    cgP4.iteraciones.length <= 4 &&
    vecClose(cgP4.x, [1, 1, 1, 1], 1e-8),
)

// ---------------------------------------------------------------------------
// 7. Casos borde: 1×1, diagonal cero, no mutación, x0 exacto y dimensiones inválidas
// ---------------------------------------------------------------------------
const j1 = jacobi([[4]], [8], [0], 1e-10, 10)
const gs1 = gaussSeidel([[4]], [8], [0], 1e-10, 10)
const sor1 = sor([[4]], [8], [0], 1.2, 1e-10, 20)
const cg1 = conjugateGradient([[4]], [8], [0], 1e-10, 10)
check(
  'Caso 1×1 [[4]] x = [8]: los 4 métodos convergen a [2]',
  j1.convergio &&
    vecClose(j1.x, [2], 1e-9) &&
    gs1.convergio &&
    vecClose(gs1.x, [2], 1e-9) &&
    sor1.convergio &&
    vecClose(sor1.x, [2], 1e-9) &&
    !('error' in cg1) &&
    cg1.convergio &&
    vecClose(cg1.x, [2], 1e-9),
)

const rExactStart = jacobi(aPreset, bPreset, [1, 1, 1], 1e-8, 50)
check(
  'Cuando x0 ya es la solución exacta, converge de inmediato en k = 0',
  rExactStart.convergio &&
    rExactStart.iteraciones.length === 1 &&
    rExactStart.iteraciones[0].k === 0 &&
    rExactStart.iteraciones[0].error === 0,
)

const rZeroDiag = jacobi(
  [
    [0, 1],
    [1, 2],
  ],
  [1, 3],
  [0, 0],
  1e-6,
  50,
)
check(
  'Diagonal cero en Jacobi devuelve convergio: false con motivo claro',
  !rZeroDiag.convergio &&
    rZeroDiag.iteraciones.length === 0 &&
    typeof rZeroDiag.motivo === 'string' &&
    rZeroDiag.motivo.toLowerCase().includes('diagonal'),
)

// No mutación de entradas
const aImm = [
  [4, 1, 1],
  [1, 5, 1],
  [1, 1, 6],
]
const bImm = [6, 7, 8]
const x0Imm = [0, 0, 0]
freezeMatrix(aImm)
Object.freeze(bImm)
Object.freeze(x0Imm)
const rImmJ = jacobi(aImm, bImm, x0Imm, 1e-8, 100)
const rImmGs = gaussSeidel(aImm, bImm, x0Imm, 1e-8, 100)
const rImmSor = sor(aImm, bImm, x0Imm, 1.05, 1e-8, 100)
const rImmCg = conjugateGradient(aImm, bImm, x0Imm, 1e-8, 100)
check(
  'Ningún solucionador iterativo muta a, b ni x0',
  rImmJ.convergio && rImmGs.convergio && rImmSor.convergio && !('error' in rImmCg) && rImmCg.convergio,
)

// Matriz no cuadrada y dimensiones incompatibles
check(
  'Jacobi, Gauss-Seidel y SOR lanzan Error en matriz no cuadrada',
  throwsErrorMatching(
    () => jacobi([[1, 2, 3], [4, 5, 6]], [1, 2], [0, 0], 1e-6, 10),
    (m) => m.toLowerCase().includes('cuadrada'),
  ) &&
    throwsErrorMatching(
      () => gaussSeidel([[1, 2, 3], [4, 5, 6]], [1, 2], [0, 0], 1e-6, 10),
      (m) => m.toLowerCase().includes('cuadrada'),
    ) &&
    throwsErrorMatching(
      () => sor([[1, 2, 3], [4, 5, 6]], [1, 2], [0, 0], 1, 1e-6, 10),
      (m) => m.toLowerCase().includes('cuadrada'),
    ),
)

const cgRect = conjugateGradient(
  [
    [1, 2, 3],
    [4, 5, 6],
  ],
  [1, 2],
  [0, 0],
  1e-6,
  10,
)
check(
  'Gradiente conjugado devuelve { error } en matriz no cuadrada',
  'error' in cgRect && cgRect.error.toLowerCase().includes('cuadrada'),
)

check(
  'Dimensiones incompatibles en b o x0 lanzan Error en español',
  throwsErrorMatching(
    () => jacobi(aNoDom, [1], [0, 0], 1e-6, 10),
    (m) => m.toLowerCase().includes('dimensiones incompatibles'),
  ) &&
    throwsErrorMatching(
      () => jacobi(aNoDom, [1, 2], [0], 1e-6, 10),
      (m) => m.toLowerCase().includes('dimensiones incompatibles'),
    ),
)

// ---------------------------------------------------------------------------
// 8. Regresión radio espectral: contraejemplo donde (1,1,1,1)ᵀ ∈ ker(T_J) y ρ = 2
// ---------------------------------------------------------------------------
const aContra = [
  [1, -1, 1, 0],
  [-1, 1, 0, 1],
  [1, 0, 1, -1],
  [0, 1, -1, 1],
]
const bContra = [1, 0, 0, 0]
const x0Contra = [0, 0, 0, 0]

const rhoContra = estimaRadioEspectralJacobi(aContra)
check(
  'Contraejemplo ker(T_J)∋(1,1,1,1)ᵀ: estimaRadioEspectralJacobi devuelve ρ ≈ 2 (no 0)',
  rhoContra >= 2 - 1e-6 && Math.abs(rhoContra - 2) < 1e-6,
  `rho=${rhoContra}`,
)
check(
  'Contraejemplo ker(T_J)∋(1,1,1,1)ᵀ: omegaOptimo devuelve null',
  omegaOptimo(aContra) === null,
  `omega=${String(omegaOptimo(aContra))}`,
)
const rJacobiContra = jacobi(aContra, bContra, x0Contra, 1e-6, 50)
check(
  'Contraejemplo ker(T_J)∋(1,1,1,1)ᵀ: Jacobi no converge',
  !rJacobiContra.convergio &&
    typeof rJacobiContra.motivo === 'string' &&
    rJacobiContra.motivo.length > 0,
)

// Casos adicionales de radio espectral: ciclo 3 no balanceado, autovalores complejos y nilpotente
const aCiclo3 = [
  [1, -4, 0],
  [0, 1, -1],
  [-1, 0, 1],
] // T_J tiene autovalores de módulo 4^(1/3) ≈ 1.58740105
const rhoCiclo3 = estimaRadioEspectralJacobi(aCiclo3)
check(
  'estimaRadioEspectralJacobi estima con precisión ciclo 3 con pesos desiguales (ρ = 4^(1/3))',
  Math.abs(rhoCiclo3 - Math.cbrt(4)) < 1e-6 && omegaOptimo(aCiclo3) === null,
  `rho=${rhoCiclo3}`,
)

const aComplejo = [
  [1, -2],
  [2, 1],
] // T_J = [[0, 2], [-2, 0]], autovalores ±2i -> ρ = 2
const rhoComplejo = estimaRadioEspectralJacobi(aComplejo)
check(
  'estimaRadioEspectralJacobi estima autovalores complejos conjugados (±2i -> ρ = 2)',
  Math.abs(rhoComplejo - 2) < 1e-9 && omegaOptimo(aComplejo) === null,
  `rho=${rhoComplejo}`,
)

const aNilpSup = [
  [1, 10, 0],
  [0, 1, 10],
  [0, 0, 1],
]
const aNilpInf = [
  [1, 0, 0],
  [10, 1, 0],
  [0, 10, 1],
]
check(
  'estimaRadioEspectralJacobi devuelve ρ ≈ 0 en matrices triangulares (T_J nilpotente)',
  estimaRadioEspectralJacobi(aNilpSup) < 1e-10 && estimaRadioEspectralJacobi(aNilpInf) < 1e-10,
)

const aSingularRho1 = [
  [1, -1, 0],
  [-1, 2, -1],
  [0, -1, 1],
] // T_J tiene ρ = 1 exacto
const rhoSing1 = estimaRadioEspectralJacobi(aSingularRho1)
check(
  'Matriz singular con ρ = 1 exacto: estimaRadioEspectralJacobi ≈ 1 y omegaOptimo === null',
  Math.abs(rhoSing1 - 1) < 1e-9 && omegaOptimo(aSingularRho1) === null,
  `rho=${rhoSing1}, omega=${String(omegaOptimo(aSingularRho1))}`,
)

// ---------------------------------------------------------------------------
// 9. Pruebas de frontera: maxIter = 0, x0 no nulo/no exacto en SPD, tol ≤ 0 y dimensiones CG
// ---------------------------------------------------------------------------
const msgMaxIter = (m: string): boolean => m.includes('entero mayor o igual a 1')

check(
  'Jacobi rechaza maxIter = 0 con Error en español',
  throwsErrorMatching(() => jacobi(aPreset, bPreset, x0Preset, 1e-6, 0), msgMaxIter),
)
check(
  'Gauss-Seidel rechaza maxIter = 0 con Error en español',
  throwsErrorMatching(() => gaussSeidel(aPreset, bPreset, x0Preset, 1e-6, 0), msgMaxIter),
)
check(
  'SOR rechaza maxIter = 0 con Error en español',
  throwsErrorMatching(() => sor(aPreset, bPreset, x0Preset, 1.1, 1e-6, 0), msgMaxIter),
)
check(
  'Gradiente conjugado rechaza maxIter = 0 con Error en español (matriz SPD)',
  throwsErrorMatching(() => conjugateGradient(aSpd3, bSpd3, [0, 0, 0], 1e-6, 0), msgMaxIter),
)

// Caso SPD con x0 = [2, -1, 0] (no solución): se conserva como inicio y converge a la misma solución
const x0NoNulo = [2, -1, 0]
Object.freeze(x0NoNulo)

const rJFromX0 = jacobi(aPreset, bPreset, x0NoNulo, 1e-10, 200)
const rGsFromX0 = gaussSeidel(aPreset, bPreset, x0NoNulo, 1e-10, 200)
const rSorFromX0 = sor(aPreset, bPreset, x0NoNulo, wOptPreset ?? 1.05, 1e-10, 200)
const rCgPresetFromX0 = conjugateGradient(aPreset, bPreset, x0NoNulo, 1e-10, 50)
const rCgSpdFromX0 = conjugateGradient(aSpd3, bSpd3, x0NoNulo, 1e-10, 50)

// En Jacobi con aPreset y x0 = [2, -1, 0]: x^(1) = [(6-(-1)-0)/4, (7-2-0)/5, (8-2-(-1))/6] = [1.75, 1, 7/6]
const expectedJStep1 = [1.75, 1, 7 / 6]
check(
  'SPD con x0 = [2, -1, 0]: Jacobi parte realmente de x0 y converge a [1, 1, 1]',
  rJFromX0.convergio &&
    rJFromX0.iteraciones.length > 1 &&
    rJFromX0.iteraciones[0].k === 1 &&
    vecClose(rJFromX0.iteraciones[0].x, expectedJStep1, 1e-12) &&
    vecClose(rJFromX0.x, exactPreset, 1e-8),
)
check(
  'SPD con x0 = [2, -1, 0]: Gauss-Seidel y SOR parten de x0 y convergen a [1, 1, 1]',
  rGsFromX0.convergio &&
    rGsFromX0.iteraciones.length > 1 &&
    rGsFromX0.iteraciones[0].k === 1 &&
    !vecClose(rGsFromX0.iteraciones[0].x, rGs.iteraciones[0].x, 1e-3) &&
    vecClose(rGsFromX0.x, exactPreset, 1e-8) &&
    rSorFromX0.convergio &&
    rSorFromX0.iteraciones.length > 1 &&
    rSorFromX0.iteraciones[0].k === 1 &&
    !vecClose(rSorFromX0.iteraciones[0].x, rSorOpt.iteraciones[0].x, 1e-3) &&
    vecClose(rSorFromX0.x, exactPreset, 1e-8),
)
check(
  'SPD con x0 = [2, -1, 0]: Gradiente conjugado parte de x0 y converge a la misma solución exacta',
  !('error' in rCgPresetFromX0) &&
    rCgPresetFromX0.convergio &&
    rCgPresetFromX0.iteraciones.length > 0 &&
    rCgPresetFromX0.iteraciones[0].k === 1 &&
    !vecClose(rCgPresetFromX0.iteraciones[0].x, (!('error' in rCgPreset) ? rCgPreset.iteraciones[0].x : []), 1e-3) &&
    vecClose(rCgPresetFromX0.x, exactPreset, 1e-8) &&
    !('error' in rCgSpdFromX0) &&
    rCgSpdFromX0.convergio &&
    rCgSpdFromX0.iteraciones.length > 0 &&
    rCgSpdFromX0.iteraciones[0].k === 1 &&
    vecClose(rCgSpdFromX0.x, [1, 2, -1], 1e-8),
)

// Validaciones de tol = 0 y tolerancia negativa en los cuatro solucionadores
const msgTolPos = (m: string): boolean => m.toLowerCase().includes('tolerancia') && m.toLowerCase().includes('positivo')

check(
  'tol = 0 lanza Error en español en Jacobi, Gauss-Seidel, SOR y CG (SPD)',
  throwsErrorMatching(() => jacobi(aPreset, bPreset, x0Preset, 0, 10), msgTolPos) &&
    throwsErrorMatching(() => gaussSeidel(aPreset, bPreset, x0Preset, 0, 10), msgTolPos) &&
    throwsErrorMatching(() => sor(aPreset, bPreset, x0Preset, 1.1, 0, 10), msgTolPos) &&
    throwsErrorMatching(() => conjugateGradient(aSpd3, bSpd3, [0, 0, 0], 0, 10), msgTolPos),
)
check(
  'Tolerancia negativa lanza Error en español en Jacobi, Gauss-Seidel, SOR y CG (SPD)',
  throwsErrorMatching(() => jacobi(aPreset, bPreset, x0Preset, -1e-6, 10), msgTolPos) &&
    throwsErrorMatching(() => gaussSeidel(aPreset, bPreset, x0Preset, -1e-6, 10), msgTolPos) &&
    throwsErrorMatching(() => sor(aPreset, bPreset, x0Preset, 1.1, -1e-6, 10), msgTolPos) &&
    throwsErrorMatching(() => conjugateGradient(aSpd3, bSpd3, [0, 0, 0], -1e-6, 10), msgTolPos),
)

// Validación de dimensiones en CG con matriz SPD
check(
  'Gradiente conjugado con matriz SPD lanza Error en español por dimensiones incompatibles en b o x0',
  throwsErrorMatching(
    () => conjugateGradient(aSpd3, [1, 2], [0, 0, 0], 1e-6, 10),
    (m) => m.toLowerCase().includes('dimensiones incompatibles'),
  ) &&
    throwsErrorMatching(
      () => conjugateGradient(aSpd3, bSpd3, [0, 0], 1e-6, 10),
      (m) => m.toLowerCase().includes('dimensiones incompatibles'),
    ),
)

if (failed > 0) process.exit(1)
console.log(`\niterative: ${passed} comprobaciones pasaron.`)
