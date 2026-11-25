// Pruebas de validez matemática de los presets del módulo 0 (src/presets.ts) y helpers puros de UI.
import { type Matrix, det, identity, inverse, multiply, rank, solveRref } from '../src/matrix.ts'
import { PRESETS, presetToMatrices } from '../src/presets.ts'
import { Rat } from '../src/rational.ts'
import {
  buildProductExampleFormula,
  filterTopLevelDetSteps,
  formatFreeVariables,
  formatStepsLatex,
  getInverseEmptyStepsNote,
} from '../src/ui.ts'

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

check('hay al menos 4 presets definidos', PRESETS.length >= 4)

// Todos los presets deben parsear sin errores y respetar tamaño máximo 6×6
for (const preset of PRESETS) {
  let valid = false
  try {
    const { a, vector, b } = presetToMatrices(preset)
    valid =
      a.length >= 1 &&
      a.length <= 6 &&
      a[0].length >= 1 &&
      a[0].length <= 6 &&
      (vector === null || vector.length === a.length) &&
      (b === null || (b.length >= 1 && b.length <= 6))
  } catch {
    valid = false
  }
  check(`preset «${preset.id}» es válido`, valid)
}

// Identidad 3×3
const pId = PRESETS.find((p) => p.id === 'identidad-3x3')
check('existe preset identidad-3x3', pId !== undefined)
if (pId) {
  const { a } = presetToMatrices(pId)
  const id3 = identity(3)
  const eqId = a.every((row, i) => row.every((val, j) => val.eq(id3[i][j])))
  check('identidad-3x3 coincide con I₃ y tiene det = 1', eqId && det(a).valor.isOne())
}

// Invertible 3×3 (det = -15)
const pInv = PRESETS.find((p) => p.id === 'invertible-3x3')
check('existe preset invertible-3x3', pInv !== undefined)
if (pInv) {
  const { a } = presetToMatrices(pInv)
  const d = det(a).valor
  const inv = inverse(a)
  check('invertible-3x3 tiene determinante exacto igual a -15', d.eq(new Rat(-15, 1)))
  check(
    'invertible-3x3 tiene inversa exacta y A · A⁻¹ = I',
    inv.inversa !== null &&
      multiply(a, inv.inversa).every((row, i) =>
        row.every((v, j) => v.eq(identity(3)[i][j])),
      ),
  )
}

// Singular 3×3 (det = 0)
const pSing = PRESETS.find((p) => p.id === 'singular-3x3')
check('existe preset singular-3x3', pSing !== undefined)
if (pSing) {
  const { a } = presetToMatrices(pSing)
  const d = det(a).valor
  const inv = inverse(a)
  check('singular-3x3 tiene determinante exactamente 0', d.isZero())
  check(
    'singular-3x3 no tiene inversa y reporta atasco en fila 3',
    inv.inversa === null && inv.atasco.includes('fila 3') && rank(a) < 3,
  )
}

// Sistema con solución única
const pSys = PRESETS.find((p) => p.id === 'sistema-unico')
check('existe preset sistema-unico', pSys !== undefined)
if (pSys) {
  const { a, vector } = presetToMatrices(pSys)
  check('sistema-unico incluye vector b', vector !== null)
  if (vector) {
    const sol = solveRref(a, vector)
    check(
      'sistema-unico tiene solución única',
      sol.tipo === 'unica' && sol.solucion !== undefined,
    )
    if (sol.tipo === 'unica' && sol.solucion) {
      const expected = [new Rat(-8, 1), new Rat(1, 1), new Rat(-2, 1)]
      check(
        'sistema-unico tiene solución exacta (-8, 1, -2)',
        sol.solucion.length === 3 && sol.solucion.every((v, i) => v.eq(expected[i])),
      )
      const colX = sol.solucion.map((v) => [v])
      const prod = multiply(a, colX)
      const matchesB = prod.every((row, i) => row[0].eq(vector[i]))
      check('la solución del sistema-unico verifica A · x = b exactamente', matchesB)
    }
  }
}

// Helpers puros de la UI (C-3, I-2, I-3, M-2)
const matProdA: Matrix = [
  [new Rat(1, 1), new Rat(2, 1), new Rat(3, 1)],
  [new Rat(4, 1), new Rat(5, 1), new Rat(6, 1)],
]
const matProdB: Matrix = [
  [new Rat(7, 1), new Rat(8, 1)],
  [new Rat(9, 1), new Rat(10, 1)],
  [new Rat(11, 1), new Rat(12, 1)],
]
const matProdRes = multiply(matProdA, matProdB)
const prodFormula = buildProductExampleFormula(matProdA, matProdB, matProdRes)
check(
  'buildProductExampleFormula genera la fórmula exacta para c11',
  prodFormula.tex ===
    'c_{11} = a_{11} b_{11} + a_{12} b_{21} + a_{13} b_{31} = (1)(7) + (2)(9) + (3)(11) = 58' &&
    prodFormula.plain === 'c11 = (1)(7) + (2)(9) + (3)(11) = 58',
)

// I-2: mensaje de 0 pasos de inversa para identidad vs matriz singular [[1,0],[0,0]]
const invId2 = inverse(identity(2))
const noteId = getInverseEmptyStepsNote(invId2.inversa !== null)
check(
  'I-2: mensaje de 0 pasos para la identidad menciona que ya es la identidad',
  invId2.inversa !== null && invId2.pasos.length === 0 && noteId.includes('identidad'),
)

const singZeroSteps: Matrix = [
  [Rat.ONE, Rat.ZERO],
  [Rat.ZERO, Rat.ZERO],
]
const invSing0 = inverse(singZeroSteps)
const noteSing = getInverseEmptyStepsNote(invSing0.inversa !== null)
check(
  'I-2: mensaje de 0 pasos para [[1,0],[0,0]] indica que es singular y NO identidad',
  invSing0.inversa === null &&
    invSing0.pasos.length === 0 &&
    noteSing.includes('singular') &&
    !noteSing.includes('identidad'),
)

// Etiquetas de variables libres (C-3)
const sysInf1 = solveRref(
  [
    [new Rat(1, 1), new Rat(2, 1), new Rat(3, 1)],
    [new Rat(2, 1), new Rat(4, 1), new Rat(6, 1)],
  ],
  [Rat.ZERO, Rat.ZERO],
)
const free1 = formatFreeVariables(sysInf1.libres ?? [])
check(
  'formatFreeVariables formatea x_2, x_3 y x2, x3 para [[1,2,3],[2,4,6]]',
  sysInf1.tipo === 'infinitas' && free1.tex === 'x_2, x_3' && free1.plain === 'x2, x3',
)

const sysInf2 = solveRref(
  [
    [Rat.ZERO, Rat.ONE, new Rat(2, 1)],
    [Rat.ZERO, Rat.ZERO, Rat.ONE],
  ],
  [Rat.ZERO, Rat.ZERO],
)
const free2 = formatFreeVariables(sysInf2.libres ?? [])
check(
  'formatFreeVariables formatea x_1 y x1 para [[0,1,2],[0,0,1]]',
  sysInf2.tipo === 'infinitas' && free2.tex === 'x_1' && free2.plain === 'x1',
)

// I-3 & M-2: filtrado de pasos de primer nivel para det 6×6 y bloques \[ ... \] en LaTeX
const dense6: Matrix = Array.from({ length: 6 }, (_, i) =>
  Array.from({ length: 6 }, (_, j) => new Rat((i + 1) * (j + 2) + (i === j ? 1 : 0), 1)),
)
const det6 = det(dense6)
const topDet6 = filterTopLevelDetSteps(det6.pasos, 6)
const latexDet6 = formatStepsLatex(topDet6)
check(
  'I-3: filterTopLevelDetSteps reduce los 673 pasos de una 6×6 densa a 7 pasos de primer nivel',
  det6.pasos.length === 673 && topDet6.length === 7,
)
check(
  'M-2: formatStepsLatex envuelve cada matriz en \\[ ... \\] con comentarios % fuera',
  latexDet6.includes('% Paso 1:') &&
    latexDet6.includes('\\[ \\begin{bmatrix}') &&
    latexDet6.includes('\\end{bmatrix} \\]'),
)

if (failed > 0) process.exit(1)
console.log(`\npresets: ${passed} comprobaciones pasaron.`)
