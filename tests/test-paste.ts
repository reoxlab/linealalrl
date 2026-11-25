// Pruebas del parser de matrices pegadas desde el portapapeles (src/paste.ts).
import { parsePastedAugmented, parsePastedMatrix } from '../src/paste.ts'
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

// 1. Separado por tabuladores (estilo hoja de cálculo)
const rTab = parsePastedMatrix('1\t2\t3\n4\t5/2\t-6')
check(
  'parsea valores separados por tabulador',
  rTab.ok &&
    rTab.filas === 2 &&
    rTab.cols === 3 &&
    rTab.matriz[0][0].eq(new Rat(1, 1)) &&
    rTab.matriz[1][1].eq(new Rat(5, 2)) &&
    rTab.matriz[1][2].eq(new Rat(-6, 1)),
)

// 2. Separado por comas (con y sin espacios)
const rComma = parsePastedMatrix('1, 2, 3\n0.5,-3/4,2')
check(
  'parsea valores separados por comas',
  rComma.ok &&
    rComma.filas === 2 &&
    rComma.cols === 3 &&
    rComma.matriz[1][0].eq(new Rat(1, 2)) &&
    rComma.matriz[1][1].eq(new Rat(-3, 4)),
)

// 3. Separado por espacios múltiples y saltos CRLF con líneas vacías
const rSpace = parsePastedMatrix('\n  1   -2  \r\n  3/4   0.25 \r\n\n')
check(
  'parsea valores separados por espacios con CRLF y líneas en blanco',
  rSpace.ok &&
    rSpace.filas === 2 &&
    rSpace.cols === 2 &&
    rSpace.matriz[0][1].eq(new Rat(-2, 1)) &&
    rSpace.matriz[1][0].eq(new Rat(3, 4)) &&
    rSpace.matriz[1][1].eq(new Rat(1, 4)),
)

// 4. Filas de distinta longitud -> error claro
const rRagged = parsePastedMatrix('1 2 3\n4 5')
check(
  'filas de distinta longitud producen error',
  !rRagged.ok && rRagged.error.includes('fila 2'),
)

// 5. Texto vacío o solo espacios -> error
const rEmpty = parsePastedMatrix('')
const rBlank = parsePastedMatrix('   \n\t\n  ')
check('texto vacío produce error', !rEmpty.ok && rEmpty.error.length > 0)
check('texto con solo espacios produce error', !rBlank.ok && rBlank.error.length > 0)

// 6. Celdas inválidas -> error indicando fila y columna
const rBadCell = parsePastedMatrix('1 2\n3 abc')
check(
  'celda no numérica produce error con fila y columna',
  !rBadCell.ok && rBadCell.error.includes('fila 2') && rBadCell.error.includes('columna 2'),
)

const rDivZero = parsePastedMatrix('1 1/0')
check('fracción con denominador 0 produce error', !rDivZero.ok)

const rConsecutiveComma = parsePastedMatrix('1,,2\n3,4,5')
check('comas consecutivas (celda vacía) producen error', !rConsecutiveComma.ok)

// 7. Excede límite de 6×6 -> error
const rTooManyRows = parsePastedMatrix('1\n2\n3\n4\n5\n6\n7')
const rTooManyCols = parsePastedMatrix('1 2 3 4 5 6 7')
check('más de 6 filas produce error', !rTooManyRows.ok)
check('más de 6 columnas produce error', !rTooManyCols.ok)

// 8. Matriz aumentada [A | b]
const rAugPipe = parsePastedAugmented('1 2 -1 | -4\n2 3 -1 | -11\n-2 0 -3 | 22')
check(
  'parsePastedAugmented acepta separador "|" y separa A y vector b',
  rAugPipe.ok &&
    rAugPipe.filas === 3 &&
    rAugPipe.cols === 3 &&
    rAugPipe.vector[0].eq(new Rat(-4, 1)) &&
    rAugPipe.vector[1].eq(new Rat(-11, 1)) &&
    rAugPipe.vector[2].eq(new Rat(22, 1)),
)

const rAugNoPipe = parsePastedAugmented('2, 1, 5\n1, -1, 1')
check(
  'parsePastedAugmented funciona sin "|"',
  rAugNoPipe.ok &&
    rAugNoPipe.filas === 2 &&
    rAugNoPipe.cols === 2 &&
    rAugNoPipe.vector[0].eq(new Rat(5, 1)) &&
    rAugNoPipe.vector[1].eq(new Rat(1, 1)),
)

const rAugTooNarrow = parsePastedAugmented('1\n2')
check('parsePastedAugmented con solo 1 columna produce error', !rAugTooNarrow.ok)

// 9. Una sola fila y una sola celda (C-2)
const rSingleRow = parsePastedMatrix('1 2 3')
check(
  'parsea una sola fila (1×3)',
  rSingleRow.ok &&
    rSingleRow.filas === 1 &&
    rSingleRow.cols === 3 &&
    rSingleRow.matriz[0][2].eq(new Rat(3, 1)),
)

const rSingleCell = parsePastedMatrix('-7/3')
check(
  'parsea una sola celda (1×1)',
  rSingleCell.ok &&
    rSingleCell.filas === 1 &&
    rSingleCell.cols === 1 &&
    rSingleCell.matriz[0][0].eq(new Rat(-7, 3)),
)

// 10. Decimales con coma separados por tabuladores (I-1)
const rTabDec1 = parsePastedMatrix('0,5\t1\n2\t0,25')
check(
  'I-1: "0,5\\t1\\n2\\t0,25" se parsea como 2×2 [[1/2,1],[2,1/4]]',
  rTabDec1.ok &&
    rTabDec1.filas === 2 &&
    rTabDec1.cols === 2 &&
    rTabDec1.matriz[0][0].eq(new Rat(1, 2)) &&
    rTabDec1.matriz[0][1].eq(new Rat(1, 1)) &&
    rTabDec1.matriz[1][0].eq(new Rat(2, 1)) &&
    rTabDec1.matriz[1][1].eq(new Rat(1, 4)),
)

const rTabDec2 = parsePastedMatrix('0,5\t1,5\n2,5\t0,25')
check(
  'I-1: "0,5\\t1,5\\n2,5\\t0,25" se parsea como 2×2 [[1/2,3/2],[5/2,1/4]]',
  rTabDec2.ok &&
    rTabDec2.filas === 2 &&
    rTabDec2.cols === 2 &&
    rTabDec2.matriz[0][0].eq(new Rat(1, 2)) &&
    rTabDec2.matriz[0][1].eq(new Rat(3, 2)) &&
    rTabDec2.matriz[1][0].eq(new Rat(5, 2)) &&
    rTabDec2.matriz[1][1].eq(new Rat(1, 4)),
)

// 11. Separador punto y coma (;) con y sin coma decimal (I-1 / C-2)
const rSemi = parsePastedMatrix('0,5; 1,5\n-2,5; 0,25')
check(
  'separador ";" interpreta comas decimales correctamente',
  rSemi.ok &&
    rSemi.filas === 2 &&
    rSemi.cols === 2 &&
    rSemi.matriz[0][0].eq(new Rat(1, 2)) &&
    rSemi.matriz[0][1].eq(new Rat(3, 2)) &&
    rSemi.matriz[1][0].eq(new Rat(-5, 2)) &&
    rSemi.matriz[1][1].eq(new Rat(1, 4)),
)

const rMixedEmpty = parsePastedMatrix('1,;2')
check('M-7: separadores mezclados consecutivos ("1,;2") producen error', !rMixedEmpty.ok)

// 12. Fracciones con espacios alrededor de "/" (M-7)
const rFracSpaces = parsePastedMatrix('1 / 2  3\n-3 / 4  2')
check(
  'M-7: "1 / 2" con espacios se normaliza antes de separar',
  rFracSpaces.ok &&
    rFracSpaces.filas === 2 &&
    rFracSpaces.cols === 2 &&
    rFracSpaces.matriz[0][0].eq(new Rat(1, 2)) &&
    rFracSpaces.matriz[0][1].eq(new Rat(3, 1)) &&
    rFracSpaces.matriz[1][0].eq(new Rat(-3, 4)) &&
    rFracSpaces.matriz[1][1].eq(new Rat(2, 1)),
)

// 13. Validación estricta de "|" y límite en aumentada (M-8)
const rPipeBadPos = parsePastedAugmented('1 | 2 3\n4 | 5 6')
check('M-8: "|" que no está justo antes del último valor produce error', !rPipeBadPos.ok)

const rPipeMissingRow = parsePastedAugmented('1 2 | 3\n4 5 6')
check('M-8: "|" presente solo en algunas filas produce error', !rPipeMissingRow.ok)

const rPipeDouble = parsePastedAugmented('1 | 2 | 3')
check('M-8: más de un "|" en una fila produce error', !rPipeDouble.ok)

const rAugTooWide = parsePastedAugmented('1 2 3 4 5 6 7 | 8')
check(
  'M-8: aumentada con más de 6 columnas para A indica el límite específico',
  !rAugTooWide.ok && rAugTooWide.error.includes('A admite como máximo 6 columnas (más la columna b)'),
)

if (failed > 0) process.exit(1)
console.log(`\npaste: ${passed} comprobaciones pasaron.`)
