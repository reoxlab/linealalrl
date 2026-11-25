// Pruebas de codificación y decodificación de estado en la URL (src/share.ts).
import { type Matrix } from '../src/matrix.ts'
import { Rat } from '../src/rational.ts'
import {
  MAX_RAT_DIGITS,
  SHARE_SCHEMA_VERSION,
  buildShareUrl,
  decodeState,
  encodeState,
} from '../src/share.ts'

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

// 1. Round-trip exacto: matriz 3×3 con fracciones
const matFracs: Matrix = [
  [new Rat(1, 3), new Rat(-5, 7), Rat.ZERO],
  [new Rat(11, 4), new Rat(2, 1), new Rat(-9, 13)],
  [new Rat(-1, 2), new Rat(100, 3), new Rat(7, 1)],
]

const encodedA = encodeState(matFracs)
const decodedA = decodeState(encodedA)
check(
  'round-trip exacto de matriz 3×3 con fracciones',
  decodedA !== null &&
    matEq(decodedA.a, matFracs) &&
    decodedA.b === null &&
    decodedA.vector === null,
)

// 2. Round-trip con prefijo #m= y buildShareUrl
const hashUrl = buildShareUrl(matFracs)
const decodedHash = decodeState(hashUrl)
check(
  'buildShareUrl produce prefijo #m= y decodeState lo recupera exacto',
  hashUrl.startsWith('#m=') && decodedHash !== null && matEq(decodedHash.a, matFracs),
)

// 3. Round-trip con vector b incluido
const vecB: Rat[] = [new Rat(-4, 1), new Rat(3, 5), new Rat(-11, 2)]
const encodedWithVec = encodeState(matFracs, null, vecB)
const decodedWithVec = decodeState(`#m=${encodedWithVec}`)
check(
  'round-trip con vector b incluido conserva A y vector exactos',
  decodedWithVec !== null &&
    matEq(decodedWithVec.a, matFracs) &&
    decodedWithVec.b === null &&
    decodedWithVec.vector !== null &&
    vecEq(decodedWithVec.vector, vecB),
)

// 4. Round-trip con A, B y vector b incluidos
const matB: Matrix = [
  [new Rat(1, 2), new Rat(3, 1)],
  [new Rat(-2, 5), Rat.ZERO],
  [Rat.ONE, new Rat(7, 8)],
]
const encodedAll = encodeState(matFracs, matB, vecB)
const decodedAll = decodeState(encodedAll)
check(
  'round-trip con A, B y vector b conserva los tres exactos',
  decodedAll !== null &&
    matEq(decodedAll.a, matFracs) &&
    decodedAll.b !== null &&
    matEq(decodedAll.b, matB) &&
    decodedAll.vector !== null &&
    vecEq(decodedAll.vector, vecB),
)

// 5. Estados vacíos o manipulados -> null sin lanzar excepciones
check('decodeState("") devuelve null', decodeState('') === null)
check('decodeState("   ") devuelve null', decodeState('   ') === null)
check('decodeState("#m=") devuelve null', decodeState('#m=') === null)
check('decodeState con caracteres inválidos devuelve null', decodeState('#m=***no-valido***') === null)
check(
  'decodeState con base64 truncado/corrupto devuelve null',
  decodeState(encodedA.slice(0, Math.floor(encodedA.length / 2))) === null,
)

function rawEncode(obj: unknown): string {
  const bytes = new TextEncoder().encode(JSON.stringify(obj))
  let bin = ''
  for (const b of bytes) bin += String.fromCharCode(b)
  return btoa(bin).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')
}

function rawDecode(encoded: string): Record<string, unknown> {
  const b64 = encoded.replace(/-/g, '+').replace(/_/g, '/')
  const pad = b64.length % 4 === 0 ? '' : '='.repeat(4 - (b64.length % 4))
  const bin = atob(b64 + pad)
  const bytes = new Uint8Array(bin.length)
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i)
  return JSON.parse(new TextDecoder().decode(bytes)) as Record<string, unknown>
}

// Bucle de truncamiento sobre TODOS los prefijos del estado completo (A, B, v)
let allPrefixesNull = true
for (let len = 1; len < encodedAll.length; len++) {
  if (decodeState(encodedAll.slice(0, len)) !== null) {
    allPrefixesNull = false
    break
  }
}
check('todos los prefijos truncados de un estado con A, B y v devuelven null', allPrefixesNull)

// encodeState(a, [], []) omite b y v e incluye versión de esquema s: 1
const encodedEmptyExtras = encodeState(matFracs, [], [])
const payloadEmptyExtras = rawDecode(encodedEmptyExtras)
const decodedEmptyExtras = decodeState(encodedEmptyExtras)
check(
  'encodeState(a, [], []) incluye s: 1 y omite b y v',
  payloadEmptyExtras.s === SHARE_SCHEMA_VERSION &&
    !('b' in payloadEmptyExtras) &&
    !('v' in payloadEmptyExtras) &&
    decodedEmptyExtras !== null &&
    decodedEmptyExtras.b === null &&
    decodedEmptyExtras.vector === null,
)

// M-3: versión de esquema y compatibilidad con versiones anteriores / futuras
check(
  'decodeState acepta estado sin campo s (versión anterior sin esquema)',
  decodeState(rawEncode({ a: [['1', '2'], ['3', '4']] })) !== null,
)
check(
  'decodeState acepta estado con s: 1 e ignora claves desconocidas de otra versión',
  decodeState(rawEncode({ s: 1, a: [['1', '2']], legacyFlag: true })) !== null,
)
check(
  'decodeState rechaza versión de esquema futura (s > SHARE_SCHEMA_VERSION)',
  decodeState(rawEncode({ s: SHARE_SCHEMA_VERSION + 1, a: [['1']] })) === null,
)
check(
  'decodeState rechaza versión de esquema inválida (0, negativa, decimal o cadena)',
  decodeState(rawEncode({ s: 0, a: [['1']] })) === null &&
    decodeState(rawEncode({ s: -1, a: [['1']] })) === null &&
    decodeState(rawEncode({ s: 1.5, a: [['1']] })) === null &&
    decodeState(rawEncode({ s: '1', a: [['1']] })) === null,
)

// Alias `vector`
const decodedAliasVec = decodeState(
  rawEncode({ a: [['1', '2'], ['3', '4']], vector: ['5/2', '-1'] }),
)
check(
  'decodeState acepta el alias "vector" para el vector b',
  decodedAliasVec !== null &&
    decodedAliasVec.vector !== null &&
    vecEq(decodedAliasVec.vector, [new Rat(5, 2), new Rat(-1, 1)]),
)

// M-4: longitud de vector distinta de las filas de A
check(
  'vector más corto que las filas de A devuelve null (M-4)',
  decodeState(rawEncode({ a: [['1', '2'], ['3', '4']], v: ['1'] })) === null,
)
check(
  'vector más largo que las filas de A devuelve null (M-4)',
  decodeState(rawEncode({ a: [['1', '2'], ['3', '4']], v: ['1', '2', '3'] })) === null,
)

// Celdas numéricas en vez de cadenas
check(
  'celdas numéricas en A devuelven null',
  decodeState(rawEncode({ a: [[1, 2], [3, 4]] })) === null,
)
check(
  'entradas numéricas en v devuelven null',
  decodeState(rawEncode({ a: [['1']], v: [2] })) === null,
)

// M-6: límite de dígitos tras parseRat (evita 1e9999)
const maxDigitsOk = '9'.repeat(MAX_RAT_DIGITS)
const maxDigitsTooBig = '9'.repeat(MAX_RAT_DIGITS + 1)
check(
  'rechaza valores gigantes como 1e9999 y 1e-9999 en A o v (M-6)',
  decodeState(rawEncode({ a: [['1e9999']] })) === null &&
    decodeState(rawEncode({ a: [['1e-9999']] })) === null &&
    decodeState(rawEncode({ a: [['1']], v: ['1e9999'] })) === null &&
    decodeState(rawEncode({ a: [[maxDigitsTooBig]] })) === null,
)
check(
  'acepta enteros en el límite exacto de 60 dígitos',
  decodeState(rawEncode({ a: [[maxDigitsOk]] })) !== null,
)

check('JSON sin matriz A devuelve null', decodeState(rawEncode({})) === null)
check('matriz A vacía devuelve null', decodeState(rawEncode({ a: [] })) === null)
check(
  'matriz A no rectangular devuelve null',
  decodeState(rawEncode({ a: [['1', '2'], ['3']] })) === null,
)
check(
  'matriz A con división por cero (1/0) devuelve null',
  decodeState(rawEncode({ a: [['1/0']] })) === null,
)
check(
  'matriz A con celda no numérica devuelve null',
  decodeState(rawEncode({ a: [['hola']] })) === null,
)
check(
  'matriz A mayor que 6×6 devuelve null',
  decodeState(rawEncode({ a: Array.from({ length: 7 }, () => ['1']) })) === null,
)
check(
  'vector b manipulado con valor inválido devuelve null',
  decodeState(rawEncode({ a: [['1']], v: ['abc'] })) === null,
)
check(
  'matriz B manipulada no rectangular devuelve null',
  decodeState(rawEncode({ a: [['1']], b: [['1', '2'], ['3']] })) === null,
)

if (failed > 0) process.exit(1)
console.log(`\nshare: ${passed} comprobaciones pasaron.`)
