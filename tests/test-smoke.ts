// Prueba mínima (smoke test) del scaffold de LinealAlRL.
import { readFileSync } from 'node:fs'
import { resolveTheme } from 'rl-ui'

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

const html = readFileSync('index.html', 'utf8')
const titleMatch = html.match(/<title>([^<]+)<\/title>/)
const title = titleMatch ? titleMatch[1] : ''
const appName = title.split('—')[0].trim()

check('el título de la app es LinealAlRL', appName === 'LinealAlRL', `obtenido: "${appName}"`)
check(
  'index.html incluye el script de Umami',
  html.includes('https://analytics.reoxlab.com/script.js') &&
    html.includes('0bf0567e-1e8c-4977-a79e-0314f652956a'),
)

const pkg = JSON.parse(readFileSync('package.json', 'utf8')) as {
  name?: string
  license?: string
}
check('package.json tiene nombre linealalrl', pkg.name === 'linealalrl')
check('package.json declara GPL-3.0-or-later', pkg.license === 'GPL-3.0-or-later')

check('rl-ui resuelve tema system/dark correctamente', resolveTheme('system', true) === 'dark')
check('rl-ui resuelve tema system/light correctamente', resolveTheme('system', false) === 'light')

if (failed > 0) process.exit(1)
console.log(`\nsmoke: ${passed} comprobaciones pasaron.`)
