// Guarda la app en el teléfono para que abra sin internet una vez instalada.
const CACHE = 'entometria-v1'

/** Descarga la página y todo lo que ella enlaza (código, estilos, íconos), y borra lo de versiones viejas. */
async function precargar() {
  const cache = await caches.open(CACHE)
  const pagina = await fetch('./', { cache: 'no-cache' })
  const html = await pagina.clone().text()
  const recursos = [...html.matchAll(/(?:src|href)="(\.\/[^"]+)"/g)].map((m) => m[1])
  await cache.put('./', pagina)
  await cache.addAll(recursos)
  const vigentes = new Set(['./', ...recursos].map((r) => new URL(r, self.location.href).href))
  for (const pedido of await cache.keys()) {
    if (!vigentes.has(pedido.url)) await cache.delete(pedido)
  }
}

self.addEventListener('install', (evento) => {
  evento.waitUntil(precargar().then(() => self.skipWaiting()))
})

self.addEventListener('activate', (evento) => {
  // Se tira lo guardado por versiones anteriores de este archivo.
  const limpiar = caches.keys().then((claves) => Promise.all(claves.filter((c) => c !== CACHE).map((c) => caches.delete(c))))
  evento.waitUntil(limpiar.then(() => self.clients.claim()))
})

function guardar(pedido, respuesta) {
  if (!respuesta.ok) return
  const copia = respuesta.clone()
  caches.open(CACHE).then((cache) => cache.put(pedido, copia))
}

self.addEventListener('fetch', (evento) => {
  const pedido = evento.request
  const url = new URL(pedido.url)
  if (pedido.method !== 'GET' || url.origin !== self.location.origin) return
  if (url.pathname.includes('/assets/')) {
    // El código y los estilos llevan la versión en su nombre: si ya están guardados, no cambian.
    evento.respondWith(
      caches.match(pedido).then(
        (guardado) =>
          guardado ??
          fetch(pedido).then((respuesta) => {
            guardar(pedido, respuesta)
            return respuesta
          }),
      ),
    )
    return
  }
  // La página, el nombre y los íconos pueden cambiar sin cambiar de dirección:
  // primero la red, para recibir lo nuevo; sin internet, lo guardado.
  const clave = pedido.mode === 'navigate' ? './' : pedido
  evento.respondWith(
    fetch(pedido, { cache: 'no-cache' })
      .then((respuesta) => {
        guardar(clave, respuesta)
        return respuesta
      })
      .catch(() => caches.match(clave)),
  )
})
