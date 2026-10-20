// Guarda la app en el teléfono para que abra sin internet una vez instalada.
//
// Lo guardado se entrega de inmediato y la versión nueva se baja por detrás: así la app abre igual
// sin red, con mala señal o cuando la red responde pero se queda colgada (algunas redes móviles con
// github.io), en vez de esperar a la red antes de mostrar nada.
const CACHE = 'entometria-v2'

/** Lo que no se sabe si llegará: se espera a la red solo este tiempo cuando no hay nada guardado. */
const ESPERA_MS = 8000

function conLimite(promesa, ms) {
  return Promise.race([promesa, new Promise((_, rechazar) => setTimeout(() => rechazar(new Error('La red no respondió')), ms))])
}

/** Descarga la página y todo lo que ella enlaza (código, estilos, íconos), y borra lo de versiones viejas. */
async function precargar() {
  const cache = await caches.open(CACHE)
  const pagina = await fetch('./', { cache: 'no-cache' })
  if (!pagina.ok) return
  const html = await pagina.clone().text()
  const recursos = [...html.matchAll(/(?:src|href)="(\.\/[^"]+)"/g)].map((m) => m[1])
  // Primero el código de la versión nueva y después la página: así nunca queda una página sin su código.
  await cache.addAll(recursos)
  await cache.put('./', pagina)
  const vigentes = new Set(['./', ...recursos].map((r) => new URL(r, self.location.href).href))
  for (const pedido of await cache.keys()) {
    const url = new URL(pedido.url)
    // Lo bajado aparte (el PDF, los modelos de la IA) se conserva; solo se tira el código viejo.
    if (url.pathname.includes('/assets/') && !vigentes.has(pedido.url)) await cache.delete(pedido)
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
      // Se ignora «Vary»: la página pide el código con otras cabeceras que al guardarlo, y sin esto no coincide.
      caches.match(pedido, { ignoreVary: true }).then(
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
  const clave = pedido.mode === 'navigate' ? './' : pedido
  evento.respondWith(
    caches.match(clave, { ignoreVary: true, ignoreSearch: pedido.mode === 'navigate' }).then((guardado) => {
      if (guardado) {
        // Se muestra lo guardado y la versión nueva se baja por detrás para la próxima vez.
        evento.waitUntil((pedido.mode === 'navigate' ? precargar() : fetch(pedido).then((r) => guardar(clave, r))).catch(() => {}))
        return guardado
      }
      return conLimite(fetch(pedido, { cache: 'no-cache' }), ESPERA_MS).then((respuesta) => {
        guardar(clave, respuesta)
        return respuesta
      })
    }),
  )
})
