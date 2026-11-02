import { vi } from 'vitest'

export interface RecordedCall {
  method: string
  path: string
  body: unknown
}

type Handler = (call: RecordedCall) => Response | Promise<Response>

export function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

export function noContent(): Response {
  return new Response(null, { status: 204 })
}

/**
 * Replace `fetch` with a router keyed by "METHOD /path".
 *
 * A request nobody registered fails loudly (HTTP 599) instead of silently
 * succeeding, so a test can't pass by accident on an endpoint it never meant
 * to exercise. Returns the recorded calls for assertions.
 */
export function mockApi(routes: Record<string, Handler>): RecordedCall[] {
  const calls: RecordedCall[] = []

  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const method = (init?.method ?? 'GET').toUpperCase()
      const path = typeof input === 'string' ? input : input.toString()
      let body: unknown = init?.body
      if (typeof body === 'string') {
        try {
          body = JSON.parse(body)
        } catch {
          /* leave as text */
        }
      }

      const call = { method, path, body }
      calls.push(call)

      const handler = routes[`${method} ${path}`]
      if (!handler) {
        return json({ detail: `Unmocked request: ${method} ${path}` }, 599)
      }
      return handler(call)
    }),
  )

  return calls
}
