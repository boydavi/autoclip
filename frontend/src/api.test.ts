import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  ApiError,
  api,
  formatBytes,
  formatDuration,
  formatTimecode,
  waitForExport,
  waitForSource,
  type ExportRecord,
  type Source,
} from './api'

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

function source(overrides: Partial<Source> = {}): Source {
  return {
    id: 's1',
    type: 'youtube',
    title: 'A talk',
    url: null,
    filename: null,
    channel: null,
    duration_s: 600,
    width: 1920,
    height: 1080,
    fps: 30,
    has_audio: true,
    has_video: true,
    created_at: '2026-01-01T00:00:00Z',
    ingest_status: 'ready',
    ...overrides,
  }
}

function exportRecord(overrides: Partial<ExportRecord> = {}): ExportRecord {
  return {
    id: 'e1',
    clip_id: 'c1',
    ratio: '9:16',
    style: 'bold_pop',
    size_bytes: 0,
    created_at: '2026-01-01T00:00:00Z',
    download_url: '/api/exports/e1/download',
    status: 'pending',
    error: null,
    ...overrides,
  }
}

const fetchMock = vi.fn<typeof fetch>()

beforeEach(() => {
  vi.stubGlobal('fetch', fetchMock)
})

afterEach(() => {
  vi.useRealTimers()
  vi.unstubAllGlobals()
  fetchMock.mockReset()
})

describe('formatters', () => {
  it('formats durations as m:ss and h:mm:ss', () => {
    expect(formatDuration(0)).toBe('0:00')
    expect(formatDuration(65)).toBe('1:05')
    expect(formatDuration(3725)).toBe('1:02:05')
    expect(formatDuration(-4)).toBe('0:00')
  })

  it('formats timecodes with centiseconds', () => {
    expect(formatTimecode(0)).toBe('00:00.00')
    expect(formatTimecode(65.5)).toBe('01:05.50')
  })

  it('formats byte sizes', () => {
    expect(formatBytes(512)).toBe('512 B')
    expect(formatBytes(1536)).toBe('1.5 KB')
    expect(formatBytes(5 * 1024 * 1024)).toBe('5.0 MB')
    expect(formatBytes(20 * 1024 ** 3)).toBe('20 GB')
  })
})

describe('request()', () => {
  it('turns a string detail into an ApiError', async () => {
    fetchMock.mockResolvedValue(json({ detail: 'Source not found.' }, 404))

    const error = await api.getSource('nope').catch((e: unknown) => e)

    expect(error).toBeInstanceOf(ApiError)
    expect(error).toMatchObject({ message: 'Source not found.', status: 404, hint: '' })
  })

  it('carries the hint from a structured detail', async () => {
    fetchMock.mockResolvedValue(
      json({ detail: { message: 'Not a supported format.', hint: 'Accepted: .mp4' } }, 415),
    )

    const error = await api.getSource('x').catch((e: unknown) => e)

    expect(error).toMatchObject({ message: 'Not a supported format.', hint: 'Accepted: .mp4' })
  })

  it('falls back to the status line when the body is not JSON', async () => {
    fetchMock.mockResolvedValue(new Response('oops', { status: 502, statusText: 'Bad Gateway' }))

    const error = await api.getSource('x').catch((e: unknown) => e)

    expect(error).toMatchObject({ message: '502 Bad Gateway', status: 502 })
  })

  it('returns undefined for a 204 response', async () => {
    fetchMock.mockResolvedValue(new Response(null, { status: 204 }))

    await expect(api.deleteJob('j1')).resolves.toBeUndefined()
    expect(fetchMock).toHaveBeenCalledWith('/api/jobs/j1', expect.objectContaining({ method: 'DELETE' }))
  })

  it('sends JSON bodies with a JSON content type', async () => {
    fetchMock.mockResolvedValue(json({ id: 'j1' }))

    await api.createJob('s1', { max_clips: 3 })

    const init = fetchMock.mock.calls[0][1] as RequestInit
    expect(init.method).toBe('POST')
    expect(init.body).toBe(JSON.stringify({ source_id: 's1', settings: { max_clips: 3 } }))
    expect((init.headers as Record<string, string>)['Content-Type']).toBe('application/json')
  })

  it('leaves the content type to the browser for file uploads', async () => {
    fetchMock.mockResolvedValue(json({ filename: 'a.mp3', size_bytes: 1 }, 201))

    await api.uploadMusicTrack(new File(['x'], 'a.mp3'))

    const init = fetchMock.mock.calls[0][1] as RequestInit
    expect(init.body).toBeInstanceOf(FormData)
    expect(init.headers as Record<string, string>).not.toHaveProperty('Content-Type')
  })

  it('percent-encodes track names in delete URLs', async () => {
    fetchMock.mockResolvedValue(new Response(null, { status: 204 }))

    await api.deleteMusicTrack('my song #1.mp3')

    expect(fetchMock.mock.calls[0][0]).toBe('/api/music/my%20song%20%231.mp3')
  })
})

describe('waitForSource()', () => {
  it('returns a ready source without polling', async () => {
    const ready = source()

    await expect(waitForSource(ready)).resolves.toBe(ready)
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('polls a pending download until it is ready', async () => {
    vi.useFakeTimers()
    fetchMock
      .mockResolvedValueOnce(json(source({ ingest_status: 'pending' })))
      .mockResolvedValueOnce(json(source({ ingest_status: 'ready', title: 'Done' })))

    const result = waitForSource({ id: 's1', ingest_status: 'pending' })
    await vi.advanceTimersByTimeAsync(1500)
    await vi.advanceTimersByTimeAsync(1500)

    await expect(result).resolves.toMatchObject({ title: 'Done', ingest_status: 'ready' })
    expect(fetchMock).toHaveBeenCalledTimes(2)
    expect(fetchMock.mock.calls[0][0]).toBe('/api/sources/s1')
  })

  it('keeps polling a full source record that is still pending', async () => {
    vi.useFakeTimers()
    fetchMock.mockResolvedValue(json(source({ ingest_status: 'ready' })))

    const result = waitForSource(source({ ingest_status: 'pending' }))
    await vi.advanceTimersByTimeAsync(1500)

    await expect(result).resolves.toMatchObject({ ingest_status: 'ready' })
  })

  it('rejects with the download error when ingest failed', async () => {
    vi.useFakeTimers()
    fetchMock.mockResolvedValue(
      json(source({ ingest_status: 'failed', ingest_error: 'Video unavailable' })),
    )

    const result = waitForSource({ id: 's1', ingest_status: 'pending' })
    const assertion = expect(result).rejects.toMatchObject({ message: 'Video unavailable' })
    await vi.advanceTimersByTimeAsync(1500)

    await assertion
  })

  it('stops as soon as it is cancelled', async () => {
    await expect(
      waitForSource({ id: 's1', ingest_status: 'pending' }, () => true),
    ).rejects.toThrow('Cancelled.')
    expect(fetchMock).not.toHaveBeenCalled()
  })
})

describe('waitForExport()', () => {
  it('returns a finished export without polling', async () => {
    const done = exportRecord({ status: 'done', size_bytes: 10 })

    await expect(waitForExport(done)).resolves.toBe(done)
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('polls until the render finishes', async () => {
    vi.useFakeTimers()
    fetchMock
      .mockResolvedValueOnce(json(exportRecord({ status: 'pending' })))
      .mockResolvedValueOnce(json(exportRecord({ status: 'done', size_bytes: 99 })))

    const result = waitForExport(exportRecord())
    await vi.advanceTimersByTimeAsync(1000)
    await vi.advanceTimersByTimeAsync(1000)

    await expect(result).resolves.toMatchObject({ status: 'done', size_bytes: 99 })
    expect(fetchMock.mock.calls[0][0]).toBe('/api/exports/e1')
  })

  it('rejects with the render error when the export failed', async () => {
    vi.useFakeTimers()
    fetchMock.mockResolvedValue(json(exportRecord({ status: 'failed', error: 'ffmpeg exploded' })))

    const result = waitForExport(exportRecord())
    const assertion = expect(result).rejects.toMatchObject({ message: 'ffmpeg exploded' })
    await vi.advanceTimersByTimeAsync(1000)

    await assertion
  })

  it('uses a generic message when a failed export has no error text', async () => {
    await expect(waitForExport(exportRecord({ status: 'failed', error: null }))).rejects.toMatchObject({
      message: 'Export failed.',
    })
  })

  it('stops when cancelled', async () => {
    await expect(waitForExport(exportRecord(), () => true)).rejects.toThrow('Cancelled.')
  })
})
