import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { CostEstimate, Job, Source } from '../api'
import { json, mockApi, noContent } from '../test/mockApi'
import { Ingest } from './Ingest'

function source(overrides: Partial<Source> = {}): Source {
  return {
    id: 's1',
    type: 'youtube',
    title: 'A long talk',
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

const estimate: CostEstimate = {
  duration_s: 600,
  windows: 2,
  whisper_provider: 'local',
  whisper_cost_usd: 0,
  llm_provider: 'anthropic',
  llm_model: 'claude-sonnet-5-5',
  llm_cost_per_window_usd: 0.0123,
  llm_cost_usd: 0.0246,
  total_usd: 0.0246,
}

function job(overrides: Partial<Job> = {}): Job {
  return {
    id: 'j9',
    source_id: 's1',
    status: 'done',
    current_stage: 'export',
    progress: 1,
    error: null,
    provider: 'anthropic',
    created_at: '2026-01-01T00:00:00Z',
    updated_at: '2026-01-01T00:00:00Z',
    started_at: null,
    finished_at: null,
    source: source(),
    ...overrides,
  }
}

const baseRoutes = {
  'GET /api/jobs?limit=8': () => json([]),
  'GET /api/providers/status': () => json([]),
  'GET /api/music': () => json([]),
}

function renderIngest() {
  return render(
    <MemoryRouter
      initialEntries={['/']}
      future={{ v7_startTransition: true, v7_relativeSplatPath: true }}
    >
      <Routes>
        <Route path="/" element={<Ingest />} />
        <Route path="/jobs/:id" element={<p>progress page</p>} />
      </Routes>
    </MemoryRouter>,
  )
}

async function paste(link: string) {
  const user = userEvent.setup()
  await user.type(screen.getByLabelText(/paste a link/i), link)
  await user.click(screen.getByRole('button', { name: 'Start' }))
  return user
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('pasting a link', () => {
  it('waits for the download, shows the cost, and only then creates the job', async () => {
    const calls = mockApi({
      ...baseRoutes,
      'POST /api/sources/youtube': () => json({ id: 's1', ingest_status: 'pending' }, 202),
      'GET /api/sources/s1': () => json(source()),
      'GET /api/estimate/s1': () => json(estimate),
      'POST /api/jobs': () => json(job({ id: 'j1', status: 'queued' }), 201),
    })
    renderIngest()

    const user = await paste('https://www.youtube.com/watch?v=abc123')

    expect(await screen.findByText('Ready to run', {}, { timeout: 5000 })).toBeInTheDocument()
    expect(screen.getByText('A long talk')).toBeInTheDocument()
    expect(screen.getByText('Estimated total')).toBeInTheDocument()
    expect(screen.getAllByText('$0.025').length).toBeGreaterThan(0)
    // The estimate must come after the download finished, and no job exists yet.
    expect(calls.map((c) => `${c.method} ${c.path}`)).toEqual([
      'GET /api/jobs?limit=8',
      'GET /api/providers/status',
      'GET /api/music',
      'POST /api/sources/youtube',
      'GET /api/sources/s1',
      'GET /api/estimate/s1',
    ])

    await user.click(screen.getByRole('button', { name: 'Run' }))

    expect(await screen.findByText('progress page')).toBeInTheDocument()
    const created = calls.find((c) => c.method === 'POST' && c.path === '/api/jobs')
    expect(created?.body).toEqual({ source_id: 's1', settings: {} })
  }, 15000)

  it('sends a non-YouTube link to the generic URL endpoint', async () => {
    const calls = mockApi({
      ...baseRoutes,
      'POST /api/sources/url': () => json(source({ id: 's2', title: 'A Vimeo talk' })),
      'GET /api/estimate/s2': () => json(estimate),
    })
    renderIngest()

    await paste('https://vimeo.com/12345')

    expect(await screen.findByText('A Vimeo talk')).toBeInTheDocument()
    expect(calls.some((c) => c.path === '/api/sources/url')).toBe(true)
    expect(calls.some((c) => c.path === '/api/sources/youtube')).toBe(false)
  })

  it('shows the download error and never offers to run', async () => {
    const calls = mockApi({
      ...baseRoutes,
      'POST /api/sources/youtube': () => json({ id: 's3', ingest_status: 'pending' }, 202),
      'GET /api/sources/s3': () =>
        json(source({ id: 's3', ingest_status: 'failed', ingest_error: 'Video unavailable' })),
    })
    renderIngest()

    await paste('https://youtu.be/gone')

    expect(await screen.findByText('Video unavailable', {}, { timeout: 5000 })).toBeInTheDocument()
    expect(screen.queryByText('Ready to run')).not.toBeInTheDocument()
    expect(calls.some((c) => c.path === '/api/jobs' && c.method === 'POST')).toBe(false)
  }, 15000)

  it('still offers to run when the estimate is unavailable', async () => {
    mockApi({
      ...baseRoutes,
      'POST /api/sources/url': () => json(source({ id: 's4' })),
      'GET /api/estimate/s4': () => json({ detail: 'boom' }, 500),
    })
    renderIngest()

    await paste('https://example.com/video.mp4')

    expect(await screen.findByText('Cost estimate unavailable.')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Run' })).toBeEnabled()
  })

  it('cancelling the confirmation creates no job', async () => {
    const calls = mockApi({
      ...baseRoutes,
      'POST /api/sources/url': () => json(source({ id: 's5' })),
      'GET /api/estimate/s5': () => json(estimate),
    })
    renderIngest()
    const user = await paste('https://example.com/a.mp4')

    await user.click(await screen.findByRole('button', { name: 'Cancel' }))

    expect(screen.queryByText('Ready to run')).not.toBeInTheDocument()
    expect(calls.some((c) => c.method === 'POST' && c.path === '/api/jobs')).toBe(false)
  })
})

describe('recent jobs', () => {
  const finished = job({ id: 'j9', status: 'done' })

  it('deletes a finished job after confirmation', async () => {
    const calls = mockApi({
      ...baseRoutes,
      'GET /api/jobs?limit=8': () => json([finished]),
      'DELETE /api/jobs/j9': () => noContent(),
    })
    vi.spyOn(window, 'confirm').mockReturnValue(true)
    renderIngest()
    const user = userEvent.setup()

    await user.click(await screen.findByRole('button', { name: /delete a long talk/i }))

    await waitFor(() => expect(screen.queryByText('A long talk')).not.toBeInTheDocument())
    expect(calls.some((c) => c.method === 'DELETE' && c.path === '/api/jobs/j9')).toBe(true)
  })

  it('keeps the job when the confirmation is declined', async () => {
    const calls = mockApi({
      ...baseRoutes,
      'GET /api/jobs?limit=8': () => json([finished]),
    })
    vi.spyOn(window, 'confirm').mockReturnValue(false)
    renderIngest()
    const user = userEvent.setup()

    await user.click(await screen.findByRole('button', { name: /delete a long talk/i }))

    expect(screen.getByText('A long talk')).toBeInTheDocument()
    expect(calls.some((c) => c.method === 'DELETE')).toBe(false)
  })

  it('offers no delete button for queued or running jobs', async () => {
    mockApi({
      ...baseRoutes,
      'GET /api/jobs?limit=8': () =>
        json([job({ id: 'a', status: 'queued' }), job({ id: 'b', status: 'running', progress: 0.4 })]),
    })
    renderIngest()

    const list = await screen.findByRole('list')
    expect(within(list).getAllByRole('listitem')).toHaveLength(2)
    expect(screen.queryByRole('button', { name: /delete/i })).not.toBeInTheDocument()
  })

  it('shows the server error when a delete is refused', async () => {
    mockApi({
      ...baseRoutes,
      'GET /api/jobs?limit=8': () => json([finished]),
      'DELETE /api/jobs/j9': () => json({ detail: 'Job is running. Cancel it before deleting.' }, 409),
    })
    vi.spyOn(window, 'confirm').mockReturnValue(true)
    renderIngest()
    const user = userEvent.setup()

    await user.click(await screen.findByRole('button', { name: /delete a long talk/i }))

    expect(await screen.findByText('Job is running. Cancel it before deleting.')).toBeInTheDocument()
    expect(screen.getByText('A long talk')).toBeInTheDocument()
  })
})
