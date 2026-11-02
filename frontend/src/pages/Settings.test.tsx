import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { Settings as SettingsData } from '../api'
import { json, mockApi, type RecordedCall } from '../test/mockApi'
import { Settings } from './Settings'

function settings(overrides: Partial<SettingsData> = {}): SettingsData {
  return {
    active_provider: 'anthropic',
    providers: {
      anthropic: { model: 'claude-sonnet-5-5', base_url: null, concurrency: null },
    },
    whisper: {
      model: 'small',
      compute_type: '',
      language: '',
      diarization: false,
      provider: 'local',
      groq_model: 'whisper-large-v3-turbo',
    },
    clips: { min_duration_s: 20, max_duration_s: 90, max_clips: 10 },
    ingest: { ytdlp_format: 'best', cookies_from_browser: '', prefer_youtube_captions: false },
    export: {
      ratio: '9:16',
      caption_style: 'bold_pop',
      loudness_lufs: -14,
      prefer_hardware_encoder: true,
      crf: 18,
      preset: 'medium',
      scan_only: false,
      write_srt: false,
      music_enabled: false,
      music_volume: 0.12,
    },
    insecure_secret_storage: false,
    keys_present: {},
    ...overrides,
  }
}

/** Serve `current` and apply every PUT to it, like the real endpoint does. */
function serve(current: SettingsData): RecordedCall[] {
  return mockApi({
    'GET /api/settings': () => json(current),
    'PUT /api/settings': ({ body }) => {
      const patch = body as Partial<SettingsData>
      current = { ...current, ...patch }
      return json(current)
    },
    'GET /api/providers/status': () =>
      json([
        { name: 'anthropic', available: false, detail: 'No API key set', models: [], requires_key: true, has_key: false },
      ]),
    'GET /api/system': () =>
      json({
        ready: true,
        python_version: '3.11',
        platform: 'win',
        ffmpeg_version: '7',
        has_libass: true,
        nvenc_works: false,
        accel: 'cpu',
        gpu_name: null,
        compute_type: 'int8',
        diarization_available: false,
      }),
    'GET /api/music': () => json([]),
  })
}

const puts = (calls: RecordedCall[]) => calls.filter((c) => c.method === 'PUT')

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('transcription engine', () => {
  it('saves the engine choice and swaps in the Groq model picker', async () => {
    const calls = serve(settings())
    render(<Settings />)
    const user = userEvent.setup()

    await user.selectOptions(await screen.findByLabelText(/^engine/i), 'groq')

    await waitFor(() => expect(puts(calls)).toHaveLength(1))
    expect(puts(calls)[0].body).toMatchObject({ whisper: { provider: 'groq' } })
    expect(await screen.findByLabelText(/groq model/i)).toBeInTheDocument()
    expect(screen.queryByLabelText(/whisper model/i)).not.toBeInTheDocument()
  })

  it('warns when Groq is selected without a stored key', async () => {
    serve(
      settings({
        whisper: { ...settings().whisper, provider: 'groq' },
        keys_present: { groq: false },
      }),
    )
    render(<Settings />)

    expect(await screen.findByText(/needs a groq api key/i)).toBeInTheDocument()
  })

  it('does not warn once a Groq key is stored', async () => {
    serve(
      settings({
        whisper: { ...settings().whisper, provider: 'groq' },
        keys_present: { groq: true },
      }),
    )
    render(<Settings />)

    await screen.findByLabelText(/groq model/i)
    expect(screen.queryByText(/needs a groq api key/i)).not.toBeInTheDocument()
  })

  it('offers a Groq key field alongside the other keys', async () => {
    serve(settings())
    render(<Settings />)

    expect(await screen.findByText(/groq api key/i)).toBeInTheDocument()
  })
})

describe('input validation', () => {
  it('rejects a non-numeric "parallel requests" value without saving', async () => {
    const calls = serve(settings())
    render(<Settings />)
    const user = userEvent.setup()

    await user.type(await screen.findByLabelText(/parallel requests/i), 'abc')
    await user.tab()

    expect(await screen.findByText(/whole number of 1 or more/i)).toBeInTheDocument()
    expect(puts(calls)).toHaveLength(0)
  })

  it('saves a valid "parallel requests" value on the active provider', async () => {
    const calls = serve(settings())
    render(<Settings />)
    const user = userEvent.setup()

    await user.type(await screen.findByLabelText(/parallel requests/i), '1')
    await user.tab()

    await waitFor(() => expect(puts(calls)).toHaveLength(1))
    expect(puts(calls)[0].body).toMatchObject({
      providers: { anthropic: { model: 'claude-sonnet-5-5', concurrency: 1 } },
    })
  })

  it('clears the limit when the field is emptied', async () => {
    const calls = serve(
      settings({
        providers: {
          anthropic: { model: 'claude-sonnet-5-5', base_url: null, concurrency: 2 },
        },
      }),
    )
    render(<Settings />)
    const user = userEvent.setup()

    await user.clear(await screen.findByLabelText(/parallel requests/i))
    await user.tab()

    await waitFor(() => expect(puts(calls)).toHaveLength(1))
    expect(puts(calls)[0].body).toMatchObject({
      providers: { anthropic: { concurrency: null } },
    })
  })

  it.each(['0', '52', '18.5'])('rejects the out-of-range quality value %s', async (value) => {
    const calls = serve(settings())
    render(<Settings />)
    const user = userEvent.setup()

    const field = await screen.findByLabelText(/quality \(crf/i)
    await user.clear(field)
    await user.type(field, value)
    await user.tab()

    expect(await screen.findByText(/whole number from 1 to 51/i)).toBeInTheDocument()
    expect(puts(calls)).toHaveLength(0)
  })

  it('saves a valid quality value', async () => {
    const calls = serve(settings())
    render(<Settings />)
    const user = userEvent.setup()

    const field = await screen.findByLabelText(/quality \(crf/i)
    await user.clear(field)
    await user.type(field, '23')
    await user.tab()

    await waitFor(() => expect(puts(calls)).toHaveLength(1))
    expect(puts(calls)[0].body).toMatchObject({ export: { crf: 23, preset: 'medium' } })
  })
})

describe('export and ingest switches', () => {
  it('saves scan-only mode', async () => {
    const calls = serve(settings())
    render(<Settings />)
    const user = userEvent.setup()

    await user.click(await screen.findByLabelText(/find clips only/i))

    await waitFor(() => expect(puts(calls)).toHaveLength(1))
    expect(puts(calls)[0].body).toMatchObject({ export: { scan_only: true } })
  })

  it('saves the encoder preset', async () => {
    const calls = serve(settings())
    render(<Settings />)
    const user = userEvent.setup()

    await user.selectOptions(await screen.findByLabelText(/encoder speed/i), 'slow')

    await waitFor(() => expect(puts(calls)).toHaveLength(1))
    expect(puts(calls)[0].body).toMatchObject({ export: { preset: 'slow' } })
  })

  it('saves the YouTube captions preference', async () => {
    const calls = serve(settings())
    render(<Settings />)
    const user = userEvent.setup()

    await user.click(await screen.findByLabelText(/youtube's own captions/i))

    await waitFor(() => expect(puts(calls)).toHaveLength(1))
    expect(puts(calls)[0].body).toMatchObject({ ingest: { prefer_youtube_captions: true } })
  })
})
