/**
 * Typed client for the AutoClip API.
 *
 * Same-origin throughout: the server hands out the built bundle, so there is no
 * base URL to configure and no CORS to negotiate.
 */

export interface Source {
  id: string
  type: string
  title: string
  url: string | null
  filename: string | null
  channel: string | null
  duration_s: number
  width: number | null
  height: number | null
  fps: number | null
  has_audio: boolean
  has_video: boolean
  created_at: string
  ingest_status?: 'pending' | 'ready' | 'failed'
  ingest_error?: string | null
}

/** 202 body from the async URL ingest endpoints; poll getSource until ready. */
export interface IngestAccepted {
  id: string
  ingest_status: 'pending'
}

export type JobStatus = 'queued' | 'running' | 'failed' | 'done' | 'cancelled'

export interface Job {
  id: string
  source_id: string
  status: JobStatus
  current_stage: string
  progress: number
  error: string | null
  provider: string
  created_at: string
  updated_at: string
  started_at: string | null
  finished_at: string | null
  source: Source | null
}

export interface ExportRecord {
  id: string
  clip_id: string
  ratio: string
  style: string
  size_bytes: number
  created_at: string
  download_url: string
  /** The render runs in the background; only 'done' records have a file to download. */
  status: 'pending' | 'done' | 'failed'
  error: string | null
}

export type ClipStatus = 'candidate' | 'kept' | 'discarded' | 'exported'

export interface Clip {
  id: string
  job_id: string
  rank: number
  start_s: number
  end_s: number
  duration_s: number
  start_word: number
  end_word: number
  title: string
  hook: string
  score: number
  reason: string
  status: ClipStatus
  user_trimmed: boolean
  caption_style: string
  ratio: string
  exports: ExportRecord[]
}

export interface Word {
  text: string
  start: number
  end: number
  speaker: string | null
}

export interface CropKeyframe {
  t: number
  x: number
  y: number
}

export interface CropSegment {
  start_s: number
  end_s: number
  width: number
  height: number
  keyframes: CropKeyframe[]
  strategy: 'track' | 'wide' | 'general'
  zoom: number
  fit: boolean
}

/** Mirrors autoclip.pipeline.reframe.croppath.CropPath. */
export interface CropPath {
  source_width: number
  source_height: number
  segments: CropSegment[]
}

export interface CaptionStyle {
  key: string
  label: string
  description: string
  preview: {
    font: string
    primary: string
    accent: string | null
    outline: string
    outlineWidth: number
    allCaps: boolean
    sizeRatio: number
    marginRatio: number
    boxed: boolean
    animation: string
    maxWords: number
  }
}

export interface ProviderStatus {
  name: string
  available: boolean
  detail: string
  models: string[]
  requires_key: boolean
  has_key: boolean
}

export interface Settings {
  active_provider: string
  providers: Record<
    string,
    { model: string; base_url: string | null; concurrency: number | null }
  >
  whisper: {
    model: string
    compute_type: string
    language: string
    diarization: boolean
    provider: 'local' | 'groq'
    groq_model: string
  }
  clips: { min_duration_s: number; max_duration_s: number; max_clips: number }
  ingest: {
    ytdlp_format: string
    cookies_from_browser: string
    prefer_youtube_captions: boolean
  }
  export: {
    ratio: string
    caption_style: string
    loudness_lufs: number
    prefer_hardware_encoder: boolean
    crf: number
    preset: string
    scan_only: boolean
    write_srt: boolean
    music_enabled: boolean
    music_volume: number
  }
  insecure_secret_storage: boolean
  keys_present: Record<string, boolean>
}

export interface SystemStatus {
  ready: boolean
  python_version: string
  platform: string
  ffmpeg_version: string | null
  has_libass: boolean
  nvenc_works: boolean
  accel: string
  gpu_name: string | null
  compute_type: string
  diarization_available: boolean
}

export type PromptStyle =
  | 'general'
  | 'interview'
  | 'tutorial'
  | 'sales'
  | 'comedy'
  | 'keynote'
  | 'product-demo'
  | 'debate'
  | 'storytime'

export interface MusicTrack {
  filename: string
  size_bytes: number
}

export interface CostEstimate {
  duration_s: number
  windows: number
  whisper_provider: string
  whisper_cost_usd: number
  llm_provider: string
  llm_model: string
  llm_cost_per_window_usd: number
  llm_cost_usd: number
  total_usd: number
}

export interface JobSettingsOverrides {
  provider?: string
  whisper_model?: string
  language?: string
  diarization?: boolean
  min_duration_s?: number
  max_duration_s?: number
  max_clips?: number
  caption_style?: string
  ratio?: string
  prompt_style?: PromptStyle
  music_track?: string
}

/** An API error carrying the server's message and its actionable hint. */
export class ApiError extends Error {
  hint: string
  status: number

  constructor(message: string, status: number, hint = '') {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.hint = hint
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: {
      ...(init?.body instanceof FormData ? {} : { 'Content-Type': 'application/json' }),
      ...init?.headers,
    },
  })

  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`
    let hint = ''
    try {
      const body = await response.json()
      const detail = body.detail
      if (typeof detail === 'string') {
        message = detail
      } else if (detail && typeof detail === 'object') {
        // The ingest endpoints return {message, hint} so the UI can show the
        // fix alongside the failure.
        message = detail.message ?? message
        hint = detail.hint ?? ''
      }
    } catch {
      /* Response had no JSON body; the status line is the best we have. */
    }
    throw new ApiError(message, response.status, hint)
  }

  if (response.status === 204) return undefined as T
  return response.json() as Promise<T>
}

export const api = {
  health: () => request<{ status: string; version: string }>('/api/health'),
  system: () => request<SystemStatus>('/api/system'),
  fetchModels: () => request<void>('/api/system/models', { method: 'POST' }),

  listSources: () => request<Source[]>('/api/sources'),

  getSource: (id: string) => request<Source>(`/api/sources/${id}`),

  ingestYouTube: (url: string, cookiesFromBrowser?: string) =>
    request<Source | IngestAccepted>('/api/sources/youtube', {
      method: 'POST',
      body: JSON.stringify({ url, cookies_from_browser: cookiesFromBrowser || null }),
    }),

  ingestUrl: (url: string, cookiesFromBrowser?: string) =>
    request<Source | IngestAccepted>('/api/sources/url', {
      method: 'POST',
      body: JSON.stringify({ url, cookies_from_browser: cookiesFromBrowser || null }),
    }),

  uploadSource: (file: File) => {
    const form = new FormData()
    form.append('file', file)
    return request<Source>('/api/sources/upload', { method: 'POST', body: form })
  },

  listJobs: (limit = 25) => request<Job[]>(`/api/jobs?limit=${limit}`),
  getJob: (id: string) => request<Job>(`/api/jobs/${id}`),

  createJob: (sourceId: string, settings: JobSettingsOverrides = {}) =>
    request<Job>('/api/jobs', {
      method: 'POST',
      body: JSON.stringify({ source_id: sourceId, settings }),
    }),

  deleteJob: (id: string) => request<void>(`/api/jobs/${id}`, { method: 'DELETE' }),
  deleteSource: (id: string) => request<void>(`/api/sources/${id}`, { method: 'DELETE' }),

  cancelJob: (id: string) => request<Job>(`/api/jobs/${id}/cancel`, { method: 'POST' }),
  retryJob: (id: string) => request<Job>(`/api/jobs/${id}/retry`, { method: 'POST' }),

  listClips: (jobId: string) => request<Clip[]>(`/api/jobs/${jobId}/clips`),
  getClip: (clipId: string) => request<Clip>(`/api/clips/${clipId}`),

  getCropPath: (clipId: string) => request<CropPath>(`/api/clips/${clipId}/crop-path`),
  getClipWords: (clipId: string) => request<Word[]>(`/api/clips/${clipId}/words`),

  patchClip: (
    clipId: string,
    patch: { start_s?: number; end_s?: number; title?: string; status?: ClipStatus },
  ) => request<Clip>(`/api/clips/${clipId}`, { method: 'PATCH', body: JSON.stringify(patch) }),

  patchCaptions: (
    clipId: string,
    patch: { words?: Word[]; caption_style?: string; ratio?: string },
  ) =>
    request<Clip>(`/api/clips/${clipId}/captions`, {
      method: 'PATCH',
      body: JSON.stringify(patch),
    }),

  exportClip: (clipId: string, ratio: string, style: string, writeSrt = false) =>
    request<ExportRecord>(`/api/clips/${clipId}/export`, {
      method: 'POST',
      body: JSON.stringify({ ratio, style, write_srt: writeSrt }),
    }),

  getExport: (exportId: string) => request<ExportRecord>(`/api/exports/${exportId}`),

  captionStyles: () => request<CaptionStyle[]>('/api/caption-styles'),
  providerStatus: () => request<ProviderStatus[]>('/api/providers/status'),

  getSettings: () => request<Settings>('/api/settings'),
  putSettings: (patch: Partial<Settings>) =>
    request<Settings>('/api/settings', { method: 'PUT', body: JSON.stringify(patch) }),

  putSecret: (key: string, value: string) =>
    request<void>('/api/settings/secrets', {
      method: 'PUT',
      body: JSON.stringify({ key, value }),
    }),

  deleteSecret: (key: string) =>
    request<void>(`/api/settings/secrets/${key}`, { method: 'DELETE' }),

  mediaUrl: (jobId: string) => `/api/jobs/${jobId}/media`,

  estimateJob: (sourceId: string) => request<CostEstimate>(`/api/estimate/${sourceId}`),

  listMusicTracks: () => request<MusicTrack[]>('/api/music'),
  uploadMusicTrack: (file: File) => {
    const form = new FormData()
    form.append('file', file)
    return request<MusicTrack>('/api/music', { method: 'POST', body: form })
  },
  deleteMusicTrack: (filename: string) =>
    request<void>(`/api/music/${encodeURIComponent(filename)}`, { method: 'DELETE' }),
}

const INGEST_POLL_MS = 1500
const INGEST_TIMEOUT_MS = 60 * 60 * 1000

/**
 * Resolve an ingest response to a ready Source. Reused/uploaded sources arrive
 * complete (201); new downloads arrive as 202 {id} and are polled until the
 * background download finishes or fails.
 */
export async function waitForSource(
  accepted: Source | IngestAccepted,
  isCancelled: () => boolean = () => false,
): Promise<Source> {
  if ('duration_s' in accepted && accepted.ingest_status !== 'pending') return accepted
  const deadline = Date.now() + INGEST_TIMEOUT_MS
  while (Date.now() < deadline) {
    if (isCancelled()) throw new Error('Cancelled.')
    await new Promise((resolve) => setTimeout(resolve, INGEST_POLL_MS))
    const source = await api.getSource(accepted.id)
    if (source.ingest_status === 'failed') {
      throw new ApiError(source.ingest_error || 'Download failed.', 422)
    }
    if (source.ingest_status !== 'pending') return source
  }
  throw new ApiError('Download timed out.', 504)
}

const EXPORT_POLL_MS = 1000
const EXPORT_TIMEOUT_MS = 30 * 60 * 1000

/** Poll a background export until it is done; throws if it fails or times out. */
export async function waitForExport(
  record: ExportRecord,
  isCancelled: () => boolean = () => false,
): Promise<ExportRecord> {
  const deadline = Date.now() + EXPORT_TIMEOUT_MS
  let current = record
  while (current.status === 'pending') {
    if (isCancelled()) throw new Error('Cancelled.')
    if (Date.now() > deadline) throw new ApiError('Export timed out.', 504)
    await new Promise((resolve) => setTimeout(resolve, EXPORT_POLL_MS))
    current = await api.getExport(record.id)
  }
  if (current.status === 'failed') {
    throw new ApiError(current.error || 'Export failed.', 500)
  }
  return current
}

/** Format seconds as m:ss, or h:mm:ss past an hour. */
export function formatDuration(seconds: number): string {
  const total = Math.max(0, Math.round(seconds))
  const s = total % 60
  const m = Math.floor(total / 60) % 60
  const h = Math.floor(total / 3600)
  const pad = (n: number) => String(n).padStart(2, '0')
  return h > 0 ? `${h}:${pad(m)}:${pad(s)}` : `${m}:${pad(s)}`
}

/** Format seconds as timecode with centiseconds, for trim handles. */
export function formatTimecode(seconds: number): string {
  const total = Math.max(0, seconds)
  const cs = Math.floor((total % 1) * 100)
  const s = Math.floor(total) % 60
  const m = Math.floor(total / 60)
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${pad(m)}:${pad(s)}.${pad(cs)}`
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  const units = ['KB', 'MB', 'GB']
  let value = bytes / 1024
  let unit = 0
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024
    unit += 1
  }
  return `${value.toFixed(value < 10 ? 1 : 0)} ${units[unit]}`
}
