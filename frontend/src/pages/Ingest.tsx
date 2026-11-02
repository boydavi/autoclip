import { useCallback, useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'

import {
  ApiError,
  api,
  formatDuration,
  waitForSource,
  type CostEstimate,
  type IngestAccepted,
  type Job,
  type JobSettingsOverrides,
  type MusicTrack,
  type PromptStyle,
  type ProviderStatus,
  type Source,
} from '../api'
import { ErrorNote } from '../components/ErrorNote'

export function Ingest() {
  const navigate = useNavigate()
  const [url, setUrl] = useState('')
  const [busy, setBusy] = useState<'url' | 'file' | null>(null)
  const [error, setError] = useState<ApiError | Error | null>(null)
  const [jobs, setJobs] = useState<Job[]>([])
  const [providers, setProviders] = useState<ProviderStatus[]>([])
  const [overrides, setOverrides] = useState<JobSettingsOverrides>({})
  const [advancedOpen, setAdvancedOpen] = useState(false)
  const [dragging, setDragging] = useState(false)
  const [pendingSource, setPendingSource] = useState<Source | null>(null)
  const [estimate, setEstimate] = useState<CostEstimate | null>(null)
  const [musicTracks, setMusicTracks] = useState<MusicTrack[]>([])
  const fileInput = useRef<HTMLInputElement>(null)
  const alive = useRef(true)

  useEffect(() => {
    alive.current = true
    return () => {
      alive.current = false
    }
  }, [])

  useEffect(() => {
    api.listJobs(8).then(setJobs).catch(() => undefined)
    api.providerStatus().then(setProviders).catch(() => undefined)
    api.listMusicTracks().then(setMusicTracks).catch(() => undefined)
  }, [])

  // Step 1: ingest the source and fetch cost estimate — do not create job yet.
  const ingest = useCallback(
    async (kind: 'url' | 'file', run: () => Promise<Source | IngestAccepted>) => {
      setBusy(kind)
      setError(null)
      setPendingSource(null)
      setEstimate(null)
      try {
        const source = await waitForSource(await run(), () => !alive.current)
        if (!alive.current) return
        const est = await api.estimateJob(source.id).catch(() => null)
        setPendingSource(source)
        setEstimate(est)
      } catch (err) {
        setError(err as Error)
      } finally {
        setBusy(null)
      }
    },
    [],
  )

  // Step 2: user confirmed — create the job and navigate.
  const confirmRun = useCallback(async () => {
    if (!pendingSource) return
    setBusy('url')
    setError(null)
    try {
      const job = await api.createJob(pendingSource.id, overrides)
      navigate(`/jobs/${job.id}`)
    } catch (err) {
      setError(err as Error)
      setBusy(null)
    }
  }, [pendingSource, overrides, navigate])

  const deleteJob = async (job: Job) => {
    const name = job.source?.title || 'this job'
    if (!window.confirm(`Delete "${name}" with its clips and exports? The source video is kept.`)) {
      return
    }
    try {
      await api.deleteJob(job.id)
      setJobs((current) => current.filter((item) => item.id !== job.id))
    } catch (err) {
      setError(err as Error)
    }
  }

  const cancelPending = () => {
    setPendingSource(null)
    setEstimate(null)
    setError(null)
  }

  const submitUrl = (event: React.FormEvent) => {
    event.preventDefault()
    if (!url.trim()) return
    const link = url.trim()
    const isYouTube = /^https?:\/\/((www|m|music)\.)?(youtube\.com|youtu\.be)\//i.test(link)
    void ingest('url', () => (isYouTube ? api.ingestYouTube(link) : api.ingestUrl(link)))
  }

  const submitFile = (file: File) => void ingest('file', () => api.uploadSource(file))

  const usableProvider = providers.find((p) => p.available)

  return (
    <div className="pt-14">
      {/* Masthead. Left-aligned and asymmetric — the field is the subject, not
          a centred hero card. */}
      <div className="rise max-w-3xl">
        <p className="eyebrow">Local · No accounts · No watermarks</p>
        <h1 className="mt-5 font-display text-[clamp(2.75rem,7vw,5.5rem)] leading-[0.95] text-ink-100">
          Long video in.
          <br />
          <span className="italic text-sodium-500">Shorts</span> out.
        </h1>
      </div>

      <div className="mt-16 grid gap-x-16 gap-y-12 lg:grid-cols-[1.35fr_1fr]">
        {/* URL */}
        <section className="rise" style={{ animationDelay: '90ms' }}>
          <form onSubmit={submitUrl}>
            <label htmlFor="url" className="eyebrow">
              Paste a link
            </label>
            <div className="mt-3 flex items-end gap-4">
              <input
                id="url"
                className="field font-display text-xl md:text-2xl"
                placeholder="https://youtube.com/watch?v=…"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                autoComplete="off"
                spellCheck={false}
                disabled={busy !== null}
              />
              <button
                type="submit"
                className="btn btn-primary shrink-0"
                disabled={busy !== null || !url.trim()}
              >
                {busy === 'url' ? 'Fetching…' : 'Start'}
              </button>
            </div>
          </form>

          <p className="mt-3 text-xs leading-relaxed text-ink-500">
            Only download video you own or have the rights to process.
          </p>

          <AdvancedOptions
            open={advancedOpen}
            onToggle={() => setAdvancedOpen((v) => !v)}
            overrides={overrides}
            onChange={setOverrides}
            providers={providers}
            musicTracks={musicTracks}
          />
        </section>

        {/* Upload */}
        <section className="rise" style={{ animationDelay: '160ms' }}>
          <p className="eyebrow">Or drop a file</p>
          <div
            onDragOver={(e) => {
              e.preventDefault()
              setDragging(true)
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault()
              setDragging(false)
              const file = e.dataTransfer.files[0]
              if (file) submitFile(file)
            }}
            className={[
              'mt-3 flex min-h-52 cursor-pointer flex-col items-center justify-center gap-2 border border-dashed px-6 text-center transition-colors duration-200',
              dragging
                ? 'border-sodium-500 bg-sodium-700/10'
                : 'border-ink-700 hover:border-ink-600',
            ].join(' ')}
            onClick={() => fileInput.current?.click()}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') fileInput.current?.click()
            }}
          >
            <span className="font-display text-2xl text-ink-200">
              {busy === 'file' ? 'Uploading…' : 'Drop video or audio'}
            </span>
            <span className="text-xs text-ink-500">mp4 · mov · mkv · webm · mp3 · wav · m4a</span>
          </div>
          <input
            ref={fileInput}
            type="file"
            className="hidden"
            accept="video/*,audio/*"
            onChange={(e) => {
              const file = e.target.files?.[0]
              if (file) submitFile(file)
              e.target.value = ''
            }}
          />
        </section>
      </div>

      {error && (
        <div className="mt-10 max-w-3xl">
          <ErrorNote error={error} onDismiss={() => setError(null)} />
        </div>
      )}

      {pendingSource && (
        <CostConfirmCard
          source={pendingSource}
          estimate={estimate}
          busy={busy !== null}
          onConfirm={() => void confirmRun()}
          onCancel={cancelPending}
        />
      )}

      {providers.length > 0 && !usableProvider && !pendingSource && (
        <p className="mt-10 max-w-3xl border-l-2 border-sodium-600 pl-4 text-sm text-ink-300">
          No AI provider is reachable yet, so clip selection will fail. Add an API key or
          start Ollama in{' '}
          <a href="/settings" className="text-sodium-500 underline underline-offset-4">
            Settings
          </a>
          .
        </p>
      )}


      <RecentJobs jobs={jobs} onDelete={deleteJob} />
    </div>
  )
}

function AdvancedOptions({
  open,
  onToggle,
  overrides,
  onChange,
  providers,
  musicTracks,
}: {
  open: boolean
  onToggle: () => void
  overrides: JobSettingsOverrides
  onChange: (next: JobSettingsOverrides) => void
  providers: ProviderStatus[]
  musicTracks: MusicTrack[]
}) {
  const set = <K extends keyof JobSettingsOverrides>(key: K, value: JobSettingsOverrides[K]) =>
    onChange({ ...overrides, [key]: value })

  return (
    <div className="mt-8">
      <button type="button" onClick={onToggle} className="btn btn-quiet -ml-1">
        <span
          className="inline-block transition-transform duration-300"
          style={{ transform: open ? 'rotate(90deg)' : 'none' }}
          aria-hidden
        >
          ›
        </span>
        {open ? 'Hide options' : 'Options'}
      </button>

      {/* grid-template-rows rather than height, so the reveal animates without
          touching layout properties. */}
      <div
        className="grid transition-[grid-template-rows] duration-400 ease-[cubic-bezier(0.16,1,0.3,1)]"
        style={{ gridTemplateRows: open ? '1fr' : '0fr' }}
      >
        <div className="overflow-hidden">
          <div className="grid gap-x-8 gap-y-5 pt-5 sm:grid-cols-2">
            <Selector
              label="Content type"
              value={overrides.prompt_style ?? ''}
              onChange={(v) => set('prompt_style', (v as PromptStyle) || undefined)}
              options={[
                { value: '', label: 'General — any content' },
                { value: 'interview', label: 'Interview — Q&A, podcasts' },
                { value: 'tutorial', label: 'Tutorial — how-to, explainers' },
                { value: 'sales', label: 'Sales — demos, pitches' },
                { value: 'comedy', label: 'Comedy — stand-up, sketches' },
                { value: 'keynote', label: 'Keynote — talks, speeches' },
                { value: 'product-demo', label: 'Product demo — walkthroughs' },
                { value: 'debate', label: 'Debate — panels, discussions' },
                { value: 'storytime', label: 'Storytime — personal stories' },
              ]}
            />
            <Selector
              label="Provider"
              value={overrides.provider ?? ''}
              onChange={(v) => set('provider', v || undefined)}
              options={[
                { value: '', label: 'Use default' },
                ...providers.map((p) => ({
                  value: p.name,
                  label: p.available ? p.name : `${p.name} — unavailable`,
                  disabled: !p.available,
                })),
              ]}
            />
            <Selector
              label="Whisper model"
              value={overrides.whisper_model ?? ''}
              onChange={(v) => set('whisper_model', v || undefined)}
              options={[
                { value: '', label: 'Use default' },
                { value: 'tiny', label: 'tiny — fastest, roughest' },
                { value: 'base', label: 'base' },
                { value: 'small', label: 'small — balanced' },
                { value: 'medium', label: 'medium' },
                { value: 'large-v3', label: 'large-v3 — slowest, best' },
              ]}
            />
            <NumberField
              label="Max clips"
              value={overrides.max_clips}
              placeholder="10"
              min={1}
              max={50}
              onChange={(v) => set('max_clips', v)}
            />
            <div className="grid grid-cols-2 gap-4">
              <NumberField
                label="Min length (s)"
                value={overrides.min_duration_s}
                placeholder="20"
                min={5}
                max={300}
                onChange={(v) => set('min_duration_s', v)}
              />
              <NumberField
                label="Max length (s)"
                value={overrides.max_duration_s}
                placeholder="90"
                min={5}
                max={300}
                onChange={(v) => set('max_duration_s', v)}
              />
            </div>
            <label className="flex items-center gap-3 text-sm text-ink-200 sm:col-span-2">
              <input
                type="checkbox"
                checked={overrides.diarization ?? false}
                onChange={(e) => set('diarization', e.target.checked || undefined)}
                className="size-4 accent-sodium-500"
              />
              Multiple speakers — label who is talking, and cut to them
            </label>
            {musicTracks.length > 0 && (
              <Selector
                label="Music track"
                value={overrides.music_track ?? ''}
                onChange={(v) => set('music_track', v || undefined)}
                options={[
                  { value: '', label: 'Random (use any track)' },
                  ...musicTracks.map((t) => ({ value: t.filename, label: t.filename })),
                ]}
              />
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

function Selector({
  label,
  value,
  onChange,
  options,
}: {
  label: string
  value: string
  onChange: (value: string) => void
  options: { value: string; label: string; disabled?: boolean }[]
}) {
  return (
    <label className="block">
      <span className="eyebrow">{label}</span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="field mt-1 cursor-pointer text-sm"
      >
        {options.map((option) => (
          <option
            key={option.value}
            value={option.value}
            disabled={option.disabled}
            className="bg-ink-850"
          >
            {option.label}
          </option>
        ))}
      </select>
    </label>
  )
}

function NumberField({
  label,
  value,
  placeholder,
  min,
  max,
  onChange,
}: {
  label: string
  value: number | undefined
  placeholder: string
  min: number
  max: number
  onChange: (value: number | undefined) => void
}) {
  return (
    <label className="block">
      <span className="eyebrow">{label}</span>
      <input
        type="number"
        className="field numeric mt-1 text-sm"
        placeholder={placeholder}
        value={value ?? ''}
        min={min}
        max={max}
        onChange={(e) => onChange(e.target.value ? Number(e.target.value) : undefined)}
      />
    </label>
  )
}

function RecentJobs({ jobs, onDelete }: { jobs: Job[]; onDelete: (job: Job) => void }) {
  if (jobs.length === 0) return null

  return (
    <section className="rise mt-24" style={{ animationDelay: '240ms' }}>
      <div className="flex items-baseline justify-between border-b border-ink-800 pb-3">
        <h2 className="eyebrow">Recent</h2>
        <span className="numeric text-xs text-ink-600">{jobs.length}</span>
      </div>

      <ul>
        {jobs.map((job) => (
          <li key={job.id} className="flex items-center border-b border-ink-850">
            <a
              href={job.status === 'done' ? `/jobs/${job.id}/clips` : `/jobs/${job.id}`}
              className="group grid min-w-0 flex-1 grid-cols-[1fr_auto] items-baseline gap-4 py-4 transition-colors duration-200 hover:bg-ink-850/40 sm:grid-cols-[1fr_7rem_6rem_5rem]"
            >
              <span className="truncate text-[0.9375rem] text-ink-200 group-hover:text-ink-100">
                {job.source?.title || 'Untitled'}
              </span>
              <span className="numeric hidden text-xs text-ink-500 sm:block">
                {job.source ? formatDuration(job.source.duration_s) : '—'}
              </span>
              <span className="hidden text-xs text-ink-500 sm:block">{job.provider}</span>
              <StatusTag job={job} />
            </a>
            {job.status !== 'queued' && job.status !== 'running' && (
              <button
                type="button"
                onClick={() => onDelete(job)}
                className="btn btn-quiet ml-2 shrink-0 text-xs"
                aria-label={`Delete ${job.source?.title || 'job'}`}
              >
                Delete
              </button>
            )}
          </li>
        ))}
      </ul>
    </section>
  )
}

function StatusTag({ job }: { job: Job }) {
  const tone: Record<string, string> = {
    done: 'text-signal-good',
    failed: 'text-signal-bad',
    running: 'text-sodium-500',
    queued: 'text-ink-400',
    cancelled: 'text-ink-500',
  }
  const label =
    job.status === 'running' ? `${Math.round(job.progress * 100)}%` : job.status

  return (
    <span className={`numeric justify-self-end text-xs ${tone[job.status] ?? 'text-ink-400'}`}>
      {label}
    </span>
  )
}

function fmt(usd: number): string {
  if (usd === 0) return 'free'
  if (usd < 0.001) return '<$0.001'
  return `$${usd.toFixed(usd < 0.01 ? 4 : usd < 0.1 ? 3 : 2)}`
}

function CostConfirmCard({
  source,
  estimate,
  busy,
  onConfirm,
  onCancel,
}: {
  source: Source
  estimate: CostEstimate | null
  busy: boolean
  onConfirm: () => void
  onCancel: () => void
}) {
  return (
    <div className="mt-10 max-w-3xl border border-ink-700 bg-ink-900 p-6">
      <h2 className="eyebrow mb-4">Ready to run</h2>

      <p className="truncate text-sm font-medium text-ink-100">{source.title || 'Untitled'}</p>
      <p className="numeric mt-0.5 text-xs text-ink-500">{formatDuration(source.duration_s)}</p>

      {estimate ? (
        <table className="mt-5 w-full text-sm">
          <tbody className="divide-y divide-ink-850">
            <tr>
              <td className="py-2 text-ink-400">Transcription</td>
              <td className="numeric py-2 text-right text-ink-300">
                {estimate.whisper_provider === 'groq'
                  ? `Groq · ${estimate.windows} window${estimate.windows !== 1 ? 's' : ''}`
                  : `local · ${estimate.windows} window${estimate.windows !== 1 ? 's' : ''}`}
              </td>
              <td className="numeric py-2 pl-6 text-right text-ink-200">
                {fmt(estimate.whisper_cost_usd)}
              </td>
            </tr>
            <tr>
              <td className="py-2 text-ink-400">Clip selection</td>
              <td className="numeric py-2 text-right text-ink-300">
                {estimate.llm_provider} · {estimate.windows} × {fmt(estimate.llm_cost_per_window_usd)}
              </td>
              <td className="numeric py-2 pl-6 text-right text-ink-200">
                {fmt(estimate.llm_cost_usd)}
              </td>
            </tr>
            <tr className="border-t-2 border-ink-700">
              <td className="pt-3 font-medium text-ink-100">Estimated total</td>
              <td />
              <td className="numeric pt-3 pl-6 text-right font-medium text-sodium-400">
                {fmt(estimate.total_usd)}
              </td>
            </tr>
          </tbody>
        </table>
      ) : (
        <p className="mt-4 text-xs text-ink-500">Cost estimate unavailable.</p>
      )}

      <p className="mt-4 text-xs text-ink-600">
        Estimates based on average speech rates and public provider pricing. Actual costs may vary.
      </p>

      <div className="mt-6 flex gap-3">
        <button
          type="button"
          onClick={onConfirm}
          disabled={busy}
          className="btn btn-primary"
        >
          {busy ? 'Starting…' : 'Run'}
        </button>
        <button
          type="button"
          onClick={onCancel}
          disabled={busy}
          className="btn btn-quiet"
        >
          Cancel
        </button>
      </div>
    </div>
  )
}
