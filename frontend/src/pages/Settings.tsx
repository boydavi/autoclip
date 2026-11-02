import { useEffect, useRef, useState } from 'react'

import {
  api,
  formatBytes,
  type MusicTrack,
  type ProviderStatus,
  type Settings as SettingsData,
  type SystemStatus,
} from '../api'
import { ErrorNote } from '../components/ErrorNote'

const SECRET_LABELS: Record<string, string> = {
  anthropic: 'Anthropic API key',
  openai: 'OpenAI-compatible API key',
  gemini: 'Google Gemini API key',
  groq: 'Groq API key (clip selection and Groq transcription)',
  huggingface_token: 'HuggingFace token',
}

const X264_PRESETS = [
  'ultrafast',
  'superfast',
  'veryfast',
  'faster',
  'fast',
  'medium',
  'slow',
  'veryslow',
]

export function Settings() {
  const [settings, setSettings] = useState<SettingsData | null>(null)
  const [providers, setProviders] = useState<ProviderStatus[]>([])
  const [system, setSystem] = useState<SystemStatus | null>(null)
  const [tracks, setTracks] = useState<MusicTrack[]>([])
  const [error, setError] = useState<Error | null>(null)
  const [saved, setSaved] = useState(false)

  const reload = () => {
    void api.getSettings().then(setSettings).catch((e) => setError(e as Error))
    void api.providerStatus().then(setProviders).catch(() => undefined)
    void api.system().then(setSystem).catch(() => undefined)
    void api.listMusicTracks().then(setTracks).catch(() => undefined)
  }

  useEffect(reload, [])

  const patch = async (update: Partial<SettingsData>) => {
    setError(null)
    try {
      setSettings(await api.putSettings(update))
      setSaved(true)
      setTimeout(() => setSaved(false), 1600)
    } catch (err) {
      setError(err as Error)
    }
  }

  if (!settings) return <p className="pt-24 text-sm text-ink-500">Loading…</p>

  return (
    <div className="max-w-4xl pt-14">
      <div className="rise flex items-baseline justify-between border-b border-ink-800 pb-5">
        <h1 className="font-display text-[clamp(2rem,4vw,3rem)] leading-none text-ink-100">
          Settings
        </h1>
        <span
          className="text-xs text-signal-good transition-opacity duration-300"
          style={{ opacity: saved ? 1 : 0 }}
        >
          saved
        </span>
      </div>

      {error && (
        <div className="mt-6">
          <ErrorNote error={error} onDismiss={() => setError(null)} />
        </div>
      )}

      {settings.insecure_secret_storage && (
        <p className="mt-6 border-l-2 border-sodium-600 pl-4 text-sm leading-relaxed text-ink-300">
          No OS keyring is available on this machine, so API keys are stored in plain text
          in <code className="text-ink-200">config.json</code>. On headless Linux, installing
          a Secret Service provider or <code className="text-ink-200">keyrings.alt</code>{' '}
          fixes this.
        </p>
      )}

      <Section title="AI provider" note="Which model picks the clips.">
        <div className="space-y-1">
          {providers.map((provider) => (
            <button
              key={provider.name}
              onClick={() => patch({ active_provider: provider.name })}
              className={[
                'block w-full border-l-2 py-3 pl-3 text-left transition-colors duration-200',
                provider.name === settings.active_provider
                  ? 'border-sodium-500 bg-ink-850/60'
                  : 'border-transparent hover:border-ink-700 hover:bg-ink-850/30',
              ].join(' ')}
            >
              <div className="flex items-baseline justify-between gap-4">
                <span className="text-sm text-ink-100">{provider.name}</span>
                <span
                  className={`text-xs ${provider.available ? 'text-signal-good' : 'text-ink-500'}`}
                >
                  {provider.available ? 'reachable' : provider.detail || 'unavailable'}
                </span>
              </div>
              {provider.models.length > 0 && (
                <span className="numeric mt-1 block truncate text-xs text-ink-600">
                  {provider.models.slice(0, 6).join(' · ')}
                </span>
              )}
            </button>
          ))}
        </div>

        <div className="mt-6 grid gap-5 sm:grid-cols-2">
          <Field
            label="Model"
            value={settings.providers[settings.active_provider]?.model ?? ''}
            placeholder="model name"
            onCommit={(value) =>
              patch({
                providers: {
                  [settings.active_provider]: {
                    ...settings.providers[settings.active_provider],
                    model: value,
                  },
                },
              })
            }
          />
          <Field
            label="Base URL"
            hint="Point at OpenRouter, Groq, DeepSeek, or a local server."
            value={settings.providers[settings.active_provider]?.base_url ?? ''}
            placeholder="https://…"
            onCommit={(value) =>
              patch({
                providers: {
                  [settings.active_provider]: {
                    ...settings.providers[settings.active_provider],
                    base_url: value || null,
                  },
                },
              })
            }
          />
          <Field
            label="Parallel requests"
            hint="Leave empty for the default. Use 1 on rate-limited free-tier keys."
            value={String(settings.providers[settings.active_provider]?.concurrency ?? '')}
            placeholder="auto"
            onCommit={(value) => {
              const trimmed = value.trim()
              const parsed = Number.parseInt(trimmed, 10)
              if (trimmed !== '' && (Number.isNaN(parsed) || parsed < 1)) {
                setError(new Error('Parallel requests must be a whole number of 1 or more.'))
                return
              }
              void patch({
                providers: {
                  [settings.active_provider]: {
                    ...settings.providers[settings.active_provider],
                    concurrency: trimmed === '' ? null : parsed,
                  },
                },
              })
            }}
          />
        </div>
      </Section>

      <Section title="Keys" note="Stored in your OS keyring. Never sent anywhere but the provider.">
        <div className="space-y-4">
          {Object.entries(SECRET_LABELS).map(([key, label]) => (
            <SecretField
              key={key}
              secretKey={key}
              label={label}
              present={settings.keys_present[key] ?? false}
              onChanged={reload}
              onError={setError}
            />
          ))}
        </div>
      </Section>

      <Section title="Transcription">
        <div className="grid gap-5 sm:grid-cols-2">
          <Select
            label="Engine"
            hint={
              settings.whisper.provider === 'groq'
                ? 'Audio is sent to Groq for transcription. Fast, billed per hour of audio.'
                : 'Runs on this machine. Private and free.'
            }
            value={settings.whisper.provider}
            onChange={(value) =>
              patch({ whisper: { ...settings.whisper, provider: value as 'local' | 'groq' } })
            }
            options={['local', 'groq']}
            labels={{ local: 'Local (this machine)', groq: 'Groq cloud' }}
          />
          {settings.whisper.provider === 'groq' ? (
            <Select
              label="Groq model"
              value={settings.whisper.groq_model}
              onChange={(value) => patch({ whisper: { ...settings.whisper, groq_model: value } })}
              options={['whisper-large-v3-turbo', 'whisper-large-v3']}
              labels={{
                'whisper-large-v3-turbo': 'large-v3-turbo — fastest, cheapest',
                'whisper-large-v3': 'large-v3 — most accurate',
              }}
            />
          ) : (
            <Select
              label="Whisper model"
              value={settings.whisper.model}
              onChange={(value) => patch({ whisper: { ...settings.whisper, model: value } })}
              options={['tiny', 'base', 'small', 'medium', 'large-v3']}
            />
          )}
          <Field
            label="Language"
            hint="Leave empty to detect automatically."
            value={settings.whisper.language}
            placeholder="auto"
            onCommit={(value) => patch({ whisper: { ...settings.whisper, language: value } })}
          />
        </div>

        {settings.whisper.provider === 'groq' && !settings.keys_present.groq && (
          <p className="mt-5 border-l-2 border-sodium-600 pl-4 text-sm text-ink-300">
            Groq transcription needs a Groq API key. Add it under Keys above, or switch the
            engine back to Local — otherwise jobs will fail at the transcription step.
          </p>
        )}

        <label className="mt-5 flex items-start gap-3 text-sm text-ink-200">
          <input
            type="checkbox"
            checked={settings.whisper.diarization}
            disabled={!system?.diarization_available}
            onChange={(e) =>
              patch({ whisper: { ...settings.whisper, diarization: e.target.checked } })
            }
            className="mt-0.5 size-4 accent-sodium-500"
          />
          <span>
            Identify speakers
            {!system?.diarization_available && (
              <span className="mt-1 block text-xs text-ink-500">
                Needs the diarization extra:{' '}
                <code className="text-ink-300">uv pip install &apos;autoclip[diarization]&apos;</code>
              </span>
            )}
          </span>
        </label>
      </Section>

      <Section title="Clips">
        <div className="grid gap-5 sm:grid-cols-3">
          <NumberField
            label="Min length (s)"
            value={settings.clips.min_duration_s}
            onCommit={(value) => patch({ clips: { ...settings.clips, min_duration_s: value } })}
          />
          <NumberField
            label="Max length (s)"
            value={settings.clips.max_duration_s}
            onCommit={(value) => patch({ clips: { ...settings.clips, max_duration_s: value } })}
          />
          <NumberField
            label="Max clips"
            value={settings.clips.max_clips}
            onCommit={(value) => patch({ clips: { ...settings.clips, max_clips: value } })}
          />
        </div>
      </Section>

      <Section title="Ingest">
        <div className="grid gap-5 sm:grid-cols-2">
          <Select
            label="YouTube cookies from"
            hint="YouTube blocks most anonymous downloads. Sign in in that browser and close it before downloading."
            value={settings.ingest.cookies_from_browser}
            onChange={(value) =>
              patch({ ingest: { ...settings.ingest, cookies_from_browser: value } })
            }
            options={['', 'chrome', 'firefox', 'edge', 'brave', 'chromium', 'safari']}
            labels={{ '': 'None' }}
          />
        </div>

        <label className="mt-5 flex items-start gap-3 text-sm text-ink-200">
          <input
            type="checkbox"
            checked={settings.ingest.prefer_youtube_captions}
            onChange={(e) =>
              patch({ ingest: { ...settings.ingest, prefer_youtube_captions: e.target.checked } })
            }
            className="mt-0.5 size-4 accent-sodium-500"
          />
          <span>
            Use YouTube&apos;s own captions when available
            <span className="mt-1 block text-xs text-ink-500">
              Skips transcription for videos that already have captions. Faster and free, but
              less accurate than Whisper and without speaker labels.
            </span>
          </span>
        </label>
      </Section>

      <Section title="Export">
        <div className="grid gap-5 sm:grid-cols-2">
          <Select
            label="Default ratio"
            value={settings.export.ratio}
            onChange={(value) => patch({ export: { ...settings.export, ratio: value } })}
            options={['9:16', '1:1', '16:9']}
          />
          <NumberField
            label="Loudness target (LUFS)"
            value={settings.export.loudness_lufs}
            onCommit={(value) => patch({ export: { ...settings.export, loudness_lufs: value } })}
          />
          <NumberField
            label="Quality (CRF, lower = better)"
            value={settings.export.crf}
            onCommit={(value) => {
              if (!Number.isInteger(value) || value < 1 || value > 51) {
                setError(new Error('Quality (CRF) must be a whole number from 1 to 51.'))
                return
              }
              void patch({ export: { ...settings.export, crf: value } })
            }}
          />
          <Select
            label="Encoder speed (CPU)"
            hint="Slower presets make smaller files at the same quality. GPU encoding ignores this."
            value={settings.export.preset}
            onChange={(value) => patch({ export: { ...settings.export, preset: value } })}
            options={X264_PRESETS}
          />
        </div>

        <label className="mt-5 flex items-start gap-3 text-sm text-ink-200">
          <input
            type="checkbox"
            checked={settings.export.scan_only}
            onChange={(e) => patch({ export: { ...settings.export, scan_only: e.target.checked } })}
            className="mt-0.5 size-4 accent-sodium-500"
          />
          <span>
            Find clips only — don&apos;t render
            <span className="mt-1 block text-xs text-ink-500">
              Jobs stop after picking highlights. You review them first and export only the
              ones you keep.
            </span>
          </span>
        </label>

        <label className="mt-5 flex items-start gap-3 text-sm text-ink-200">
          <input
            type="checkbox"
            checked={settings.export.prefer_hardware_encoder}
            onChange={(e) =>
              patch({
                export: { ...settings.export, prefer_hardware_encoder: e.target.checked },
              })
            }
            className="mt-0.5 size-4 accent-sodium-500"
          />
          <span>
            Use GPU encoding when available
            {system && !system.nvenc_works && (
              <span className="mt-1 block text-xs text-ink-500">
                Not usable on this machine — exports will use the CPU encoder. Same quality,
                slower.
              </span>
            )}
          </span>
        </label>

        <label className="mt-4 flex items-center gap-3 text-sm text-ink-200">
          <input
            type="checkbox"
            checked={settings.export.write_srt}
            onChange={(e) => patch({ export: { ...settings.export, write_srt: e.target.checked } })}
            className="size-4 accent-sodium-500"
          />
          Also write an .srt sidecar
        </label>

        <div className="mt-6 border-t border-ink-850 pt-5">
          <label className="flex items-start gap-3 text-sm text-ink-200">
            <input
              type="checkbox"
              checked={settings.export.music_enabled}
              onChange={(e) =>
                patch({ export: { ...settings.export, music_enabled: e.target.checked } })
              }
              className="mt-0.5 size-4 accent-sodium-500"
            />
            <span>
              Mix background music under speech
              <span className="mt-1 block text-xs text-ink-500">
                Drop .mp3 / .m4a / .wav / .flac files into the{' '}
                <code className="text-ink-300">music/</code> folder next to your data directory.
                AutoClip picks one at random per clip.
              </span>
            </span>
          </label>

          {settings.export.music_enabled && (
            <div className="mt-4 max-w-xs">
              <MusicVolumeSlider
                value={settings.export.music_volume}
                onCommit={(v) => patch({ export: { ...settings.export, music_volume: v } })}
              />
            </div>
          )}
        </div>
      </Section>

      <Section
        title="Music tracks"
        note="Tracks mixed under speech when music is enabled. AutoClip picks one at random per clip."
      >
        <MusicPanel
          tracks={tracks}
          onChanged={() => void api.listMusicTracks().then(setTracks)}
          onError={setError}
        />
      </Section>

      {system && (
        <Section title="This machine">
          <dl className="grid gap-x-8 gap-y-3 text-sm sm:grid-cols-2">
            <Row label="Platform" value={system.platform} />
            <Row label="Python" value={system.python_version} />
            <Row label="ffmpeg" value={system.ffmpeg_version ?? 'not found'} />
            <Row label="Acceleration" value={system.accel.toUpperCase()} />
            <Row label="Device" value={system.gpu_name ?? '—'} />
            <Row label="Whisper compute" value={system.compute_type} />
            <Row label="GPU encode" value={system.nvenc_works ? 'available' : 'unavailable'} />
            <Row label="Captions" value={system.has_libass ? 'libass present' : 'libass missing'} />
          </dl>
        </Section>
      )}
    </div>
  )
}

function Section({
  title,
  note,
  children,
}: {
  title: string
  note?: string
  children: React.ReactNode
}) {
  return (
    <section className="rise mt-14">
      <div className="border-b border-ink-800 pb-2">
        <h2 className="eyebrow">{title}</h2>
        {note && <p className="mt-1 text-xs text-ink-500">{note}</p>}
      </div>
      <div className="mt-5">{children}</div>
    </section>
  )
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline justify-between gap-4 border-b border-ink-850 pb-2">
      <dt className="text-ink-500">{label}</dt>
      <dd className="numeric truncate text-right text-ink-200">{value}</dd>
    </div>
  )
}

/** Commits on blur rather than per keystroke, so a PUT isn't fired per letter. */
function Field({
  label,
  hint,
  value,
  placeholder,
  onCommit,
}: {
  label: string
  hint?: string
  value: string
  placeholder?: string
  onCommit: (value: string) => void
}) {
  const [draft, setDraft] = useState(value)
  useEffect(() => setDraft(value), [value])

  return (
    <label className="block">
      <span className="eyebrow">{label}</span>
      <input
        className="field mt-1 text-sm"
        value={draft}
        placeholder={placeholder}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={() => draft !== value && onCommit(draft)}
        onKeyDown={(e) => e.key === 'Enter' && e.currentTarget.blur()}
        spellCheck={false}
      />
      {hint && <span className="mt-1.5 block text-xs leading-snug text-ink-500">{hint}</span>}
    </label>
  )
}

function NumberField({
  label,
  value,
  onCommit,
}: {
  label: string
  value: number
  onCommit: (value: number) => void
}) {
  const [draft, setDraft] = useState(String(value))
  useEffect(() => setDraft(String(value)), [value])

  return (
    <label className="block">
      <span className="eyebrow">{label}</span>
      <input
        type="number"
        className="field numeric mt-1 text-sm"
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={() => {
          const parsed = Number(draft)
          if (!Number.isNaN(parsed) && parsed !== value) onCommit(parsed)
        }}
        onKeyDown={(e) => e.key === 'Enter' && e.currentTarget.blur()}
      />
    </label>
  )
}

function Select({
  label,
  hint,
  value,
  onChange,
  options,
  labels = {},
}: {
  label: string
  hint?: string
  value: string
  onChange: (value: string) => void
  options: string[]
  labels?: Record<string, string>
}) {
  return (
    <label className="block">
      <span className="eyebrow">{label}</span>
      <select
        className="field mt-1 cursor-pointer text-sm"
        value={value}
        onChange={(e) => onChange(e.target.value)}
      >
        {options.map((option) => (
          <option key={option} value={option} className="bg-ink-850">
            {labels[option] ?? option}
          </option>
        ))}
      </select>
      {hint && <span className="mt-1.5 block text-xs leading-snug text-ink-500">{hint}</span>}
    </label>
  )
}

function MusicPanel({
  tracks,
  onChanged,
  onError,
}: {
  tracks: MusicTrack[]
  onChanged: () => void
  onError: (error: Error) => void
}) {
  const fileInput = useRef<HTMLInputElement>(null)
  const [busy, setBusy] = useState(false)

  const upload = async (file: File) => {
    setBusy(true)
    try {
      await api.uploadMusicTrack(file)
      onChanged()
    } catch (err) {
      onError(err as Error)
    } finally {
      setBusy(false)
    }
  }

  const remove = async (filename: string) => {
    try {
      await api.deleteMusicTrack(filename)
      onChanged()
    } catch (err) {
      onError(err as Error)
    }
  }

  return (
    <div>
      {tracks.length === 0 ? (
        <p className="text-sm text-ink-500">No tracks loaded yet.</p>
      ) : (
        <ul className="divide-y divide-ink-850">
          {tracks.map((track) => (
            <li key={track.filename} className="flex items-baseline justify-between gap-4 py-2.5">
              <span className="truncate text-sm text-ink-200">{track.filename}</span>
              <div className="flex shrink-0 items-baseline gap-4">
                <span className="numeric text-xs text-ink-600">{formatBytes(track.size_bytes)}</span>
                <button
                  onClick={() => void remove(track.filename)}
                  className="btn btn-quiet text-xs"
                >
                  Remove
                </button>
              </div>
            </li>
          ))}
        </ul>
      )}

      <input
        ref={fileInput}
        type="file"
        className="hidden"
        accept=".mp3,.m4a,.wav,.ogg,.flac,.aac,audio/*"
        onChange={(e) => {
          const file = e.target.files?.[0]
          if (file) void upload(file)
          e.target.value = ''
        }}
      />
      <button
        type="button"
        onClick={() => fileInput.current?.click()}
        disabled={busy}
        className="btn btn-ghost mt-4"
      >
        {busy ? 'Uploading…' : '+ Add track'}
      </button>
    </div>
  )
}

function MusicVolumeSlider({
  value,
  onCommit,
}: {
  value: number
  onCommit: (value: number) => void
}) {
  const [draft, setDraft] = useState(value)
  useEffect(() => setDraft(value), [value])
  const pct = Math.round(draft * 100)

  return (
    <label className="block">
      <span className="eyebrow flex justify-between">
        <span>Music volume</span>
        <span className="numeric text-ink-400">{pct}%</span>
      </span>
      <input
        type="range"
        className="mt-2 w-full accent-sodium-500"
        min={0}
        max={0.5}
        step={0.01}
        value={draft}
        onChange={(e) => setDraft(Number(e.target.value))}
        onMouseUp={() => draft !== value && onCommit(draft)}
        onTouchEnd={() => draft !== value && onCommit(draft)}
        onKeyUp={() => draft !== value && onCommit(draft)}
      />
      <div className="mt-1 flex justify-between text-xs text-ink-600">
        <span>silent</span>
        <span>50%</span>
      </div>
    </label>
  )
}

function SecretField({
  secretKey,
  label,
  present,
  onChanged,
  onError,
}: {
  secretKey: string
  label: string
  present: boolean
  onChanged: () => void
  onError: (error: Error) => void
}) {
  const [value, setValue] = useState('')
  const [busy, setBusy] = useState(false)

  const save = async () => {
    if (!value.trim()) return
    setBusy(true)
    try {
      await api.putSecret(secretKey, value.trim())
      setValue('')
      onChanged()
    } catch (err) {
      onError(err as Error)
    } finally {
      setBusy(false)
    }
  }

  const remove = async () => {
    setBusy(true)
    try {
      await api.deleteSecret(secretKey)
      onChanged()
    } catch (err) {
      onError(err as Error)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="flex flex-wrap items-end gap-3">
      <label className="min-w-56 flex-1">
        <span className="eyebrow">
          {label}
          {present && <span className="ml-2 text-signal-good">set</span>}
        </span>
        <input
          type="password"
          className="field mt-1 text-sm"
          value={value}
          placeholder={present ? '••••••••••••' : 'paste to add'}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && save()}
          autoComplete="off"
        />
      </label>
      <button onClick={save} disabled={!value.trim() || busy} className="btn btn-ghost">
        Save
      </button>
      {present && (
        <button onClick={remove} disabled={busy} className="btn btn-quiet">
          Remove
        </button>
      )}
    </div>
  )
}
