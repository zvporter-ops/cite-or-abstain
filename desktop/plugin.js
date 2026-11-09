// Coldstart — desktop half.
//
// Shows what the gate just did, what the corpus is made of, and lets the user change the
// mode or run the trust battery without leaving the app. Every value comes from the plugin's
// own backend (/api/plugins/coldstart/) — the same code the /trust command runs.

import { STATUSBAR_AREAS } from '@hermes/plugin-sdk'
import { jsx, jsxs } from 'react/jsx-runtime'
import { useCallback, useEffect, useRef, useState } from 'react'

const MODES = ['off', 'flag', 'strict']

const muted = { color: 'var(--ui-text-tertiary, #8b8b8b)', fontSize: '0.75rem' }
const row = { display: 'flex', justifyContent: 'space-between', gap: '0.5rem' }
const label = { fontSize: '0.75rem', fontWeight: 600, opacity: 0.9 }
const panel = {
  border: '1px solid var(--ui-border, rgba(127,127,127,0.25))',
  borderRadius: '0.375rem',
  padding: '0.5rem',
  display: 'flex',
  flexDirection: 'column',
  gap: '0.25rem'
}
const button = {
  border: '1px solid var(--ui-border, rgba(127,127,127,0.25))',
  borderRadius: '0.25rem',
  padding: '0.125rem 0.5rem',
  fontSize: '0.6875rem',
  background: 'transparent',
  color: 'inherit',
  cursor: 'pointer'
}

function verdictColour(verdict) {
  if (verdict === 'trusted') return 'var(--ui-success, #3fb950)'
  if (verdict === 'partial') return 'var(--ui-warning, #d29922)'
  return 'var(--ui-error, #f85149)'
}

export default {
  id: 'coldstart',
  name: 'Coldstart',
  defaultEnabled: true,

  register(ctx) {
    const get = (path) => ctx.rest(path)
    const post = (path, body) => ctx.rest(path, { method: 'POST', body })

    function EvidencePane() {
      const [status, setStatus] = useState(null)
      const [audit, setAudit] = useState(null)
      const [busy, setBusy] = useState(false)
      const [error, setError] = useState('')
      const timer = useRef(null)

      const load = useCallback(async () => {
        try {
          setStatus(await get('/status'))
          setError('')
        } catch (err) {
          setError(err && err.message ? err.message : String(err))
        }
      }, [])

      useEffect(() => {
        load()
      }, [load])

      // A finished turn rewrites last_gate.json; refresh shortly after, coalesced.
      useEffect(() => {
        const off = ctx.onEvent('*', () => {
          if (timer.current) clearTimeout(timer.current)
          timer.current = setTimeout(() => {
            timer.current = null
            load()
          }, 600)
        })
        return () => {
          off()
          if (timer.current) clearTimeout(timer.current)
        }
      }, [load])

      const chooseMode = async (mode) => {
        try {
          await post('/mode', { mode })
          await load()
        } catch (err) {
          setError(err && err.message ? err.message : String(err))
        }
      }

      const runAudit = async () => {
        setBusy(true)
        setAudit(null)
        try {
          setAudit(await post('/audit', {}))
        } catch (err) {
          setError(err && err.message ? err.message : String(err))
        } finally {
          setBusy(false)
        }
      }

      const corpus = (status && status.corpus) || { documents: 0, passages: 0, by_tier: {} }
      const tiers = corpus.by_tier || {}
      const gate = status && status.last_gate
      const mode = (status && status.mode) || 'flag'

      return jsxs('div', {
        style: { padding: '0.75rem', display: 'flex', flexDirection: 'column', gap: '0.625rem' },
        children: [
          jsxs('div', {
            style: row,
            children: [
              jsx('div', { style: label, children: 'Evidence gate' }),
              jsx('div', {
                style: { ...muted, fontVariantNumeric: 'tabular-nums' },
                children: `${corpus.documents} docs · ${tiers.internal || 0} internal · ${tiers.public || 0} public`
              })
            ]
          }),

          jsxs('div', {
            style: { ...panel, ...row, alignItems: 'center' },
            children: [
              jsx('div', { style: muted, children: 'Mode' }),
              jsxs('div', {
                style: { display: 'flex', gap: '0.25rem' },
                children: MODES.map((option) =>
                  jsx('button', {
                    type: 'button',
                    onClick: () => chooseMode(option),
                    style: {
                      ...button,
                      fontWeight: option === mode ? 600 : 400,
                      borderColor: option === mode
                        ? 'var(--ui-accent, #58a6ff)'
                        : 'var(--ui-border, rgba(127,127,127,0.25))'
                    },
                    children: option
                  }, option)
                )
              })
            ]
          }),

          jsxs('div', {
            style: { ...panel, ...row },
            children: [
              jsx('div', { style: muted, children: 'Session model' }),
              jsx('div', {
                style: { fontSize: '0.75rem' },
                children: status && status.model === 'local'
                  ? 'local — internal material shareable'
                  : 'remote — internal material withheld'
              })
            ]
          }),

          jsx('div', {
            style: label,
            children: 'Last reply'
          }),
          gate
            ? jsxs('div', {
              style: { ...panel, gap: '0.25rem' },
              children: [
                jsxs('div', {
                  style: row,
                  children: [
                    jsx('div', {
                      style: {
                        fontSize: '0.75rem',
                        fontWeight: 600,
                        color: gate.ok ? 'var(--ui-success, #3fb950)' : 'var(--ui-warning, #d29922)'
                      },
                      children: gate.ok
                        ? 'supported'
                        : `${gate.violations.length} unsupported claim${gate.violations.length === 1 ? '' : 's'}`
                    }),
                    jsx('div', {
                      style: muted,
                      children: `${Math.round((gate.coverage_internal || 0) * 100)}% internal`
                    })
                  ]
                }),
                gate.withheld_internal
                  ? jsx('div', {
                    style: { ...muted, color: 'var(--ui-warning, #d29922)' },
                    children: `${gate.withheld_internal} internal passage(s) withheld from this model`
                  })
                  : null,
                gate.public_only && gate.public_only.length
                  ? jsx('div', {
                    style: muted,
                    children: `${gate.public_only.length} claim(s) rested on public material`
                  })
                  : null,
                ...(gate.violations || []).slice(0, 4).map((violation, index) =>
                  jsx('div', {
                    style: { ...muted, wordBreak: 'break-word' },
                    children: `${violation.kind}: ${violation.span}`
                  }, `v${index}`)
                )
              ]
            })
            : jsx('div', { style: muted, children: 'No gated turn yet in this session.' }),

          jsxs('div', {
            style: { ...panel, ...row, alignItems: 'center' },
            children: [
              jsx('div', { style: muted, children: 'Trust score' }),
              jsxs('div', {
                style: { display: 'flex', alignItems: 'center', gap: '0.5rem' },
                children: [
                  audit && !audit.error
                    ? jsx('div', {
                      style: {
                        fontSize: '0.75rem',
                        fontWeight: 700,
                        color: verdictColour(audit.verdict)
                      },
                      children: `${audit.verdict} ${audit.score}/100`
                    })
                    : null,
                  jsx('button', {
                    type: 'button',
                    onClick: runAudit,
                    disabled: busy,
                    style: { ...button, opacity: busy ? 0.5 : 1 },
                    children: busy ? 'measuring…' : 'Run audit'
                  })
                ]
              })
            ]
          }),

          audit && audit.error
            ? jsx('div', { style: { ...muted, color: 'var(--ui-error, #f85149)' }, children: audit.error })
            : null,
          error
            ? jsx('div', { style: { ...muted, color: 'var(--ui-error, #f85149)' }, children: error })
            : null
        ]
      })
    }

    function StatusChip() {
      const [status, setStatus] = useState(null)
      const timer = useRef(null)

      const load = useCallback(async () => {
        try {
          setStatus(await get('/status'))
        } catch (err) {
          setStatus(null)
        }
      }, [])

      useEffect(() => {
        load()
        const off = ctx.onEvent('*', () => {
          if (timer.current) clearTimeout(timer.current)
          timer.current = setTimeout(() => {
            timer.current = null
            load()
          }, 600)
        })
        return () => {
          off()
          if (timer.current) clearTimeout(timer.current)
        }
      }, [load])

      const gate = status && status.last_gate
      const text = !status
        ? 'evidence'
        : !gate
          ? 'evidence ready'
          : gate.ok
            ? 'evidence ok'
            : `${gate.violations.length} flagged`
      const colour = gate && !gate.ok ? 'var(--ui-warning, #d29922)' : 'var(--ui-text-tertiary, #8b8b8b)'

      return jsx('div', {
        style: { fontSize: '0.6875rem', color: colour, padding: '0 0.375rem' },
        children: text
      })
    }

    ctx.register({
      id: 'pane',
      area: 'panes',
      title: 'evidence',
      data: { placement: 'right', width: '320px' },
      render: () => jsx(EvidencePane, {})
    })

    ctx.register({
      id: 'chip',
      area: STATUSBAR_AREAS.right,
      order: 140,
      render: () => jsx(StatusChip, {})
    })
  }
}
