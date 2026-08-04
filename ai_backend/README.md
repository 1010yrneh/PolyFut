# AI scout report backend

Moves the Groq API key off the user's machine and onto a Modal endpoint we host,
so nobody has to make a Groq account to get a scouting report.

```
[PolyFut app] --messages--> [Modal endpoint] --GROQ_API_KEY--> [Groq]
              <---text-----                 <---text---------
```

The Groq key exists only in Modal's secret store. It is not in the installer,
not in `script.js`, and never travels to a client.

---

## What you need to do (the parts that need your accounts)

### 1. Groq key

`console.groq.com` → sign up → **API Keys** → **Create API Key** → name it
`polyfut-backend`. Copy the `gsk_...` value. You paste it exactly once, in
step 3.

### 2. Modal

```bash
pip install modal
modal setup
```

### 3. Create both secrets

```bash
modal secret create groq-secret GROQ_API_KEY=gsk_your_actual_key_here
```

Generate the shared app token, then store it:

```bash
python -c "import secrets; print(secrets.token_hex(16))"
```

```bash
modal secret create polyfut-app-token POLYFUT_APP_TOKEN=the_hex_string_you_just_generated
```

Keep that hex string — you need it again in step 6.

> The app token is stored as a secret rather than written into
> `report_backend.py` so it stays out of git and can be rotated without a commit.

### 4. Test before deploying

```bash
modal serve ai_backend/report_backend.py
```

Modal prints a temporary URL. Check it end to end:

```bash
curl -X POST https://YOUR-TEMP-URL.modal.run -H "Content-Type: application/json" -d "{\"app_token\":\"YOUR_TOKEN\",\"messages\":[{\"role\":\"user\",\"content\":\"Say OK\"}]}"
```

Expect `{"report":"OK...","model":"llama-3.3-70b-versatile","usage":{...}}`.
A wrong token must return HTTP 401 `{"error":"unauthorized"}` — worth checking,
since that guard is what protects your free quota.

### 5. Deploy

```bash
modal deploy ai_backend/report_backend.py
```

This prints two stable URLs — one for `generate_report`, one for `health`. You
want the **generate_report** one.

### 6. Point the app at it

Copy `ai_backend/ai_config.example.json` to the **repo root** as `ai_config.json`
and fill in both fields:

```json
{
  "proxy_url": "https://you--polyfut-report-backend-generate-report.modal.run",
  "app_token": "the_hex_string_from_step_3"
}
```

`ai_config.json` is gitignored. `POLYFUT_AI_PROXY_URL` / `POLYFUT_AI_APP_TOKEN`
env vars override it if you'd rather not use a file.

Restart `server.py` and confirm:

```bash
curl http://127.0.0.1:5000/api/ai_config
```

`{"enabled": true, ...}` means the client will use the proxy. `enabled: false`
means it will fall back to a user-pasted key, i.e. the old behaviour.

### 7. Ship it

`ai_config.json` must be included in the installer for users to get the proxy —
add it to `packaging/pyinstaller.spec` datas alongside the other root files.
**This is not wired up yet** — see "Still to do" below.

---

## How the client uses it

`script.js` has one helper, `pfAiChat(messages)`, used by both
`generateScoutReport()` and `askFollowUp()`:

1. On load, `GET /api/ai_config`. If a proxy is configured, the bring-your-own-key
   onboarding and the results-page key box are hidden.
2. Report and follow-up calls go to the proxy with the full `aiChatHistory`, so
   multi-turn conversation works exactly as before.
3. If the proxy is unreachable (network failure) **and** the user has a saved key,
   it falls back to calling Groq directly. A 429 never falls back — that means
   the shared quota is exhausted and hammering Groq directly won't help.
4. If there is no proxy and no key, the AI buttons explain that reports are
   unavailable instead of asking for a key.

## Limits, and why they're there

Groq's free tier is shared at the **organisation** level — every PolyFut user
draws from one bucket. `report_backend.py` caps:

| Limit | Value | Why |
|---|---|---|
| `MAX_REPORTS_PER_IP_PER_HOUR` | 20 | One extracted token can't drain the day's quota |
| `MAX_MESSAGES` | 40 | A report plus ~19 follow-ups; then start fresh |
| `MAX_TOTAL_CHARS` | 24,000 | A long match timeline can't blow the shared per-minute token budget |
| `MAX_OUTPUT_TOKENS` | 2,000 | The report asks for ~500 words over 9 sections; 800 would truncate it |
| `MAX_IMAGES` | 5 | Groq's per-request cap for vision |
| `MAX_IMAGE_BYTES` | 400,000 | Per image; a 640x360 JPEG still is far under it |
| `MAX_IMAGE_BYTES_TOTAL` | 1,500,000 | One kit read can't push a multi-MB upload through the shared key |

Rate-limit state lives in a `modal.Dict`, so it holds across containers when
Modal scales out.

## Vision: the kit-colour read

The endpoint also accepts image content, used once per match to decide the two
team kit colours. This matters more than it sounds: measured on an ISB v TAS
recording, the team gate keeps **98%** of your touches with the right kit pair
and **16-26%** with a wrong one, because everything it decides is "the other
team" is dropped outright.

The model is shown a few match stills **plus the colours measured off players in
that video**, and picks two of them by letter. It cannot return a colour that
isn't on the pitch. That design is forced, not fussy: letting it emit a free hex
and snapping to the nearest measurement can't be made safe, because a correct
kit named vividly (`#ffa500` for a shirt measuring `#8f561b`, dE76 91) lands
further away than a plainly wrong one (`#ff0000`, dE76 72).

Only inline `data:image/...` URIs are accepted. A caller-supplied `http(s)` URL
is refused — this endpoint is public, and fetching URLs on request would make it
a proxy for arbitrary hosts.

**This needs a redeploy.** The previously deployed build validates message
content as a string and rejects images, so the app keeps the switch off by
default:

```bash
modal deploy ai_backend/report_backend.py
```

Then turn it on with `POLYFUT_KIT_VISION=1`, or `"kit_vision": true` in
`ai_config.json`. Left off, kit colours come from local k-means exactly as
before — no network, no key, no frames leaving the machine.

## Still to do

- **Installer packaging**: `ai_config.json` isn't in `packaging/pyinstaller.spec`
  yet, so a built installer will report `enabled: false` and fall back to the
  key flow. Needs doing before release.
- **`help.js` and the website** still describe the bring-your-own-key setup
  (`GroqSetup1-6.png`, `website/index.html`). That copy is now wrong for the
  default path and should be rewritten once the proxy is live.
- **Watch the Groq console** for the first few days of real use. The published
  free-tier numbers move, and the per-minute *token* ceiling — not the request
  ceiling — is what a long match timeline will hit first.
