# SF-24 harness finding

Choice: **Gemini API fallback**, model `gemini-3.8-flash`. The key is read only from the environment variable `GEMINI_API_KEY`, which CI populates from the GitHub secret of the same name.

Grok CLI is not the adapter. The vendor pages below show that the `grok` binary can run headless, and they show that unattended CI is authenticated with `XAI_API_KEY`. They do not show that this headless CI path runs on the existing Cursor plan with no new paid subscription. The instruction for an unclear case is the Gemini fallback. No xAI account, Cursor API key, or Gemini key was created for this finding, and neither API was called.

## What "Grok CLI" is

Paper figure 5 names Grok CLI (Grok 4.7) as one of six harnesses. The xAI docs for that product call the binary `grok`. Cursor's headless CLI is a different program, `agent`, and a different secret, `CURSOR_API_KEY`.

xAI, [Headless & Scripting](https://docs.x.ai/build/cli/headless-scripting) (read 6 Nov 2026):

> Use headless mode for scripts, bots, or other machine-friendly tasks.
>
> `grok -p "Your prompt here"`

The same page, on CI:

> When using headless mode (`-p`) or ACP (`grok agent stdio`) in scripts, CI, or other automated environments, pass `--no-auto-update` (e.g. `grok --no-auto-update -p "..."`) to skip background update checks.

And on auth for the ACP example:

> The example below assumes `grok` is already authenticated locally, or `XAI_API_KEY` is set.

The example then errors with:

> Run `grok login` first, or set XAI_API_KEY.

xAI, [Enterprise Deployments](https://docs.x.ai/build/enterprise) (read 6 Nov 2026), authentication table:

> API key | `XAI_API_KEY` env var or `model.api_key` in config | No | Scripts, CI/CD, headless automation

The same page:

> For CI/CD and headless automation, set the `XAI_API_KEY` environment variable — no config file needed:
>
> `export XAI_API_KEY="xai-..."`
>
> `grok -p "Review this diff" --output-format json --always-approve`

xAI, [Quickstart](https://docs.x.ai/docs/tutorial) (read 6 Nov 2026), for that API key:

> Sign up for an account at console.x.ai, then load it with credits to start using the API.
>
> Create an API key via the API Keys page, then export it or add it as an environment variable.

Loading credits is a paid meter. It is not described as included usage on a Cursor plan.

Device-code login exists for machines without a browser. xAI, [Enterprise Deployments](https://docs.x.ai/build/enterprise):

> For environments without a browser (SSH, containers, cloud devboxes), device code login follows RFC 8628:
>
> `grok login --device-auth`
>
> Grok prints a URL and a short user code. Complete login on any device with a browser.

That still needs a person to finish login. It is not an unattended GitHub Actions secret already present on the Cursor plan, and this work was not allowed to create tokens.

## What the Cursor plan actually includes

Cursor, [Models & Pricing](https://cursor.com/docs/models-and-pricing) (read 6 Nov 2026):

> Cursor is sold as part of The X Subscription, one subscription for Grok, Cursor, and X. Paid plans include a weekly pool of usage for Cursor models, shared across Grok and Cursor.

Plus, on that page:

> Also Grok Chat, Voice, Imagine, Build, and Grok API access. No Auto, Cloud Agents, Automations, Bugbot, Origin, third-party models, or your own API keys.

The included pool, same page:

> The pool is shared across everything on your subscription: Agent in the editor, CLI, Cloud Agents, Grok Bot, and the Grok apps.
>
> The pool covers Grok 4.7, Grok 4.6, Grok 4.5, and Composer 2.5.

"CLI" in that sentence is Cursor's CLI. Cursor, [Using Headless CLI](https://cursor.com/docs/cli/headless):

> # Set API key for scripts
> export CURSOR_API_KEY=your_api_key_here
> agent -p "Analyze this code"

Cursor, [GitHub Actions](https://cursor.com/docs/cli/github-actions):

> CURSOR_API_KEY: ${{ secrets.CURSOR_API_KEY }}
>
> agent -p "Your prompt here" --model gpt-5

The command is `agent`, not `grok`. A `CURSOR_API_KEY` would be a new token. This finding did not create one. The docs do not say that a Cursor plan login is accepted by the `grok` binary.

Cursor, [Pricing](https://cursor.com/pricing) (read 6 Nov 2026), lists Grok Build as included from Plus upward, and Grok API access as included from Plus upward. The same page's usage answer does not put the API on the included pool:

> Every tier includes a pool of AI usage shared across Cursor and Grok. Within Grok, the pool covers Chat, Imagine, Voice, and Bot.

And:

> Is the Grok API included in my plan?
>
> Paid plans include Grok API access. API usage is billed separately at standard model rates and doesn't draw from your plan's usage pool.

So the credential xAI documents for headless CI (`XAI_API_KEY`) is the API path, and Cursor's own pricing says that API usage is billed separately and does not draw from the plan pool. That is a new paid meter, even on a plan that already lists "Grok API access" and "Grok Build".

## Why this is not "Grok CLI on the existing plan"

Headless `grok -p` in CI is documented. What is not documented is running that binary in GitHub Actions on the included Cursor weekly pool, with no new subscription and no new API credential.

- Unattended CI auth for `grok` is `XAI_API_KEY` from console.x.ai, after the account is loaded with credits.
- Cursor prices that API separately from the weekly pool.
- Cursor's headless CLI is `agent` plus `CURSOR_API_KEY`, which is a different product and a token this task must not create.
- `grok login --device-auth` still needs an interactive browser confirmation. The docs do not say the resulting session is the Cursor plan's included pool when the process is a CI job.

Because those pages do not establish the no-new-paid-subscription path, the adapter is the Gemini fallback.

## Adapter that follows this choice

`sceneagent.harness` checks `agent_loop.harness_status` before a call. That status still lists Codex, Claude Code, Antigravity, Muse, Stirrup, and Grok CLI, with `called` empty. The one callable adapter is Gemini `gemini-3.8-flash`. `GEMINI_API_KEY` is the only credential read. A missing value raises `MissingGeminiApiKeyError` before any HTTP call. This part does not perform a live call. The live transcript and `plan.json` for dc2 and dc3 wait on the evidence run, which waits on PR #17 (SF-14 / KAN-14).
