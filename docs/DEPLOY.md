# Deploying MudraPragyan.AI

The app is a static site. There is no server, no database, and no runtime cost — see
[ADR 0001](adr/0001-client-side-inference.md). Any static host works; the options below are
ordered by how well they fit this project.

**Prerequisite:** Node 20 or newer.

```bash
npm install
npm run verify     # must pass: format, lint, typecheck, 334 tests
npm run build      # output lands in packages/web/dist/
```

`npm run preview` serves the built output at <http://127.0.0.1:4173> so you can check it
before shipping.

---

## Option 1 — Cloudflare Pages (recommended)

Unmetered bandwidth is the deciding factor. The app ships several MB of models and WASM per
cold visit, and Cloudflare does not meter it. Cloudflare Pages also honours the `_headers`
file already in `packages/web/public/`.

1. Push the repository to GitHub.
2. Cloudflare dashboard → **Workers & Pages** → **Create** → **Pages** → **Connect to Git**.
3. Select the repository and set:

   | Setting                | Value                                |
   | ---------------------- | ------------------------------------ |
   | Framework preset       | None                                 |
   | Build command          | `npm run build`                      |
   | Build output directory | `packages/web/dist`                  |
   | Node version           | `20` (add env var `NODE_VERSION=20`) |

4. **Save and Deploy.** You get `https://<project>.pages.dev`, and every push to `main`
   redeploys automatically.

Leave the base path alone — Cloudflare serves from the domain root, which is the default.

**One limit to know:** Cloudflare rejects any single asset over **25 MiB** at deploy time.
Nothing today comes close, but the MediaPipe `holistic_landmarker.task` model would. If you
hit it, serve that one file from GitHub Releases and point `modelAssetPath` at the absolute
URL — MediaPipe accepts it, and no cross-origin isolation is required.

---

## Option 2 — GitHub Pages

Free and fine, with two caveats: bandwidth is a soft 100 GB/month (roughly 6,600 cold visits
at this payload), and GitHub Pages **cannot set custom headers**, so the `_headers` file is
ignored. That costs nothing today because the app is deliberately built not to need
COOP/COEP.

A **project site** is served from `https://<user>.github.io/<repo>/`, so the base path must
be set at build time.

Add `.github/workflows/deploy-pages.yml`:

```yaml
name: Deploy to GitHub Pages
on:
  push:
    branches: [main]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: pages
  cancel-in-progress: true

jobs:
  deploy:
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: npm
      - run: npm ci
      - run: npm run build
        env:
          # Must match the repository name, with both slashes.
          BASE_PATH: /mudrapragyan/
      - uses: actions/configure-pages@v5
      - uses: actions/upload-pages-artifact@v3
        with:
          path: packages/web/dist
      - id: deployment
        uses: actions/deploy-pages@v4
```

Then: **Settings → Pages → Source → GitHub Actions**.

If you use a _user_ site (`<user>.github.io`) or a custom domain at the root, drop the
`BASE_PATH` line — the default `/` is correct.

---

## Option 3 — Any other static host

Upload the contents of `packages/web/dist/` anywhere. Netlify, Vercel, S3, and Nginx all
work unchanged. Two requirements:

- **HTTPS.** Browsers refuse camera access on plain HTTP. `localhost` is exempt.
- **Correct MIME types**, in particular `application/wasm` for `.wasm`. Most hosts get this
  right; a hand-rolled Nginx config may not.

Worth checking that your host compresses `application/wasm`. It is the single biggest lever
on first-load time — 11 MB uncompressed versus about 3.3 MB gzipped.

---

## Verifying a deployment

1. Open the site over HTTPS.
2. Go to **Translator** and confirm the status line reads
   _✅ ASL Fingerspelling (simulated) v1.0.0 — 30 signs_. If it says
   _⚠ Legacy model — 27 signs_ instead, the model pack did not deploy — check that
   `models/asl-fingerspell/manifest.json` is reachable on the site.
3. Press **Start Camera** and allow the permission prompt.
4. Open DevTools → **Network** and confirm you see only static file downloads. Nothing is
   uploaded — see [PRIVACY.md](PRIVACY.md).
5. Press **D** to open the debug overlay and confirm predictions are updating.

---

## Rolling back

Both Cloudflare Pages and GitHub Pages keep previous deployments and can promote an older
one from the dashboard. Because the app is entirely static, a rollback is instant and total —
there is no migration or server state to reconcile.

---

## What this costs

Nothing, at any scale you are likely to reach.

| Item                             | Cost                                  |
| -------------------------------- | ------------------------------------- |
| Hosting (Cloudflare Pages)       | £0, unmetered bandwidth               |
| CI (GitHub Actions, public repo) | £0, unlimited standard-runner minutes |
| Inference                        | £0 — it runs on the visitor's device  |
| Training (Kaggle)                | £0, 30 GPU-hours per week             |

The one rule that keeps it free: on GitHub Actions, stay on `ubuntu-latest`. Larger runners
are billed even for public repositories.
