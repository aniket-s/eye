#!/usr/bin/env bash
#
# Deploy MudraPragyan.AI to an Ubuntu server running Apache, with Let's Encrypt TLS.
#
# Built for a plain DigitalOcean droplet (Ubuntu 22.04 / 24.04), but nothing here is
# DigitalOcean-specific. The app is a static site, so this script's whole job is:
# get the source, build it honestly, serve it with the right headers, over HTTPS —
# the camera API refuses to run without it.
#
# First run (as root):
#
#   EMAIL=you@example.com bash deploy-apache.sh
#
# Every run after that is an update deploy — pull, rebuild, atomic swap:
#
#   bash deploy-apache.sh
#
# Configuration, all overridable via environment:
#
#   DOMAIN   The site's hostname.            default: eye.mozget.com
#   EMAIL    Let's Encrypt account email.    required once, for certificate issuance
#   REPO_URL Git remote to deploy from.      default: https://github.com/aniket-s/eye.git
#   BRANCH   Branch to deploy.               default: main
#
# Private repository? Either embed a fine-grained PAT in REPO_URL
# (https://<token>@github.com/aniket-s/eye.git) or add a read-only deploy key to the
# server and use the SSH URL (git@github.com:aniket-s/eye.git).
#
# Prerequisites this script cannot do for you:
#   - A DNS A record for $DOMAIN pointing at this server (checked before anything runs;
#     certificate issuance validates over it).
#   - Ports 80 and 443 open in any cloud firewall in front of the droplet.
#
# What it deliberately does NOT do: enable ufw (enabling a firewall from a script is
# how people lock themselves out of SSH — if ufw is already active, Apache is allowed
# through), and it never rewrites an existing vhost (certbot's edits must survive).
#
# Rollback: previous releases are kept in /var/www/eye/releases/. Point the `current`
# symlink at an older one — the swap is the deploy, so that is the whole rollback:
#   ln -sfn /var/www/eye/releases/<older> /var/www/eye/current

set -euo pipefail

DOMAIN="${DOMAIN:-eye.mozget.com}"
EMAIL="${EMAIL:-}"
REPO_URL="${REPO_URL:-https://github.com/aniket-s/eye.git}"
BRANCH="${BRANCH:-main}"
SRC_DIR="${SRC_DIR:-/opt/eye/src}"
APP_DIR="${APP_DIR:-/var/www/eye}"
KEEP_RELEASES="${KEEP_RELEASES:-3}"
LOG_FILE="${LOG_FILE:-/var/log/eye-deploy.log}"

log() { printf '\n==> %s\n' "$*"; }
warn() { printf '  ! %s\n' "$*" >&2; }
die() {
  printf '\nDEPLOY FAILED: %s\n' "$*" >&2
  exit 1
}
trap 'printf "\nDEPLOY FAILED at line %s. Full log: %s\n" "$LINENO" "$LOG_FILE" >&2' ERR

# Everything below is logged. The site keeps serving the old release the whole time —
# nothing live is touched until the atomic swap near the end.
mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1
log "Deploy started $(date -u +'%Y-%m-%dT%H:%M:%SZ') — $DOMAIN from $BRANCH"

# ---------------------------------------------------------------- preflight
[ "$(id -u)" -eq 0 ] || die "run as root (sudo bash deploy-apache.sh)"

if [ -r /etc/os-release ]; then
  # shellcheck source=/dev/null
  . /etc/os-release
  case "${ID:-}" in
    ubuntu | debian) ;;
    *) warn "built for Ubuntu; detected ${PRETTY_NAME:-unknown}. Continuing anyway." ;;
  esac
fi

# DNS must already point here, or certificate issuance fails after a long build.
# Cheap to check first, and the failure message is better than certbot's.
log "Checking DNS for $DOMAIN"
SERVER_IP="$(curl -4fsS --max-time 10 https://api.ipify.org || hostname -I | awk '{print $1}')"
DOMAIN_IP="$(getent ahostsv4 "$DOMAIN" 2>/dev/null | awk '{print $1; exit}' || true)"
if [ -z "$DOMAIN_IP" ]; then
  [ "${SKIP_DNS_CHECK:-0}" = "1" ] || die \
    "$DOMAIN does not resolve. Add an A record for it pointing at $SERVER_IP, wait for it to propagate, then re-run. (Override with SKIP_DNS_CHECK=1.)"
  warn "$DOMAIN does not resolve; continuing because SKIP_DNS_CHECK=1."
elif [ "$DOMAIN_IP" != "$SERVER_IP" ]; then
  if [ -e "/etc/letsencrypt/live/$DOMAIN/fullchain.pem" ] || [ "${SKIP_DNS_CHECK:-0}" = "1" ]; then
    warn "$DOMAIN resolves to $DOMAIN_IP but this server looks like $SERVER_IP. Continuing."
  else
    die "$DOMAIN resolves to $DOMAIN_IP, but this server looks like $SERVER_IP. Fix the A record, or re-run with SKIP_DNS_CHECK=1 if the detection is wrong (e.g. a floating IP)."
  fi
else
  echo "  ✓ $DOMAIN -> $SERVER_IP"
fi

# ---------------------------------------------------------------- packages
log "Installing packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq apache2 git rsync curl ca-certificates certbot python3-certbot-apache

# Node 20+ is a build requirement (README). Ubuntu's own nodejs is older, so use
# NodeSource — but only when needed, so re-runs don't touch a working toolchain.
NODE_MAJOR="$(command -v node >/dev/null 2>&1 && node -p 'process.versions.node.split(".")[0]' || echo 0)"
if [ "$NODE_MAJOR" -lt 20 ]; then
  log "Installing Node 20 (found major version: $NODE_MAJOR)"
  curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
  apt-get install -y -qq nodejs
fi
echo "  ✓ node $(node -v), npm $(npm -v)"

# npm ci + vite build can OOM a 1 GB droplet. Swap is slower than RAM but far
# better than a killed build; skipped whenever the machine has 2 GB or any swap.
MEM_KB="$(awk '/MemTotal/ {print $2}' /proc/meminfo)"
if [ "$MEM_KB" -lt 2000000 ] && [ -z "$(swapon --noheadings --show 2>/dev/null)" ]; then
  log "Adding a 2G swapfile (RAM is $((MEM_KB / 1024)) MB)"
  fallocate -l 2G /swapfile 2>/dev/null || dd if=/dev/zero of=/swapfile bs=1M count=2048 status=none
  chmod 600 /swapfile
  mkswap -q /swapfile
  swapon /swapfile
  grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >>/etc/fstab
fi

# ---------------------------------------------------------------- source
log "Fetching source ($BRANCH)"
if [ -d "$SRC_DIR/.git" ]; then
  git -C "$SRC_DIR" remote set-url origin "$REPO_URL"
  git -C "$SRC_DIR" fetch origin "$BRANCH"
  git -C "$SRC_DIR" checkout -q "$BRANCH" 2>/dev/null || git -C "$SRC_DIR" checkout -qb "$BRANCH" "origin/$BRANCH"
  git -C "$SRC_DIR" reset --hard "origin/$BRANCH"
else
  mkdir -p "$(dirname "$SRC_DIR")"
  git clone --branch "$BRANCH" "$REPO_URL" "$SRC_DIR"
fi
echo "  ✓ at $(git -C "$SRC_DIR" rev-parse --short HEAD): $(git -C "$SRC_DIR" log -1 --format=%s)"

# ---------------------------------------------------------------- build
cd "$SRC_DIR"

log "Installing dependencies"
npm ci --no-audit --no-fund

# Strict, unlike `npm run build`'s --allow-missing: a release must never ship without
# the MediaPipe models, or the site deploys fine and the camera silently never starts.
# Downloads are checksum-verified against models.lock.json.
log "Fetching MediaPipe assets (strict)"
npm run setup:assets

log "Building"
# BASE_PATH is for GitHub Pages project sites; this server serves from the domain
# root, and a stray value would break every asset URL.
env -u BASE_PATH npm run build

DIST="$SRC_DIR/packages/web/dist"
log "Verifying build output"
for required in \
  index.html \
  recorder.html \
  sw.js \
  model_weights.json \
  mediapipe/wasm/vision_wasm_internal.wasm \
  models/hand_landmarker.task \
  models/asl-fingerspell/manifest.json \
  models/asl-fingerspell/model.onnx; do
  [ -s "$DIST/$required" ] || die "build output is missing $required — refusing to deploy it"
  echo "  ✓ $required"
done

# ---------------------------------------------------------------- release
# Capistrano-style: rsync into a timestamped directory, then swap one symlink.
# The live site is never mid-copy, and the previous release stays for rollback.
STAMP="$(date +%Y%m%d%H%M%S)"
RELEASE_DIR="$APP_DIR/releases/$STAMP"
log "Publishing release $STAMP"
mkdir -p "$RELEASE_DIR"
rsync -a --delete "$DIST/" "$RELEASE_DIR/"
chmod -R a+rX "$RELEASE_DIR"

# `current` must be a symlink for the atomic swap; a real directory there (manual
# meddling, a stray mkdir) would make `mv -T` fail and `ln -sfn` nest silently.
# Safe to remove: every release lives under releases/, so nothing unique is lost.
if [ -e "$APP_DIR/current" ] && [ ! -L "$APP_DIR/current" ]; then
  warn "replacing non-symlink $APP_DIR/current"
  rm -rf "$APP_DIR/current"
fi
ln -sfn "$RELEASE_DIR" "$APP_DIR/current.new.$$"
mv -T "$APP_DIR/current.new.$$" "$APP_DIR/current"
echo "  ✓ $APP_DIR/current -> releases/$STAMP"

# Prune old releases, newest first, keeping the last KEEP_RELEASES.
find "$APP_DIR/releases" -mindepth 1 -maxdepth 1 -type d | sort -r |
  tail -n +$((KEEP_RELEASES + 1)) | xargs -r rm -rf

# ---------------------------------------------------------------- apache
# The vhost is written ONCE and never rewritten: certbot edits it when installing
# the certificate and the HTTPS redirect, and a rewrite would silently undo that.
VHOST="/etc/apache2/sites-available/$DOMAIN.conf"
if [ ! -f "$VHOST" ]; then
  log "Writing Apache vhost $VHOST"
  cat >"$VHOST" <<APACHE
# MudraPragyan.AI — static site. Generated by deploy/deploy-apache.sh; edits survive
# redeploys (the script never rewrites this file).
#
# The headers below mirror packages/web/public/_headers, which Apache ignores.
# COOP/COEP are deliberately NOT set (see that file and ADR 0002).
<VirtualHost *:80>
  ServerName $DOMAIN
  DocumentRoot $APP_DIR/current

  ErrorLog \${APACHE_LOG_DIR}/$DOMAIN-error.log
  CustomLog \${APACHE_LOG_DIR}/$DOMAIN-access.log combined

  <Directory $APP_DIR/>
    Options -Indexes +FollowSymLinks
    AllowOverride None
    Require all granted
  </Directory>

  # Without the wasm type the browser refuses to compile the MediaPipe runtime and
  # the camera never starts. The rest keep strict MIME checking honest.
  AddType application/wasm .wasm
  AddType application/manifest+json .webmanifest
  AddType application/octet-stream .task .onnx

  # The single biggest first-load lever: ~11 MB of WASM gzips to about a third.
  AddOutputFilterByType DEFLATE text/html text/css text/plain application/javascript application/json application/wasm image/svg+xml application/manifest+json

  Header always set X-Content-Type-Options "nosniff"
  Header always set Referrer-Policy "no-referrer"
  Header always set X-Frame-Options "DENY"
  # This app needs the camera and nothing else.
  Header always set Permissions-Policy "camera=(self), microphone=(), geolocation=(), interest-cohort=()"

  # Caching mirrors the service worker's rules (public/sw.js) — content-addressed or
  # checksum-pinned files are immutable, everything a redeploy can change revalidates.
  <LocationMatch "^/(assets|mediapipe|alphabets|numbers)/">
    Header set Cache-Control "public, max-age=31536000, immutable"
  </LocationMatch>
  <FilesMatch "\.(wasm|task|onnx|woff2?)$">
    Header set Cache-Control "public, max-age=31536000, immutable"
  </FilesMatch>
  # A pack manifest sits at a stable URL across retrains; caching it would pin users
  # to the first model they ever downloaded, with no way to tell.
  <LocationMatch "^/models/.*manifest\.json$">
    Header set Cache-Control "no-cache"
  </LocationMatch>
  <LocationMatch "^/(index\.html|recorder\.html|sw\.js|manifest\.webmanifest)?$">
    Header set Cache-Control "no-cache"
  </LocationMatch>
  <Files "model_weights.json">
    Header set Cache-Control "public, max-age=3600, must-revalidate"
  </Files>
</VirtualHost>
APACHE
else
  log "Vhost already exists — leaving it alone (certbot may have edited it)"
fi

a2enmod -q headers deflate rewrite ssl >/dev/null
a2ensite -q "$DOMAIN" >/dev/null
# The stock placeholder site would answer for the bare IP and shadow nothing we need.
[ -e /etc/apache2/sites-enabled/000-default.conf ] && a2dissite -q 000-default >/dev/null

apache2ctl configtest
systemctl reload apache2
echo "  ✓ apache reloaded"

# ---------------------------------------------------------------- tls
if [ ! -e "/etc/letsencrypt/live/$DOMAIN/fullchain.pem" ]; then
  [ -n "$EMAIL" ] || die "no certificate yet and EMAIL is unset. Re-run as: EMAIL=you@example.com bash deploy-apache.sh"
  log "Obtaining Let's Encrypt certificate"
  certbot --apache -d "$DOMAIN" --redirect --non-interactive --agree-tos -m "$EMAIL"
  # The certbot package installs a systemd renewal timer; prove it will work now,
  # not in ninety days.
  certbot renew --dry-run -q || warn "renewal dry-run failed — investigate 'certbot renew --dry-run' before the cert expires"
else
  log "Certificate already installed — renewal is certbot.timer's job"
fi

# Never enable ufw from a script; only step aside if it is already on.
if command -v ufw >/dev/null 2>&1 && ufw status 2>/dev/null | grep -q '^Status: active'; then
  ufw allow 'Apache Full' >/dev/null
  echo "  ✓ ufw: Apache Full allowed"
fi

# ---------------------------------------------------------------- health
log "Health checks against https://$DOMAIN"
curl -fsS "https://$DOMAIN/" | grep -q 'MudraPragyan' || die "homepage did not serve the app"
echo "  ✓ homepage serves the app"
curl -fsS "https://$DOMAIN/models/asl-fingerspell/manifest.json" | grep -q '"labels"' ||
  die "model pack manifest is not being served — the Translator would fall back to the legacy model"
echo "  ✓ model pack manifest served"
WASM_HEADERS="$(curl -fsSI "https://$DOMAIN/mediapipe/wasm/vision_wasm_internal.wasm")"
echo "$WASM_HEADERS" | grep -qi '^content-type: application/wasm' ||
  die "WASM served with the wrong MIME type — the camera will not start"
echo "  ✓ WASM served as application/wasm"
if curl -fsS -o /dev/null -D - -H 'Accept-Encoding: gzip' \
  "https://$DOMAIN/mediapipe/wasm/vision_wasm_internal.wasm" |
  grep -qi '^content-encoding: gzip'; then
  echo "  ✓ WASM compressed in transit"
else
  warn "WASM not compressed — first loads will be ~3x slower, check mod_deflate"
fi

log "Deployed. Final check is a human one:"
cat <<CHECK
  Open https://$DOMAIN, go to Translator, and confirm the status line reads
      ✅ ASL Fingerspelling (simulated) v1.0.0 — 30 signs
  If it says "⚠ Legacy model — 27 signs" the pack did not deploy.
  Then Start Camera (allow the prompt), press D, and watch predictions update.
CHECK
