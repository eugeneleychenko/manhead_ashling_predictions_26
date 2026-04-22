#!/usr/bin/env bash
set -euo pipefail

# ─── Manhead: Bootstrap DO Droplet ───
# Run this ON the droplet as root after rsync-ing the repo to /opt/manhead/

APP_DIR="/opt/manhead"
APP_USER="manhead"
VENV="$APP_DIR/.venv"
MODEL_URL="https://mh-forecast.nyc3.cdn.digitaloceanspaces.com/new_model/model_retrained_3gb.joblib"
MODEL_DEST="$APP_DIR/Flask/model_retrained.joblib"

echo "==> [1/9] System packages..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq software-properties-common
add-apt-repository -y ppa:deadsnakes/ppa
apt-get update -qq
apt-get install -y -qq python3.13 python3.13-venv python3.13-dev \
    nginx certbot python3-certbot-nginx \
    curl wget build-essential libffi-dev

echo "==> [2/9] Create app user..."
if ! id "$APP_USER" &>/dev/null; then
    useradd --system --shell /bin/bash --home-dir "$APP_DIR" "$APP_USER"
fi

echo "==> [3/9] Set ownership..."
chown -R "$APP_USER":"$APP_USER" "$APP_DIR"

echo "==> [4/9] Create Python venv + install deps..."
sudo -u "$APP_USER" python3.13 -m venv "$VENV"
sudo -u "$APP_USER" "$VENV/bin/pip" install --upgrade pip
sudo -u "$APP_USER" "$VENV/bin/pip" install -r "$APP_DIR/requirements-server.txt"
sudo -u "$APP_USER" "$VENV/bin/pip" install gunicorn python-dotenv

echo "==> [5/9] Download model from DO Spaces (~2.9GB, be patient)..."
if [ ! -f "$MODEL_DEST" ]; then
    sudo -u "$APP_USER" wget -q --show-progress -O "$MODEL_DEST" "$MODEL_URL"
    echo "    Model downloaded: $(du -h "$MODEL_DEST" | cut -f1)"
else
    echo "    Model already exists: $(du -h "$MODEL_DEST" | cut -f1)"
fi

echo "==> [6/9] Switch to cloud paths config..."
if [ -f "$APP_DIR/paths_config_cloud.txt" ]; then
    cp "$APP_DIR/paths_config.txt" "$APP_DIR/paths_config_local.txt.bak"
    cp "$APP_DIR/paths_config_cloud.txt" "$APP_DIR/paths_config.txt"
    echo "    Switched to cloud paths config"
fi

echo "==> [7/9] Create required directories..."
sudo -u "$APP_USER" mkdir -p "$APP_DIR/outputs/flask_downloads"
sudo -u "$APP_USER" mkdir -p "$APP_DIR/logs"
sudo -u "$APP_USER" mkdir -p "$APP_DIR/Flask/.staging"
sudo -u "$APP_USER" mkdir -p "$APP_DIR/CSVs/archive"
sudo -u "$APP_USER" mkdir -p "$APP_DIR/debug_info"

echo "==> [8/9] Install systemd service..."
cp "$APP_DIR/systemd/manhead-flask.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable manhead-flask
systemctl start manhead-flask
echo "    Flask service started"

echo "==> [9/9] Install nginx config + generate self-signed cert..."
# Self-signed cert for now (reserved IP, no domain)
CERT_DIR="/etc/ssl/manhead"
mkdir -p "$CERT_DIR"
if [ ! -f "$CERT_DIR/selfsigned.crt" ]; then
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout "$CERT_DIR/selfsigned.key" \
        -out "$CERT_DIR/selfsigned.crt" \
        -subj "/C=US/ST=NY/L=NYC/O=Manhead/CN=manhead-predict"
    echo "    Self-signed cert generated"
fi

cp "$APP_DIR/nginx/manhead.conf" /etc/nginx/sites-available/manhead
ln -sf /etc/nginx/sites-available/manhead /etc/nginx/sites-enabled/manhead
rm -f /etc/nginx/sites-enabled/default
nginx -t && systemctl reload nginx
echo "    Nginx configured and reloaded"

# Load .env to show the API key
if [ -f "$APP_DIR/.env" ]; then
    echo ""
    echo "============================================"
    echo "  Setup complete!"
    echo "  API Key (from .env):"
    grep MANHEAD_API_KEY "$APP_DIR/.env" | head -1
    echo "============================================"
else
    echo ""
    echo "============================================"
    echo "  Setup complete!"
    echo "  WARNING: No .env file found. Create one:"
    echo "    echo 'MANHEAD_API_KEY=your-key-here' > $APP_DIR/.env"
    echo "============================================"
fi

echo ""
echo "Test: curl -k https://localhost/health"
