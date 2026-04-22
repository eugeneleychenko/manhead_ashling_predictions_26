#!/usr/bin/env bash
set -euo pipefail

# ─── Manhead: Deploy to DO Droplet ───
# Usage: ./scripts/deploy.sh <RESERVED_IP>
#
# This script:
#   1. Rsyncs the repo to the droplet (excluding large model + local artifacts)
#   2. Runs the server setup script on the droplet
#
# Prerequisites:
#   - Droplet already created (run provision-droplet.sh first)
#   - SSH access as root to the droplet

if [ $# -lt 1 ]; then
    echo "Usage: $0 <DROPLET_IP>"
    echo "Example: $0 164.90.xxx.xxx"
    exit 1
fi

DROPLET_IP="$1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

echo "==> Syncing repo to root@$DROPLET_IP:/opt/manhead/ ..."
echo "    (excluding model .joblib, .venv, __pycache__)"

rsync -avz --progress \
    --exclude='Flask/model_retrained.joblib' \
    --exclude='.venv' \
    --exclude='__pycache__' \
    --exclude='.git' \
    --exclude='*.pyc' \
    "$REPO_DIR/" "root@$DROPLET_IP:/opt/manhead/"

echo ""
echo "==> Running setup script on droplet..."
ssh "root@$DROPLET_IP" "bash /opt/manhead/scripts/setup-server.sh"

echo ""
echo "==> Verifying health..."
sleep 3
ssh "root@$DROPLET_IP" "curl -sk https://localhost/health" || echo "(health check may need a moment)"

echo ""
echo "============================================"
echo "  Deploy complete!"
echo "  Test: curl -k https://$DROPLET_IP/health"
echo "============================================"
