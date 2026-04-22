#!/usr/bin/env bash
set -euo pipefail

# ─── Manhead: Provision DigitalOcean Droplet ───
# Run this from your Mac. Requires: doctl, ssh-key already in DO.

DROPLET_NAME="manhead-predict"
REGION="nyc3"
SIZE="s-4vcpu-32gb-320gb-intel"
IMAGE="ubuntu-24-04-x64"
SSH_KEY_NAME="Macbook Air"

echo "==> Looking up SSH key ID..."
SSH_KEY_ID=$(doctl compute ssh-key list --format ID,Name --no-header | grep "$SSH_KEY_NAME" | awk '{print $1}')
if [ -z "$SSH_KEY_ID" ]; then
    echo "ERROR: SSH key '$SSH_KEY_NAME' not found in DigitalOcean."
    exit 1
fi
echo "    SSH Key ID: $SSH_KEY_ID"

echo "==> Creating droplet '$DROPLET_NAME'..."
doctl compute droplet create "$DROPLET_NAME" \
    --region "$REGION" \
    --size "$SIZE" \
    --image "$IMAGE" \
    --ssh-keys "$SSH_KEY_ID" \
    --tag-name "manhead" \
    --wait

echo "==> Waiting for droplet IP..."
sleep 5
DROPLET_IP=$(doctl compute droplet list --format Name,PublicIPv4 --no-header | grep "$DROPLET_NAME" | awk '{print $2}')
echo "    Droplet IP: $DROPLET_IP"

echo "==> Reserving a static IP and assigning to droplet..."
DROPLET_ID=$(doctl compute droplet list --format Name,ID --no-header | grep "$DROPLET_NAME" | awk '{print $2}')
RESERVED_IP=$(doctl compute reserved-ip create --region "$REGION" --format IP --no-header)
echo "    Reserved IP: $RESERVED_IP"
doctl compute reserved-ip-action assign "$RESERVED_IP" "$DROPLET_ID"

echo ""
echo "============================================"
echo "  Droplet created!"
echo "  Droplet IP:   $DROPLET_IP"
echo "  Reserved IP:  $RESERVED_IP"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1) ssh root@$RESERVED_IP"
echo "  2) Upload this repo:  rsync -avz --exclude='Flask/model_retrained.joblib' --exclude='.venv' --exclude='__pycache__' ./ root@$RESERVED_IP:/opt/manhead/"
echo "  3) Run setup:         ssh root@$RESERVED_IP 'bash /opt/manhead/scripts/setup-server.sh'"
