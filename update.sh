#!/bin/bash

echo "🚀 Pulling latest code..."
git pull

echo "📦 Rebuilding and restarting containers..."
docker compose -f docker-compose.prod.yml up -d --build --remove-orphans

echo "🧹 Pruning unused images..."
docker image prune -f

echo "✅ Update Complete!"