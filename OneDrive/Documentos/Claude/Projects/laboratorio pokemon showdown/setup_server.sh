#!/usr/bin/env bash
# setup_server.sh
# ─────────────────────────────────────────────────────────────────
# Setup del servidor local de Pokémon Showdown para entrenamiento.
#
# OPCIÓN A (más fácil): Docker
#   bash setup_server.sh docker
#
# OPCIÓN B: Instalación manual con Node.js
#   bash setup_server.sh manual
#
# Una vez corriendo, el servidor escucha en localhost:8000
# Luego en otra terminal: python train.py
# ─────────────────────────────────────────────────────────────────

set -e

MODE=${1:-docker}

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║       VGC Bot — Setup del servidor Pokémon Showdown     ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

if [ "$MODE" = "docker" ]; then
    echo "── Modo Docker ──────────────────────────────────────────────"
    echo ""

    # Verificar que Docker esté instalado
    if ! command -v docker &> /dev/null; then
        echo "❌ Docker no está instalado."
        echo "   Descargarlo desde: https://www.docker.com/products/docker-desktop"
        exit 1
    fi

    echo "✓ Docker detectado."
    echo ""
    echo "Descargando imagen de Pokémon Showdown..."
    docker pull ghcr.io/smogon/pokemon-showdown:latest

    echo ""
    echo "Iniciando servidor en localhost:8000..."
    echo "(Ctrl+C para detener)"
    echo ""
    docker run --rm -p 8000:8000 ghcr.io/smogon/pokemon-showdown:latest

elif [ "$MODE" = "manual" ]; then
    echo "── Modo manual (Node.js) ────────────────────────────────────"
    echo ""

    # Verificar Node.js
    if ! command -v node &> /dev/null; then
        echo "❌ Node.js no está instalado."
        echo "   Descargarlo desde: https://nodejs.org (versión 18+)"
        exit 1
    fi

    NODE_VER=$(node --version)
    echo "✓ Node.js detectado: $NODE_VER"

    # Clonar o actualizar el repo
    if [ -d "pokemon-showdown" ]; then
        echo "Actualizando pokemon-showdown..."
        cd pokemon-showdown
        git pull
    else
        echo "Clonando pokemon-showdown..."
        git clone https://github.com/smogon/pokemon-showdown.git
        cd pokemon-showdown
    fi

    echo "Instalando dependencias de Node.js..."
    npm install

    echo ""
    echo "Iniciando servidor en localhost:8000..."
    echo "(Ctrl+C para detener)"
    echo ""
    node pokemon-showdown start --no-security

else
    echo "Uso: bash setup_server.sh [docker|manual]"
    echo ""
    echo "  docker  — usa la imagen Docker oficial (recomendado)"
    echo "  manual  — instala y corre con Node.js"
    exit 1
fi
