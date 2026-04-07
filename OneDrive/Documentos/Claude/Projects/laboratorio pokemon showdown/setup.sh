#!/bin/bash
# ─────────────────────────────────────────────────────────────────
# setup.sh — Configura el entorno de Python para el proyecto VGC bot
# Uso: bash setup.sh
# ─────────────────────────────────────────────────────────────────

set -e  # salir si cualquier comando falla

echo "╔══════════════════════════════════════════╗"
echo "║   VGC Bot — Setup del entorno Python     ║"
echo "╚══════════════════════════════════════════╝"

# 1. Verificar Python 3.10+
PYTHON=$(command -v python3 || command -v python)
PY_VERSION=$($PYTHON --version 2>&1 | awk '{print $2}')
echo ""
echo "→ Python detectado: $PY_VERSION"

# 2. Crear entorno virtual
echo "→ Creando entorno virtual en ./venv ..."
$PYTHON -m venv venv

# 3. Activar entorno
source venv/bin/activate
echo "→ Entorno activado."

# 4. Actualizar pip
pip install --upgrade pip --quiet

# 5. Instalar dependencias
echo "→ Instalando dependencias desde requirements.txt ..."
pip install -r requirements.txt

echo ""
echo "✓ Setup completo."
echo ""
echo "Para activar el entorno en el futuro:"
echo "  source venv/bin/activate"
echo ""
echo "Para descargar los datos de PokeAPI:"
echo "  python data/fetch_data.py"
