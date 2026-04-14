# setup_server.ps1 — Levanta el servidor local de Pokemon Showdown
# Ejecutar desde PowerShell:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   .\setup_server.ps1
# ─────────────────────────────────────────────────────────────────

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "   VGC Bot -- Setup del servidor Pokemon Showdown" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# ── Detectar Docker ─────────────────────────────────────────────
$dockerAvailable = $null -ne (Get-Command docker -ErrorAction SilentlyContinue)

if ($dockerAvailable) {
    Write-Host "[Opcion A] " -ForegroundColor Green -NoNewline
    Write-Host "Docker detectado. Usando Docker..."
    Write-Host ""
    Write-Host "Descargando imagen de Pokemon Showdown..."
    docker pull ghcr.io/smogon/pokemon-showdown:latest
    Write-Host ""
    Write-Host "Iniciando servidor en localhost:8000..." -ForegroundColor Yellow
    Write-Host "Deja esta ventana abierta. En otra terminal: " -NoNewline
    Write-Host "python train.py" -ForegroundColor Cyan
    Write-Host "Ctrl+C para detener el servidor."
    Write-Host ""
    docker run --rm -p 8000:8000 ghcr.io/smogon/pokemon-showdown:latest
    exit
}

# ── Detectar Node.js ────────────────────────────────────────────
$nodeAvailable = $null -ne (Get-Command node -ErrorAction SilentlyContinue)

if ($nodeAvailable) {
    $nodeVersion = node --version
    Write-Host "[Opcion B] " -ForegroundColor Green -NoNewline
    Write-Host "Node.js $nodeVersion detectado."
    Write-Host ""

    if (Test-Path "pokemon-showdown") {
        Write-Host "Actualizando pokemon-showdown..."
        Set-Location "pokemon-showdown"
        git pull
    } else {
        Write-Host "Clonando pokemon-showdown..."
        git clone https://github.com/smogon/pokemon-showdown.git
        Set-Location "pokemon-showdown"
    }

    Write-Host "Instalando dependencias de Node..."
    npm install

    Write-Host ""
    Write-Host "Iniciando servidor en localhost:8000..." -ForegroundColor Yellow
    Write-Host "Deja esta ventana abierta. En otra terminal: " -NoNewline
    Write-Host "python train.py" -ForegroundColor Cyan
    Write-Host "Ctrl+C para detener el servidor."
    Write-Host ""
    node pokemon-showdown start --no-security
    exit
}

# ── Ninguna opcion disponible ───────────────────────────────────
Write-Host "ERROR: No se encontro Docker ni Node.js." -ForegroundColor Red
Write-Host ""
Write-Host "Opciones para instalar:"
Write-Host "  A) Docker Desktop: https://www.docker.com/products/docker-desktop" -ForegroundColor Cyan
Write-Host "  B) Node.js 18+:   https://nodejs.org" -ForegroundColor Cyan
Write-Host ""
Write-Host "Despues de instalar cualquiera de los dos, volvé a correr:"
Write-Host "  .\setup_server.ps1" -ForegroundColor Yellow
Read-Host "Presiona Enter para salir"
