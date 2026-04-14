@echo off
REM setup_server.bat — Levanta el servidor local de Pokemon Showdown en Windows
REM Ejecutar desde CMD o PowerShell: setup_server.bat
REM ─────────────────────────────────────────────────────────────────

echo.
echo ============================================================
echo    VGC Bot -- Setup del servidor Pokemon Showdown
echo ============================================================
echo.

REM ── Detectar si Docker esta disponible ───────────────────────
docker --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [Opcion A] Docker detectado. Usando Docker...
    echo.
    echo Descargando imagen de Pokemon Showdown...
    docker pull ghcr.io/smogon/pokemon-showdown:latest
    echo.
    echo Iniciando servidor en localhost:8000...
    echo Deja esta ventana abierta. En otra terminal: python train.py
    echo Ctrl+C para detener el servidor.
    echo.
    docker run --rm -p 8000:8000 ghcr.io/smogon/pokemon-showdown:latest
    goto :end
)

REM ── Si no hay Docker, intentar con Node.js ──────────────────
node --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [Opcion B] Node.js detectado. Instalando Showdown manualmente...
    echo.

    if exist "pokemon-showdown" (
        echo Actualizando pokemon-showdown...
        cd pokemon-showdown
        git pull
    ) else (
        echo Clonando pokemon-showdown...
        git clone https://github.com/smogon/pokemon-showdown.git
        cd pokemon-showdown
    )

    echo Instalando dependencias...
    npm install

    echo.
    echo Iniciando servidor en localhost:8000...
    echo Deja esta ventana abierta. En otra terminal: python train.py
    echo Ctrl+C para detener el servidor.
    echo.
    node pokemon-showdown start --no-security
    goto :end
)

REM ── Ninguna opcion disponible ────────────────────────────────
echo ERROR: No se encontro Docker ni Node.js.
echo.
echo Opciones:
echo   A) Instalar Docker Desktop: https://www.docker.com/products/docker-desktop
echo   B) Instalar Node.js 18+:   https://nodejs.org
echo.
echo Despues de instalar cualquiera de los dos, volvé a correr este script.
pause

:end
