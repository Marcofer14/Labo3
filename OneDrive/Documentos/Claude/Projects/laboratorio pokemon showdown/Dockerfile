# ─────────────────────────────────────────────────────────────────
# Dockerfile — Entorno Python del VGC Bot
# ─────────────────────────────────────────────────────────────────
# Imagen base: Python 3.11 slim (liviana, sin GUI)
FROM python:3.11-slim

# Metadata
LABEL description="VGC Bot — Agente RL para Pokemon Showdown"

# Evitar prompts interactivos durante apt
ENV DEBIAN_FRONTEND=noninteractive

# ── Dependencias del sistema ──────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Directorio de trabajo ─────────────────────────────────────────
WORKDIR /app

# ── Instalar dependencias Python ──────────────────────────────────
# Se copia requirements.txt primero para aprovechar la cache de Docker:
# si los requirements no cambian, no reinstala en cada build.
COPY requirements.txt .
# --trusted-host: evita errores de SSL en redes con proxy/firewall corporativo
RUN pip install --no-cache-dir \
        --trusted-host pypi.org \
        --trusted-host files.pythonhosted.org \
        -r requirements.txt

# ── Copiar el código del proyecto ─────────────────────────────────
COPY . .

# ── Variable de entorno: URL del servidor Showdown ────────────────
# En docker-compose se sobreescribe a "showdown:8000"
# En local se deja en "localhost:8000"
ENV SHOWDOWN_SERVER=showdown:8000

# ── Entry point por defecto ───────────────────────────────────────
# Se puede sobreescribir con `docker run ... python train.py --dry-run`
CMD ["python", "train.py"]
