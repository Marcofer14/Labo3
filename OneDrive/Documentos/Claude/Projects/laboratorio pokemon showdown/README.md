# Laboratorio Pokémon Showdown — VGC Bot

Proyecto experimental para entrenar y evaluar agentes que juegan batallas dobles de Pokémon en un servidor local de Pokémon Showdown.

El repositorio incluye:

- un equipo fijo en `team.txt`;
- datos base de PokeAPI ya descargados en `data/raw/`;
- cálculo de daño y utilidades de stats/tipos;
- codificación de estados para aprendizaje por refuerzo;
- un entorno `VGCEnv` basado en `poke-env`;
- scripts de prueba para daño, encoding y batallas entre bots;
- entrenamiento con PPO usando `stable-baselines3`;
- configuración Docker para levantar Pokémon Showdown y correr el trainer.

---

## Estado actual del proyecto

El proyecto ya permite:

1. verificar módulos sin conectarse a Showdown;
2. levantar un servidor local de Pokémon Showdown con Docker;
3. correr batallas entre bots `greedy` y `random`;
4. iniciar entrenamiento PPO contra un oponente `random` o `greedy`;
5. guardar checkpoints y logs de TensorBoard.

Todavía no están implementados completamente:

- descarga selectiva con `--only pokemon`, `--only moves`, etc.;
- imitation learning desde replays;
- ladder real contra humanos;
- team building automático.

---

## Requisitos

### Opción recomendada en Windows 11

- Docker Desktop instalado y abierto.
- PowerShell o CMD.
- No hace falta instalar Python localmente si se usa Docker.

### Opción local con Python

- Python 3.10 o superior.
- `pip`.
- Servidor Pokémon Showdown corriendo en `localhost:8000`.

---

## Formato de batalla

El formato probado correctamente fue:

```text
gen9vgc2026regi
```

El ZIP original del proyecto tenía hardcodeado un formato viejo:

```text
gen9vgc2025regg
```

Ese formato puede fallar con:

```text
Unrecognized format "gen9vgc2025regg"
```

Si todavía aparece ese valor en el código, cambiarlo por:

```text
gen9vgc2026regi
```

en estos archivos:

- `battle.py`
- `train.py`

En PowerShell se puede hacer con:

```powershell
(Get-Content .\battle.py) -replace 'gen9vgc2025regg', 'gen9vgc2026regi' | Set-Content .\battle.py
(Get-Content .\train.py) -replace 'gen9vgc2025regg', 'gen9vgc2026regi' | Set-Content .\train.py
```

Luego los comandos pueden correrse sin pasar `--format`.

---

## Uso rápido en Windows 11 con Docker Desktop

Apagar contenedores anteriores, si existen:

```powershell
docker compose down
```

Construir las imágenes:

```powershell
docker compose build
```

Levantar el servidor local de Pokémon Showdown:

```powershell
docker compose up -d showdown
```

Verificar que los contenedores estén corriendo:

```powershell
docker compose ps
```

Verificar el proyecto sin conectar a una batalla real:

```powershell
docker compose run --rm trainer python train.py --dry-run
```

Correr una prueba de batallas entre bots:

```powershell
docker compose run --rm trainer python battle.py --n 3 --p1 greedy --p2 random --server showdown:8000
```

Resultado esperado aproximado:

```text
RESULTADOS
GREEDY    victorias: 3 / 3
RANDOM    victorias: 0 / 3
```

Para apagar todo:

```powershell
docker compose down
```

---

## Comandos útiles con Docker

### Construir todo

```powershell
docker compose build
```

### Levantar Showdown en segundo plano

```powershell
docker compose up -d showdown
```

### Ver logs del servidor Showdown

```powershell
docker compose logs -f showdown
```

### Ver contenedores activos

```powershell
docker compose ps
```

### Dry-run del trainer

```powershell
docker compose run --rm trainer python train.py --dry-run
```

### Batalla greedy vs random

```powershell
docker compose run --rm trainer python battle.py --n 3 --p1 greedy --p2 random --server showdown:8000
```

### Batalla random vs random

```powershell
docker compose run --rm trainer python battle.py --n 3 --p1 random --p2 random --server showdown:8000
```

### Forzar un formato manualmente

```powershell
docker compose run --rm trainer python battle.py --n 3 --p1 greedy --p2 random --server showdown:8000 --format gen9vgc2026regi
```

### Apagar contenedores

```powershell
docker compose down
```

---

## Entrenamiento PPO

El entrenamiento principal está en:

```text
train.py
```

Para entrenar desde cero usando Docker:

```powershell
docker compose run --rm trainer python train.py --server showdown:8000
```

Por defecto el oponente es `random`.

Para entrenar contra el bot greedy:

```powershell
docker compose run --rm trainer python train.py --server showdown:8000 --opponent greedy
```

Para continuar desde un checkpoint:

```powershell
docker compose run --rm trainer python train.py --server showdown:8000 --resume checkpoints/vgc_ppo_100000.zip
```

Los checkpoints se guardan en:

```text
checkpoints/
```

Los logs de entrenamiento se guardan en:

```text
logs/
```

---

## TensorBoard

Para ver métricas del entrenamiento:

```powershell
docker compose --profile monitoring up -d tensorboard
```

Luego abrir en el navegador:

```text
http://localhost:6006
```

Para apagar TensorBoard:

```powershell
docker compose down
```

---

## Scripts de prueba

### Test del cálculo de daño

```powershell
docker compose run --rm trainer python scripts/test_damage.py
```

Este script prueba el módulo `src/damage_calc.py`.

### Test del encoding de estado

```powershell
docker compose run --rm trainer python scripts/test_state_encoding.py
```

Este script prueba el módulo `src/state_encoder.py`.

### Dry-run general

```powershell
docker compose run --rm trainer python train.py --dry-run
```

El dry-run verifica:

- carga de datos desde `data/raw/`;
- parseo de `team.txt`;
- cálculo de stats;
- cálculo de daño;
- shape del vector de observación;
- importación de `VGCEnv`;
- disponibilidad de `stable-baselines3`.

---

## Estructura real del repositorio

```text
laboratorio pokemon showdown/
├── README.md
├── GUIDE.md
├── FETCH_DATA_README.md
├── proyecto.html
├── team.txt
├── requirements.txt
├── setup.sh
├── setup_server.bat
├── setup_server.ps1
├── setup_server.sh
├── Dockerfile
├── Dockerfile.showdown
├── docker-compose.yml
├── battle.py
├── train.py
├── list_formats.py
├── data/
│   ├── get_data.py
│   └── raw/
│       ├── abilities.json
│       ├── items.json
│       ├── moves.json
│       ├── natures.json
│       ├── pokemon.json
│       └── type_chart.json
├── scripts/
│   ├── test_damage.py
│   └── test_state_encoding.py
└── src/
    ├── damage_calc.py
    ├── state_encoder.py
    ├── utils.py
    └── vgc_env.py
```

---

## Archivos principales

### `team.txt`

Equipo fijo usado por el bot, en formato estilo Pokepaste.

El equipo actual incluye:

- Kyogre
- Calyrex-Shadow
- Incineroar
- Rillaboom
- Urshifu-Rapid-Strike
- Roaring Moon

### `battle.py`

Corre batallas de prueba entre dos bots simples:

- `random`: elige acciones aleatorias;
- `greedy`: usa `MaxBasePowerPlayer`, que prioriza movimientos de alta potencia.

Ejemplo:

```powershell
docker compose run --rm trainer python battle.py --n 3 --p1 greedy --p2 random --server showdown:8000
```

### `train.py`

Script principal de entrenamiento.

Incluye:

- `--dry-run` para verificar módulos;
- `--resume` para continuar desde checkpoint;
- `--opponent random` o `--opponent greedy`;
- `--server` para indicar el servidor Showdown.

Ejemplo:

```powershell
docker compose run --rm trainer python train.py --server showdown:8000 --opponent random
```

### `src/utils.py`

Carga datos JSON, parsea `team.txt`, calcula stats reales a nivel 50 y provee helpers de tipos, STAB, items, habilidades y movimientos.

### `src/damage_calc.py`

Implementa cálculo de daño aproximado usando stats, tipos, STAB, clima, objetos, habilidades y condiciones de batalla.

### `src/state_encoder.py`

Convierte el estado de batalla en un vector numérico. En el dry-run actual se verifica un vector de tamaño:

```text
854
```

### `src/vgc_env.py`

Define `VGCEnv`, un entorno de dobles Gen 9 basado en `poke-env`.

Implementa:

- `calc_reward(battle)`;
- `embed_battle(battle)`;
- espacios de observación;
- reward por daño, KOs y victoria/derrota.

---

## Bots disponibles para pruebas

| Bot | Descripción |
|---|---|
| `random` | Selecciona acciones aleatorias. Sirve como baseline mínimo. |
| `greedy` | Usa `MaxBasePowerPlayer`. Tiende a elegir ataques de mayor potencia. |

Ejemplos:

```powershell
docker compose run --rm trainer python battle.py --n 5 --p1 greedy --p2 random --server showdown:8000
```

```powershell
docker compose run --rm trainer python battle.py --n 5 --p1 random --p2 random --server showdown:8000
```

---

## Flujo recomendado de trabajo

Para una sesión normal de desarrollo en Windows:

```powershell
cd "C:\Users\rodri\Desktop\R\UCA\2026 1ro\laboratorio pokemon showdown"

docker compose up -d showdown

docker compose run --rm trainer python train.py --dry-run

docker compose run --rm trainer python battle.py --n 3 --p1 greedy --p2 random --server showdown:8000
```

Si eso funciona, iniciar entrenamiento:

```powershell
docker compose run --rm trainer python train.py --server showdown:8000 --opponent random
```

Para cerrar:

```powershell
docker compose down
```

---

## Objetivo experimental

El objetivo del proyecto es construir un agente capaz de jugar batallas dobles VGC usando aprendizaje por refuerzo.

La versión actual todavía está en fase de infraestructura y prototipo:

1. cargar datos y equipo;
2. representar estados;
3. calcular rewards;
4. conectar con Pokémon Showdown;
5. correr baselines simples;
6. entrenar PPO contra bots básicos.

A partir de ahí se puede avanzar hacia:

- evaluación sistemática contra baselines;
- ajuste de reward shaping;
- entrenamiento por currículum;
- self-play;
- imitation learning desde replays;
- análisis de decisiones del agente.
