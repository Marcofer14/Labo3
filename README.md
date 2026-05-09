# Laboratorio Pokemon Showdown - VGC Bot

Proyecto experimental para conectar bots a Pokemon Showdown, probar politicas simples y avanzar hacia agentes de aprendizaje por refuerzo.

El flujo actual separa dos responsabilidades:

- `login.py`: crea bots autenticados y configura el servidor.
- `play.py`: decide que politica usa cada bot, ejecuta partidas y cierra las sesiones.

`battle.py` se mantiene como smoke test legado, pero el flujo principal nuevo vive en `play.py`.

## Estado Actual

El proyecto actualmente permite:

- levantar un servidor local de Pokemon Showdown con Docker;
- conectar el bot principal con identidad fija;
- conectar un segundo bot con otra cuenta;
- jugar partidas locales entre dos bots por challenge;
- jugar una partida real en ladder oficial en formato random battle;
- elegir politica `random` o `greedy`;
- elegir politica `alphazero_mcts` con checkpoint entrenado;
- usar `showdown-sim` para evaluar ramas MCTS con el motor real de Showdown;
- consultar el estado vivo de batallas locales desde Showdown para que MCTS no reconstruya el historial a mano;
- imprimir turnos, estado resumido y decisiones del bot;
- detectar el final de la partida desde el codigo y cerrar la sesion;
- limpiar batallas abiertas viejas al arrancar para evitar arrastrar partidas anteriores.

Las cuentas configuradas por defecto son:

| Rol | Usuario | Password |
|---|---|---|
| Bot principal | `Laboratorio3IA` | `123456789` |
| Bot rival | `Laboratorio3IA-B` | `123456789` |

Tambien se pueden sobrescribir con variables de entorno:

- `SHOWDOWN_USERNAME`
- `SHOWDOWN_PASSWORD`
- `SHOWDOWN_OPPONENT_USERNAME`
- `SHOWDOWN_OPPONENT_PASSWORD`

## Requisitos

- Docker Desktop
- Docker Compose

En Windows, los comandos pueden funcionar como `docker compose ...` o como `docker-compose ...` segun la instalacion. Este README usa `docker compose`, que es el formato recomendado por Docker Compose v2.

Si en tu maquina `docker compose` falla pero `docker-compose` funciona, usa la segunda forma.

## Crear Los Contenedores

Desde la raiz del repo:

```powershell
docker compose down
docker compose build
docker compose up -d showdown showdown-sim
docker compose ps
```

Los servicios `showdown` y `showdown-sim` deben quedar `healthy`. `showdown` expone el websocket local en `8000` y un puente de estado vivo en `9002`; `showdown-sim` expone el motor JS real de Pokemon Showdown en `9001` para busquedas MCTS profundas.

Checks rapidos:

```powershell
Invoke-WebRequest http://localhost:9002/health
Invoke-WebRequest http://localhost:9001/health
```

Para ver logs del servidor local o del simulador:

```powershell
docker compose logs -f showdown
docker compose logs -f showdown-sim
```

Para apagar todo:

```powershell
docker compose down
```

## Flujo Principal: `play.py`

`play.py` tiene dos modos:

| Modo | Uso |
|---|---|
| `challenge` | Dos bots controlados por el script juegan entre si. |
| `ladder` | El bot principal busca una partida contra un rival real/random en ladder. |

Politicas disponibles:

| Politica | Descripcion |
|---|---|
| `random` | Elige acciones aleatorias. |
| `greedy` | Usa `MaxBasePowerPlayer`, priorizando movimientos de mayor potencia. |

Opciones importantes:

| Opcion | Descripcion |
|---|---|
| `--n` | Cantidad de partidas. |
| `--mode` | `challenge` o `ladder`. |
| `--p1` | Politica del bot principal. |
| `--p2` | Politica del segundo bot en modo `challenge`. |
| `--server` | `official` para Showdown real, o `showdown:8000` para el servidor local Docker. |
| `--format` | Formato de Showdown. |
| `--battle-timeout` | Tiempo maximo para esperar las partidas. Si se cumple, abandona batallas abiertas y cierra. |
| `--login-timeout` | Tiempo maximo para esperar login inicial. |

Ver ayuda:

```powershell
docker compose run --rm trainer python play.py --help
```

## Partida Local Entre Dos Bots

Primero levantar el servidor local:

```powershell
docker compose up -d showdown
```

### Random battle local

Este formato no usa `team.txt`; Showdown genera los equipos:

```powershell
docker compose run --rm trainer python -u play.py --mode challenge --n 1 --p1 greedy --p2 random --server showdown:8000 --format gen9randombattle --battle-timeout 120 --login-timeout 30
```

Salida esperada:

- conecta `Laboratorio3IA`;
- conecta `Laboratorio3IA-B`;
- limpia batallas abiertas viejas;
- inicia una partida local;
- imprime turnos y decisiones;
- detecta victoria/derrota;
- cierra la sesion y termina.

### VGC local con `team.txt`

El formato VGC probado como valido es:

```text
gen9vgc2026regi
```

Comando:

```powershell
docker compose run --rm trainer python -u play.py --mode challenge --n 1 --p1 greedy --p2 random --server showdown:8000 --format gen9vgc2026regi --battle-timeout 120 --login-timeout 30
```

Este modo usa el equipo de `team.txt`.

## Partida Real En Showdown Oficial

Para jugar contra una persona random en ladder oficial, usar `--mode ladder` y `--server official`.

El modo mas simple es `gen9randombattle`, porque no requiere `team.txt`:

```powershell
docker compose run --rm trainer python -u play.py --mode ladder --n 1 --p1 random --server official --format gen9randombattle --battle-timeout 600 --login-timeout 30
```

Este comando:

- conecta `Laboratorio3IA` al servidor oficial;
- limpia batallas abiertas viejas;
- busca una partida en ladder;
- imprime turnos y decisiones;
- detecta el final;
- muestra resultado;
- cierra el proceso.

Ejemplo de salida final real:

```text
Final: battle-gen9randombattle-2594202794 -> DERROTA en 16 turnos.

RESULTADOS
Principal RANDOM   victorias: 0 / 1

Perdio el bot principal (RANDOM).
```

### Nota Sobre Challenges En El Servidor Oficial

El challenge directo entre `Laboratorio3IA` y `Laboratorio3IA-B` puede ser bloqueado por Pokemon Showdown oficial con un mensaje anti-spam, especialmente si la red o las cuentas son nuevas.

Mensaje observado:

```text
Due to spam from your internet provider, you can't challenge others right now.
Logging into an account you've used a lot in the past will allow you to challenge.
```

Por eso, para probar en servidor real, el flujo recomendado por ahora es ladder:

```powershell
docker compose run --rm trainer python -u play.py --mode ladder --n 1 --p1 random --server official --format gen9randombattle --battle-timeout 600 --login-timeout 30
```

Para self-play, entrenamiento o muchas pruebas repetidas, usar el servidor local Docker.

## Ingesta De Replays Y Dataset

Antes de implementar modelos nuevos, el repo tiene una ingesta comun de replays. La idea es que todos los modelos lean desde el mismo dataset base, y que cada modelo tenga luego su propio adaptador si necesita tensores distintos.

La ingesta guarda tres capas:

```text
data/replays/
|-- raw/<format>/<battle_id>.json
|-- parsed/<format>/<battle_id>.json
|-- datasets/<format>_decisions.jsonl
|-- datasets/<format>_double_decisions.jsonl
`-- index.jsonl
```

Politica de almacenamiento:

- `raw/` guarda el JSON original de Pokemon Showdown sin modificar.
- `parsed/` guarda una version normalizada por turnos, eventos, jugadores, acciones y resultado.
- `datasets/*_decisions.jsonl` guarda muestras de decision derivadas del parsed.
- `datasets/*_double_decisions.jsonl` agrupa acciones por turno y lado para VGC dobles.
- `index.jsonl` mantiene un indice de replays descargados.
- El codigo no borra ni sobreescribe replays raw ya descargados.
- `data/replays/` esta ignorado por Git porque puede crecer mucho.

### Descargar replays de las cuentas de los bots

```powershell
docker compose run --rm trainer python scripts/ingest_replays.py --format gen9randombattle --include-default-bots --limit 25 --pages 1
```

Para VGC, usar el formato vigente:

```powershell
docker compose run --rm trainer python scripts/ingest_replays.py --format gen9vgc2026regi --include-default-bots --limit 25 --pages 1
```

Reconstruir datasets derivados desde replays raw ya descargados:

```powershell
docker compose run --rm trainer python scripts/ingest_replays.py --format gen9vgc2026regi --rebuild-parsed
```

### Descargar replays de un usuario especifico

```powershell
docker compose run --rm trainer python scripts/ingest_replays.py --format gen9randombattle --user MichaelderBeste2 --limit 10 --pages 1
```

Se puede repetir `--user`:

```powershell
docker compose run --rm trainer python scripts/ingest_replays.py --format gen9randombattle --user UsuarioA --user UsuarioB --limit 10
```

### Descargar replays de una lista de usuarios

Crear un archivo, por ejemplo:

```text
data/replay_users/gen9randombattle_top.txt
```

con un usuario por linea:

```text
smokyaim
milkreo
helicopyer
```

Luego correr:

```powershell
docker compose run --rm trainer python scripts/ingest_replays.py --format gen9randombattle --users-file data/replay_users/gen9randombattle_top.txt --limit 20 --pages 1
```

### Descargar desde el top ladder

Este comando toma usuarios del ladder publico del formato y descarga replays publicos de esos usuarios:

```powershell
docker compose run --rm trainer python scripts/ingest_replays.py --format gen9randombattle --top-ladder 10 --limit 5 --pages 1
```

Esto busca hasta 10 usuarios top y descarga hasta 5 replays por usuario.

### Descargar un replay puntual por ID

```powershell
docker compose run --rm trainer python scripts/ingest_replays.py --format gen9randombattle --replay-id gen9randombattle-2592325086
```

### Reconstruir `parsed/` desde `raw/`

Si se mejora el parser, no hace falta descargar todo de nuevo:

```powershell
docker compose run --rm trainer python scripts/ingest_replays.py --format gen9randombattle --rebuild-parsed --reparse
```

### Probar sin escribir archivos

```powershell
docker compose run --rm trainer python scripts/ingest_replays.py --format gen9randombattle --top-ladder 3 --limit 1 --dry-run
```

## Smoke Test Legado: `battle.py`

`battle.py` sigue existiendo como prueba rapida del flujo viejo.

Ejemplo local:

```powershell
docker compose run --rm trainer python battle.py --n 3 --p1 greedy --p2 random --server showdown:8000 --format gen9vgc2026regi
```

El flujo principal recomendado es `play.py`, pero `battle.py` sirve para verificar rapidamente que Showdown local y `poke-env` siguen funcionando.

## Entrenamiento PPO

El entrenamiento principal sigue en:

```text
train.py
```

Dry-run:

```powershell
docker compose run --rm trainer python train.py --dry-run
```

Entrenar contra oponente random en servidor local:

```powershell
docker compose run --rm trainer python train.py --server showdown:8000 --opponent random
```

Entrenar contra greedy:

```powershell
docker compose run --rm trainer python train.py --server showdown:8000 --opponent greedy
```

Continuar desde checkpoint:

```powershell
docker compose run --rm trainer python train.py --server showdown:8000 --resume checkpoints/vgc_ppo_100000.zip
```

Los checkpoints se guardan en:

```text
checkpoints/
```

Los logs se guardan en:

```text
logs/
```

## AlphaZero-Style MCTS + PPO

La branch del modelo agrega una politica `alphazero_mcts` para `play.py` y estos scripts:

- `scripts/pretrain_alphazero_replays.py`: preentrena la red policy+value usando `datasets/<format>_double_decisions.jsonl`.
- `scripts/train_alphazero_mcts_ppo.py`: juega partidas locales con MCTS y actualiza la red con PPO + distribucion de visitas MCTS.
- `scripts/evaluate_alphazero_offline.py`: evalua checkpoints sin usar websocket.
- `scripts/summarize_simulator_diagnostics.py`: resume diagnosticos del simulador cuando se guardan.

El modelo usa un ranker de acciones legales: en cada turno toma el estado actual, enumera las acciones dobles legales de `poke-env`, puntua cada candidata y MCTS elige. Asi no depende de un vocabulario cerrado de acciones de otros equipos.

Para depth 2 o mayor hay dos servicios importantes:

- `showdown-sim` (`http://showdown-sim:9001`): evalua ramas con el motor real de Pokemon Showdown.
- `showdown-live-state` (`http://showdown:9002`): lee el estado interno serializado de la batalla viva y evita reconstruir el historial por texto.

Antes de entrenar, reconstruir el dataset doble si hace falta:

```powershell
docker compose run --rm trainer python scripts/ingest_replays.py --format gen9vgc2026regi --rebuild-parsed
```

Preentrenar desde replays descargados:

```powershell
docker compose run --rm trainer python -u scripts/pretrain_alphazero_replays.py --dataset data/replays/datasets/gen9vgc2026regi_double_decisions.jsonl --epochs 10 --batch-size 128 --output-dir checkpoints/alphazero_pretrain --device cpu
```

Continuar con MCTS+PPO en servidor local usando estado vivo:

```powershell
docker compose run --rm trainer python -u scripts/train_alphazero_mcts_ppo.py --iterations 50 --self-play-games 10 --opponent-cycle random,greedy,self,greedy --mcts-simulations 128 --mcts-depth 2 --max-candidates 96 --simulator-max-choices 8 --simulator-opponent-policy robust --simulator-robust-worst-weight 0.35 --simulator-timeout 180 --live-state-url http://showdown:9002 --require-simulator --server showdown:8000 --format gen9vgc2026regi --team team.txt --device cpu --output-dir checkpoints/alphazero_mcts_ppo_d2_required --rollout-path data/alphazero/rollouts_d2_required.jsonl
```

El script escribe eventos de entrenamiento y memoria en `logs/alphazero_train_events.jsonl` por defecto. Se puede cambiar con `--training-log-path logs/mi_run.jsonl`.

Para entrenamientos profundos (`--mcts-depth 4` o mas), usar `--simulator-eval-candidates` para podar cuantas acciones se evaluan con el simulador exacto en cada decision. Sin esa poda, `--max-candidates 96` junto con `--simulator-max-choices 8` puede crear una evaluacion exponencial demasiado grande para `showdown-sim`.

Al finalizar, tambien genera un reporte comparable con el formato usado por el entrenamiento PPO:

```text
reports/alphazero_<timestamp>/report.html
reports/alphazero_<timestamp>/report.json
reports/alphazero_<timestamp>/common_metrics.json
reports/alphazero_<timestamp>/league_stats.json
reports/alphazero_<timestamp>/plots/
```

`report.html` mantiene la misma estructura general del reporte PPO recurrente: resumen global, detalle por stage, curvas/plots, seccion de league y configuracion del entrenamiento. `common_metrics.json` contiene el bloque normalizado para comparar modelos distintos: familia del modelo, algoritmo, partidas, victorias, derrotas, empates, win rate, reward promedio, largo promedio de partida, losses finales, updates y decisiones totales.

Para regenerar el reporte de un entrenamiento AlphaZero ya existente:

```powershell
docker compose run --rm trainer python scripts/export_model_metrics.py alphazero --training-log-path logs/alphazero_train_events.jsonl --rollout-path data/alphazero/rollouts.jsonl --output-dir reports/alphazero_common
```

Para normalizar un `report.json` del PPO recurrente y medirlo con el mismo contrato:

```powershell
docker compose run --rm trainer python scripts/export_model_metrics.py ppo --report-json reports/run_YYYYMMDD_HHMMSS/report.json --output-dir reports/ppo_common
```

Si el header de `play.py` muestra `estado=http://showdown:9002`, MCTS esta usando el estado vivo del servidor local. Si no se pasa `--live-state-url`, el codigo vuelve al tracker por historial, que sirve como fallback pero puede divergir por RNG o reparaciones del replay.

Probar el checkpoint contra `random`:

```powershell
docker compose run --rm trainer python -u play.py --mode challenge --n 30 --p1 alphazero_mcts --p2 random --server showdown:8000 --format gen9vgc2026regi --team team.txt --alphazero-checkpoint checkpoints/alphazero_mcts_ppo_d2_required/best.pt --alphazero-simulations 128 --alphazero-depth 2 --alphazero-max-candidates 96 --alphazero-simulator-max-choices 8 --alphazero-simulator-opponent-policy robust --alphazero-simulator-robust-worst-weight 0.35 --alphazero-simulator-timeout 180 --alphazero-live-state-url http://showdown:9002 --alphazero-require-simulator --alphazero-device cpu --battle-timeout 1800
```

Probar contra `greedy`:

```powershell
docker compose run --rm trainer python -u play.py --mode challenge --n 30 --p1 alphazero_mcts --p2 greedy --server showdown:8000 --format gen9vgc2026regi --team team.txt --alphazero-checkpoint checkpoints/alphazero_mcts_ppo_d2_required/best.pt --alphazero-simulations 128 --alphazero-depth 2 --alphazero-max-candidates 96 --alphazero-simulator-max-choices 8 --alphazero-simulator-opponent-policy robust --alphazero-simulator-robust-worst-weight 0.35 --alphazero-simulator-timeout 180 --alphazero-live-state-url http://showdown:9002 --alphazero-require-simulator --alphazero-device cpu --battle-timeout 1800
```

Los checkpoints, rollouts, replays descargados y logs quedan fuera de Git por `.gitignore`: `checkpoints/`, `data/`, `logs/`, `models/`, `src/models/` y extensiones comunes de modelos.

Si Docker muestra `ExitCode=137` u `OOMKilled=true`, el entrenamiento fue matado por memoria. El entrenamiento ahora carga la ventana `--train-window` en streaming desde offsets del JSONL y no mantiene todos los rollouts parseados en memoria durante PPO. Si vuelve a pasar, bajar temporalmente `--train-window`, `--batch-size` o subir la memoria de Docker Desktop.

Rumbo proximo recomendado: revisar la calidad del modelo despues de un entrenamiento que llegue al numero de iteraciones planeado. Para diagnostico fino, mirar `logs/alphazero_train_events.jsonl`, el ultimo checkpoint, `data/alphazero/rollouts*.jsonl` y, si hace falta, activar `--simulator-diagnostics-path logs/simulator_diagnostics.jsonl` para separar errores de simulador, timeouts, reparaciones y cortes por excepcion.

## TensorBoard

```powershell
docker compose --profile monitoring up -d tensorboard
```

Abrir:

```text
http://localhost:6006
```

## Tests Y Scripts Utiles

Test del calculo de dano:

```powershell
docker compose run --rm trainer python scripts/test_damage.py
```

Test del encoding de estado:

```powershell
docker compose run --rm trainer python scripts/test_state_encoding.py
```

Listar formatos:

```powershell
docker compose run --rm trainer python list_formats.py
```

## Estructura Del Repo

```text
.
|-- README.md
|-- GUIDE.md
|-- FETCH_DATA_README.md
|-- team.txt
|-- requirements.txt
|-- Dockerfile
|-- Dockerfile.showdown
|-- docker-compose.yml
|-- login.py
|-- play.py
|-- battle.py
|-- train.py
|-- list_formats.py
|-- data/
|   |-- get_data.py
|   `-- raw/
|       |-- abilities.json
|       |-- items.json
|       |-- moves.json
|       |-- natures.json
|       |-- pokemon.json
|       `-- type_chart.json
|-- scripts/
|   |-- evaluate_alphazero_offline.py
|   |-- pretrain_alphazero_replays.py
|   |-- summarize_simulator_diagnostics.py
|   |-- train_alphazero_mcts_ppo.py
|   |-- test_damage.py
|   `-- test_state_encoding.py
|-- tools/
|   |-- patch_showdown_live_state.js
|   |-- showdown_live_state_bridge.js
|   `-- showdown_sim_server.js
`-- src/
    |-- alphazero/
    |   |-- features.py
    |   |-- mcts.py
    |   |-- network.py
    |   |-- offline_selfplay.py
    |   |-- player.py
    |   `-- showdown_simulator.py
    |-- damage_calc.py
    |-- format_resolver.py
    |-- state_encoder.py
    |-- utils.py
    `-- vgc_env.py
```

## Archivos Principales

### `login.py`

Contiene la logica de crear bots autenticados y resolver el servidor.

No loguea al importar. `play.py` llama explicitamente a sus funciones.

Funciones principales:

- `connect_main_bot(...)`
- `connect_opponent_bot(...)`
- `build_server_config(...)`

### `play.py`

Flujo principal actual:

```text
conectar -> limpiar batallas viejas -> jugar N partidas -> mostrar resultado -> cerrar
```

Tambien deja preparado el flujo futuro:

```text
conectar -> loop de comandos -> cerrar
```

### `team.txt`

Equipo usado para formatos que requieren equipo propio, como `gen9vgc2026regi`.

Los formatos random como `gen9randombattle` no usan `team.txt`.

### `src/vgc_env.py`

Define `VGCEnv`, entorno de dobles Gen 9 basado en `poke-env`.

### `src/state_encoder.py`

Convierte estado de batalla en vector numerico para modelos.

### `src/damage_calc.py`

Implementa calculo aproximado de dano.

## Flujo Recomendado De Desarrollo

```powershell
docker compose down
docker compose build
docker compose up -d showdown showdown-sim

docker compose run --rm trainer python train.py --dry-run

docker compose run --rm trainer python -u play.py --mode challenge --n 1 --p1 greedy --p2 random --server showdown:8000 --format gen9randombattle --battle-timeout 120 --login-timeout 30
```

Luego, para probar una partida real:

```powershell
docker compose run --rm trainer python -u play.py --mode ladder --n 1 --p1 random --server official --format gen9randombattle --battle-timeout 600 --login-timeout 30
```

Para cerrar:

```powershell
docker compose down
```

## Objetivo Experimental

El objetivo del proyecto es construir un agente capaz de jugar Pokemon Showdown usando primero politicas simples y luego modelos entrenados.

La version actual esta enfocada en infraestructura:

1. conectar cuentas;
2. jugar partidas locales y reales;
3. observar estado por turno;
4. ejecutar decisiones basicas;
5. cerrar sesiones correctamente;
6. preparar el camino para modelos entrenados.
