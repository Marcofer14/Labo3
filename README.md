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
docker compose up -d showdown
docker compose ps
```

El servicio `showdown` debe quedar `healthy`.

Para ver logs del servidor local:

```powershell
docker compose logs -f showdown
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
|   |-- test_damage.py
|   `-- test_state_encoding.py
`-- src/
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
docker compose up -d showdown

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
