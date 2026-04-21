# VGC Bot — Guía Completa de Uso y Entrenamiento

## Índice
1. [Setup inicial](#setup-inicial)
2. [Ejecución del programa](#ejecución-del-programa)
3. [Entrenamiento con imitation learning](#entrenamiento-con-imitation-learning)
4. [Debugging y logging (ver todas las decisiones)](#debugging-y-logging)
5. [Comandos útiles](#comandos-útiles)

---

## Setup inicial

### 1. Crear el entorno virtual e instalar dependencias

```bash
# Navegar a la carpeta del proyecto
cd "laboratorio pokemon showdown"

# Crear venv
python -m venv venv

# Activar venv
# En Windows:
venv\Scripts\activate
# En macOS/Linux:
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Verificar que los datos estén en el lugar correcto

Los archivos JSON deben estar en:
```
C:\Users\Lenovo\OneDrive\Escritorio\vgc_bot_data\
  ├── pokemon.json       (1025 Pokémon)
  ├── moves.json         (937 movimientos)
  ├── type_chart.json    (tabla de tipos)
  ├── items.json         (2000 objetos)
  └── abilities.json     (367 habilidades)
```

Si no los tienes, descargalos:
```bash
python data/fetch_data.py --output-dir "C:\Users\Lenovo\OneDrive\Escritorio\vgc_bot_data"
```

---

## Ejecución del programa

### Paso 0: Verificación rápida (dry-run)

Antes de hacer nada, chequea que todos los módulos funcionen:

```bash
python train.py --dry-run
```

Esto va a:
- Cargar todos los datos de PokeAPI ✓
- Parsear el equipo desde team.txt ✓
- Calcular stats de todos los Pokémon ✓
- Verificar que el damage calculator funcione ✓
- Crear el vector de observación ✓
- Verificar que el environment de Gymnasium esté OK ✓

**Output esperado:**
```
═══════════════════════════════════════════════════════════
  VGC Bot — Dry Run (verificación de módulos)
═══════════════════════════════════════════════════════════

[1/4] Verificando utils...
  ✓ 1025 Pokémon | 937 movimientos | 2000 items
  ✓ Equipo: ['kyogre', 'calyrex-shadow', 'incineroar', ...]
     kyogre              HP=175 | SpA=150 | Spe=115
     ...

[2/4] Verificando damage_calc...
  ✓ water-spout → calyrex-shadow  45.2% – 53.1% del HP  (STAB)
  ✓ 18 matchups calculados para Kyogre

[3/4] Verificando state_encoder...
  ✓ Observation vector: shape=(380,) | min=0.000 | max=1.000

[4/4] Verificando vgc_env...
  ✓ action_space:      Discrete(16)
  ✓ observation_space: Box(380,)

═══════════════════════════════════════════════════════════
  ✓ Todos los módulos funcionan correctamente.
  Listo para conectar con Pokémon Showdown y entrenar.
═══════════════════════════════════════════════════════════
```

Si todo dice ✓, estás listo para entrenar.

### Paso 1: Configurar un servidor local de Pokémon Showdown

Para entrenar necesitás un servidor local de Showdown donde el bot se conecte.

**Opción A: Usando Docker (más fácil)**

```bash
# Si tenés Docker instalado
docker pull smogon/pokemon-showdown
docker run -d -p 8000:8000 smogon/pokemon-showdown
```

**Opción B: Instalación manual**

```bash
# Clonar el repo de Showdown
git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown

# Instalar dependencias de Node.js
npm install

# Iniciar servidor
npm start
# Por defecto corre en localhost:8000
```

Después modificá `train.py` línea 150-155 para que apunte a tu servidor:
```python
server = ServerConfiguration(
    "localhost:8000",    # ← cambiar si el servidor está en otro puerto/IP
    "https://play.pokemonshowdown.com/action.php"
)
```

### Paso 2: Entrenar desde cero

**IMPORTANTE:** Este comando necesita que el servidor de Showdown esté corriendo en otra terminal.

En una terminal (Terminal 1):
```bash
# Iniciar servidor de Showdown
docker run -p 8000:8000 smogon/pokemon-showdown
# O si lo instalaste manual: cd pokemon-showdown && npm start
```

En otra terminal (Terminal 2):
```bash
cd "laboratorio pokemon showdown"
venv\Scripts\activate
python train.py
```

Esto va a:
- Conectarse al servidor local
- Crear un agente PPO con pesos aleatorios
- Jugar 1 millón de timesteps (batallas) contra oponentes aleatorios
- Guardar checkpoints cada 10,000 timesteps en `checkpoints/`
- Loguear todo en `logs/` (para TensorBoard)

**Duración estimada:**
- 1 millón de timesteps ≈ 100,000–200,000 batallas (esto puede tardar días/semanas)
- Primeros checkpoints (10k-100k) = días 1-3
- Mejoras significativas = semana 2+

### Paso 3: Continuar entrenamiento desde un checkpoint

```bash
python train.py --resume checkpoints/vgc_ppo_100000.zip
```

Esto carga el modelo guardado en el checkpoint y continúa desde ahí.

### Paso 4: Ver el progreso en TensorBoard

```bash
tensorboard --logdir logs
```

Luego abrí `http://localhost:6006` en tu navegador para ver:
- Reward por episodio
- Win rate
- Longitud de episodios
- Pérdida del modelo

---

## Entrenamiento con imitation learning

En lugar de empezar desde cero con pesos aleatorios, podés preentrenar el modelo
viendo cómo juegan los expertos.

### Fase 1: Descargar replays de alto nivel

Los replays de Pokémon Showdown están públicamente disponibles.

```bash
# Script para descargar replays (crear nuevo archivo)
# save como scripts/download_replays.py

python scripts/download_replays.py \
  --format vgc_reg_i \
  --min_rating 1500 \
  --count 1000 \
  --output data/replays/
```

O manually:
1. Abrir https://replay.pokemonshowdown.com/
2. Filtrar por VGC Regulation I
3. Descargar replays JSON (botón derecho → Guardar como)
4. Guardar en `data/replays/`

### Fase 2: Procesar replays y extraer transiciones

Crea `scripts/imitation_learning.py`:

```python
"""
imitation_learning.py
─────────────────────────────────────────────────────────────────
Preentrenamiento supervisado del modelo usando replays de jugadores expertos.

Las transiciones extraídas se guardan como:
  (observation, action) → (reward, next_observation)

que se usa para entrenar un modelo supervisado que imita las decisiones
del experto antes de hacer RL.
"""

import json
import numpy as np
from pathlib import Path
from src.utils import load_all_data, parse_team, calc_all_stats
from src.state_encoder import StateEncoder

def extract_transitions_from_replay(replay_json: dict, encoder: StateEncoder) -> list:
    """
    Parsea un replay JSON de Showdown y extrae las transiciones
    (state, action_taken) de cada turno.

    Returns:
        list de (observation_np, action_int) tuples
    """
    transitions = []

    # Parsear el replay JSON para sacar los estados y acciones
    # Esto depende de la estructura exacta del JSON de Showdown
    # Por ahora, placeholder:

    for turn in replay_json.get("turns", []):
        # Extraer el estado actual
        obs = encoder.encode_manual(
            own_field   = parse_field(turn, "p1"),
            rival_field = parse_field(turn, "p2"),
            benched_own = parse_bench(turn, "p1"),
            conditions  = parse_conditions(turn),
        )

        # Extraer la acción tomada
        action = parse_action(turn, "p1")

        transitions.append((obs, action))

    return transitions

def train_imitation_model(replays_dir: Path, output_model: Path):
    """
    Entrena un modelo supervisado usando replays.
    Después este modelo se usa como punto de partida para RL.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    import joblib

    print("Extrayendo transiciones de replays...")

    data    = load_all_data()
    encoder = StateEncoder(data["type_chart"], data["moves"])

    all_obs  = []
    all_actions = []

    for replay_file in replays_dir.glob("*.json"):
        with open(replay_file) as f:
            replay = json.load(f)

        transitions = extract_transitions_from_replay(replay, encoder)
        for obs, action in transitions:
            all_obs.append(obs)
            all_actions.append(action)

    X = np.array(all_obs, dtype=np.float32)
    y = np.array(all_actions, dtype=int)

    print(f"Transiciones extraídas: {len(X)}")
    print(f"Distribución de acciones: {np.bincount(y)}")

    # Entrenar con Random Forest (supervisado)
    print("Entrenando modelo supervisado...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        n_jobs=-1,
        verbose=1,
    )
    model.fit(X, y)

    # Guardar modelo
    joblib.dump(model, output_model)
    print(f"✓ Modelo guardado en {output_model}")

    # Extraer los pesos del modelo para inicializar la red neuronal
    # (esto es complejo, se requiere convertir los pesos del RF a una red)
    return model

if __name__ == "__main__":
    replays_dir = Path("data/replays")
    output_model = Path("models/imitation_model.pkl")

    train_imitation_model(replays_dir, output_model)
```

### Fase 3: Usar el modelo de imitación como punto de partida para RL

Modifica `train.py`:

```python
def train(resume_path: Optional[str] = None, use_imitation: bool = False):
    ...
    if use_imitation and Path("models/imitation_model.pkl").exists():
        print("Cargando pesos del modelo de imitación...")
        import joblib
        imitation_model = joblib.load("models/imitation_model.pkl")
        # Convertir los pesos del RF a la red neuronal de PPO
        # (esto requiere un proceso especial)

    # Entrenar con RL desde los pesos inicializados
    model.learn(total_timesteps=1_000_000, ...)
```

Ejecutar:
```bash
python train.py --use-imitation
```

---

## Debugging y logging

### Opción 1: Imprimir cálculos de daño (sin RL)

Crea `scripts/test_damage.py`:

```python
"""
test_damage.py — Prueba el damage calculator imprimiendo todos los cálculos.
"""

from src.utils import load_all_data, parse_team, calc_all_stats, get_pokemon
from src.damage_calc import calc_damage, BattleConditions
from pathlib import Path

data      = load_all_data()
team_path = Path(__file__).resolve().parent.parent / "team.txt"
team      = parse_team(team_path)

# Calcular stats de todos
team_stats = []
for p in team:
    stats = calc_all_stats(p, data["pokemon"])
    pinfo = get_pokemon(p["name"], data["pokemon"])
    types = pinfo["types"] if pinfo else []
    team_stats.append((p, stats, types))

# Testear todos los matchups posibles del primer Pokémon
kyogre, k_stats, k_types = team_stats[0]

print(f"\n{'='*70}")
print(f"  ANÁLISIS DE DAÑO: {kyogre['name'].upper()}")
print(f"{'='*70}\n")

for move_name in kyogre["moves"]:
    move = data["moves"].get(move_name)
    if not move or move.get("category") == "status":
        continue

    print(f"\nMOVIMIENTO: {move_name.upper()}")
    print(f"  Tipo: {move['type']} | Potencia: {move.get('power')} | "
          f"Categoría: {move['category']}")
    print(f"  Target: {move.get('target')}")

    # Contra cada posible rival
    for i, (p, stats, types) in enumerate(team_stats):
        if p["name"] == kyogre["name"]:
            continue

        # Sin rain
        result_no_rain = calc_damage(
            attacker_stats=k_stats,
            attacker_types=k_types,
            attacker_ability=kyogre.get("ability", ""),
            attacker_item=kyogre.get("item"),
            attacker_name=kyogre["name"],
            move=move,
            defender_stats=stats,
            defender_types=types,
            defender_ability="",
            defender_item=None,
            defender_name=p["name"],
            type_chart=data["type_chart"],
            conditions=BattleConditions(weather="none"),
        )

        # Con rain (favorable para Kyogre)
        result_rain = calc_damage(
            attacker_stats=k_stats,
            attacker_types=k_types,
            attacker_ability=kyogre.get("ability", ""),
            attacker_item=kyogre.get("item"),
            attacker_name=kyogre["name"],
            move=move,
            defender_stats=stats,
            defender_types=types,
            defender_ability="",
            defender_item=None,
            defender_name=p["name"],
            type_chart=data["type_chart"],
            conditions=BattleConditions(weather="rain"),
        )

        print(f"\n  vs {p['name'].upper():20} ({'/'.join(types)})")
        print(f"    Sin rain:  {result_no_rain.min_pct*100:5.1f}% – {result_no_rain.max_pct*100:5.1f}% {' [OHKO!]' if result_no_rain.ohko else ' [2HKO!]' if result_no_rain.two_hit_ko else ''}")
        print(f"    Con rain:  {result_rain.min_pct*100:5.1f}% – {result_rain.max_pct*100:5.1f}% {' [OHKO!]' if result_rain.ohko else ' [2HKO!]' if result_rain.two_hit_ko else ''}")
        print(f"    Efectividad: x{result_rain.effectiveness}")

print(f"\n{'='*70}")
```

Ejecutar:
```bash
python scripts/test_damage.py
```

### Opción 2: Logging verboso durante RL

Modificá `train.py` para agregar logging:

```python
import logging

# Configurar logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/bot_debug.log'),
        logging.StreamHandler(),  # también a consola
    ]
)

logger = logging.getLogger("VGCBot")

# Dentro de vgc_env.py, en calc_reward():
def calc_reward(self, last_battle):
    reward = 0.0
    ...
    logger.debug(f"Turn {last_battle.turn}: "
                 f"Kyogre HP: {curr_own_hp[0]*100:.1f}% | "
                 f"Rival HP: {curr_rival_hp[0]*100:.1f}% | "
                 f"Reward +{reward:.2f}")
    ...
```

Ejecutar:
```bash
python train.py 2>&1 | tee logs/training.log
```

### Opción 3: Debugger paso a paso

Para ver exactamente qué decide el modelo en cada turno:

```python
# En vgc_env.py, método step()
def step(self, action: int):
    logger.info(f"\n--- TURNO {self.battle.turn} ---")
    logger.info(f"Acción elegida: {action}")

    # Decodificar qué significa la acción
    if action < 4:
        move_idx = action
        logger.info(f"  → Movimiento {move_idx} con Pokémon 1")
    elif action < 8:
        move_idx = action - 4
        logger.info(f"  → Movimiento {move_idx} con Pokémon 2")
    ...

    # Ejecutar
    obs, reward, done, truncated, info = super().step(action)

    logger.info(f"Reward obtenido: {reward:.3f}")
    logger.info(f"Nueva observación shape: {obs.shape}")

    return obs, reward, done, truncated, info
```

---

## Comandos útiles

### Comandos básicos

```bash
# Dry-run (verificación)
python train.py --dry-run

# Entrenar desde cero
python train.py

# Continuar desde checkpoint
python train.py --resume checkpoints/vgc_ppo_100000.zip

# Ver el progreso en TensorBoard
tensorboard --logdir logs

# Testear damage calculator
python scripts/test_damage.py

# Usar imitation learning
python train.py --use-imitation
```

### Monitoring del training

```bash
# Ver logs en tiempo real
tail -f logs/bot_debug.log

# Ver estado de checkpoints
ls -lh checkpoints/

# Ver estadísticas de TensorBoard
tensorboard --logdir logs --port 6006
```

### Limpiar y resetear

```bash
# Eliminar todos los checkpoints (para empezar de cero)
rm -rf checkpoints/*

# Eliminar logs de TensorBoard
rm -rf logs/*

# Limpiar archivos .pyc
find . -type d -name __pycache__ -exec rm -rf {} +
```

---

## Troubleshooting

### Error: "No se puede conectar a Showdown"

**Causa:** El servidor local no está corriendo.

**Solución:**
```bash
# En Terminal 1:
docker run -p 8000:8000 smogon/pokemon-showdown
# O: cd pokemon-showdown && npm start

# En Terminal 2:
python train.py
```

### Error: "FileNotFoundError: pokemon.json"

**Causa:** Los datos no se descargaron.

**Solución:**
```bash
python data/fetch_data.py --output-dir "C:\Users\Lenovo\OneDrive\Escritorio\vgc_bot_data"
```

### Error: "ModuleNotFoundError: poke_env"

**Causa:** Falta instalar dependencias.

**Solución:**
```bash
pip install -r requirements.txt
```

### Model no mejora (reward plano)

**Causa:** El reward function está mal calibrado o la learning rate es baja.

**Solución:** Modificá los pesos en `train.py`:
```python
LEARNING_RATE = 1e-3  # aumentar de 3e-4
N_STEPS = 4096        # aumentar de 2048
```

---

¿Tenés más preguntas sobre cómo usar el sistema?
