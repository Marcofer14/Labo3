# IA en Juegos: ¿Puede una computadora vencer al humano en Pokémon?
### Laboratorio de Datos III — 2026

Agente de IA entrenado con **aprendizaje por refuerzo** para jugar Pokémon competitivo en formato **VGC Regulation I** sobre Pokémon Showdown. El objetivo: alcanzar el mayor ELO posible en la ladder oficial sin conocimiento explícito de estrategia.

---

## Equipo

El agente juega con un equipo fijo (4 de 6 por batalla). Ver `team.txt` para el set completo en formato Pokepaste.

| Pokémon | Objeto | Rol |
|---|---|---|
| Kyogre | Mystic Water | Restricted — setup de rain + sweeper |
| Calyrex-Shadow | Focus Sash | Restricted — sweeper / control |
| Incineroar | Safety Goggles | Support — Intimidate + Fake Out |
| Rillaboom | Assault Vest | Support — Grassy Surge + prioridad |
| Urshifu-Rapid-Strike | Choice Scarf | Attacker — rompe Protect |
| Roaring Moon | Booster Energy | Setter — Tailwind + sweeper físico |

---

## Pipeline del Proyecto

### 1. Recolección de datos
- Stats, movimientos, habilidades y objetos vía **PokeAPI** → `data/raw/`
- Replays de jugadores top de la ladder VGC Reg I de Showdown (formato JSON)
- Estadísticas de uso de Smogon (usage stats mensuales por ELO)

### 2. Diseño del espacio de estados y acciones
**Estado (observación):**
- HP actual y máximo de los 4 Pokémon en campo (propio y rival)
- Tipos, habilidades activas, objetos
- Clima activo (lluvia, sol, etc.) y efectos de campo (Grassy Terrain, etc.)
- Modificadores de stats (+/-) y estados (quemado, paralizado, etc.)
- Información oculta del rival (estimada por probabilidad de sets conocidos)
- Terastal: si fue o no usado

**Acciones disponibles:**
- Usar movimiento M sobre target T (4 movimientos × hasta 2 targets = hasta 8)
- Cambiar Pokémon (switch a uno de los banqueados)
- Terastalizar + movimiento
- Protect (caso especial de movimiento)

### 3. Preentrenamiento por imitación (Imitation Learning)
Entrenamiento supervisado sobre replays de jugadores de alto nivel como ground truth. El objetivo es arrancar con una política razonable antes del RL para evitar exploración completamente ciega.

### 4. Entrenamiento por refuerzo (RL)
Fine-tuning con self-play y partidas contra bots de dificultad creciente en Showdown.

**Función de recompensa (componentes):**
```
R = w1 * daño_infligido
  - w2 * daño_recibido
  + w3 * KOs_realizados
  - w4 * KOs_recibidos
  + w5 * movimientos_superefectivos
  + w6 * acciones_defensivas_exitosas   # Protect que evitó daño, switch útil
  + w_win * victoria_final
```
> Los pesos `w1..w6` y `w_win` están por definir y serán parte de la experimentación.

### 5. Evaluación en la Ladder
Despliegue en la ladder real de VGC Reg I contra jugadores humanos. Registro continuo de ELO, win rate y estadísticas de combate.

### 6. Análisis y presentación
Análisis de decisiones, errores sistemáticos y estrategias emergentes. Comparación contra baselines. Presentación final del proyecto.

---

## Métricas de Evaluación

| Métrica | Descripción |
|---|---|
| **ELO** | Ranking en la ladder oficial de Showdown |
| **Win %** | Tasa de victoria general |
| **KO Ratio** | KOs realizados / KOs recibidos por partida |
| **HP Δ** | Diferencial de HP al final de cada batalla |
| **SE %** | Porcentaje de ataques superefectivos sobre el total |
| **Def. Rate** | Frecuencia de acciones defensivas exitosas |

**Baselines de comparación:**
1. Agente aleatorio (acción uniformemente aleatoria)
2. Agente greedy (siempre el movimiento de mayor daño potencial)
3. Jugadores humanos promedio (~1000 ELO)
4. Jugadores competitivos de nivel medio-alto (~1400–1600 ELO)

---

## Variables Sin Definir

- **Algoritmo de RL**: por elegir entre PPO, DQN, A3C u otro según experimentación
- **Pesos de la función de recompensa**: requieren tuning (posible reward shaping o curriculum learning)
- **Arquitectura de la red**: MLP vs. atención vs. híbrida — depende de la representación del estado
- **Infraestructura de cómputo**: GPU dedicada o cloud (Colab / AWS) por confirmar
- **Encoding de información oculta**: cómo representar los sets del rival que no vemos
- **Extensión a team building**: posible si el agente base avanza bien

---

## Fuentes de Datos

| Fuente | URL | Uso |
|---|---|---|
| PokeAPI | https://pokeapi.co | Stats, movimientos, habilidades, tipos |
| Showdown Replays | https://replay.pokemonshowdown.com | Partidas de alto nivel |
| Smogon Usage Stats | https://www.smogon.com/stats/ | Metagame Reg I por nivel de ELO |
| poke-env | https://github.com/hsahovic/poke-env | Interfaz Python ↔ Showdown |

---

## Stack Tecnológico

- **Python 3.11+** con `venv`
- **poke-env** — integración con Pokémon Showdown
- **stable-baselines3** — algoritmos de RL (PPO, DQN, A2C)
- **PyTorch** — backend de la red neuronal
- **Gymnasium** — entorno compatible con los algoritmos de RL
- **pandas / numpy** — procesamiento de datos

---

## Setup

```bash
# 1. Clonar el repo
git clone <repo-url>
cd "laboratorio pokemon showdown"

# 2. Crear entorno virtual e instalar dependencias
bash setup.sh

# 3. Activar entorno
source venv/bin/activate

# 4. Descargar datos de PokeAPI (puede tardar ~10 min)
python data/fetch_data.py

# Opciones disponibles:
python data/fetch_data.py --only pokemon    # solo pokémon
python data/fetch_data.py --only moves      # solo movimientos
python data/fetch_data.py --only types      # solo tabla de tipos
python data/fetch_data.py --only items      # solo objetos
python data/fetch_data.py --only abilities  # solo habilidades
```

---

## Estructura del Proyecto

```
laboratorio pokemon showdown/
├── README.md               # este archivo
├── team.txt                # equipo VGC en formato Pokepaste
├── requirements.txt        # dependencias Python
├── setup.sh                # script de setup del entorno
├── data/
│   ├── fetch_data.py       # script de descarga de PokeAPI
│   └── raw/                # datos descargados (JSON)
│       ├── pokemon.json
│       ├── moves.json
│       ├── type_chart.json
│       ├── items.json
│       └── abilities.json
├── src/                    # código del agente (a desarrollar)
│   ├── env/                # entorno Gymnasium custom
│   ├── agent/              # política y red neuronal
│   └── training/           # scripts de entrenamiento
└── notebooks/              # exploración y análisis
```

---

## Ideas y Notas del Equipo

### Sobre el reward shaping
El reward basado solo en win/loss es muy escaso para VGC — hay que diseñar señales intermedias. Los componentes actuales apuntan en la dirección correcta, pero existe el riesgo de que el agente aprenda a spamear movimientos superefectivos en vez de jugar posicionalmente. Habrá que monitorear esto durante el entrenamiento.

### Sobre información oculta
VGC tiene información oculta: no sabemos los EVs, naturaleza, ni los otros dos movimientos del Pokémon rival. Una opción es codificar la distribución de probabilidad de los sets más comunes (usando Smogon usage stats) como parte del estado.

### Sobre Terastalización
Terastal es una mecánica de un solo uso por batalla que cambia el tipo del Pokémon. Incluirla en el espacio de acciones como una opción separada (Tera + movimiento) es importante porque es una de las decisiones más impactantes del juego.

### Sobre self-play vs. ladder
Entrenar directamente contra humanos en la ladder tiene riesgos éticos (el bot ocupa slots de jugadores reales) y prácticos (la varianza de oponentes es muy alta al principio). La estrategia sugerida es: self-play + bots propios hasta que el agente sea razonablemente competente, y recién después subirlo a la ladder para evaluación.

### Posible extensión: Team Building
Si el tiempo y los recursos lo permiten, explorar si el agente puede aprender a construir equipos (seleccionar Pokémon, moves y EVs) como parte del loop de entrenamiento, usando el ELO como señal de recompensa para el builder.
