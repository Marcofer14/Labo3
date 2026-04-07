# Descargando Datos Completos para el VGC Bot

## El Problema

Actualmente los datos JSON son **mínimos** (solo 6 pokémon, 17 movimientos, 14 habilidades). Esto es insuficiente para entrenar una IA real.

## Qué Descargará

El script `data/fetch_complete_data.py` descarga:

- **Pokémon Gen 9**: Todos los 1025 pokémon con:
  - Stats base (HP, Ataque, Defensa, etc.)
  - Tipos
  - Habilidades
  - **LEARNSET COMPLETO** ← Movimientos que cada pokémon puede aprender
  - Altura y peso

- **Movimientos**: ~900 movimientos de Gen 9 con:
  - Tipo
  - Potencia y precisión
  - Categoría (Físico, Especial, Estado)
  - PP
  - Prioridad
  - Efecto

- **Habilidades**: ~300 habilidades con efectos

- **Items**: 45+ items relevantes para VGC

- **Tabla de Efectividades**: Matriz completa tipo vs tipo

- **Nature Multipliers**: Los 25 nature con sus bonificaciones/penalizaciones

## Cómo Ejecutar (EN TU MÁQUINA WINDOWS)

### Opción 1: Script Automático (RECOMENDADO)

```bash
# 1. Ve a la carpeta del proyecto
cd "C:\Users\Lenovo\OneDrive\Documentos\Claude\Projects\laboratorio pokemon showdown"

# 2. Activa tu entorno virtual (si lo tienes)
venv\Scripts\activate

# 3. Asegúrate que pip tiene las dependencias
pip install requests tqdm

# 4. Ejecuta el script
python data/fetch_complete_data.py
```

**Tiempo estimado**: ~5–10 minutos (depende de tu conexión a internet)

### Opción 2: Paso a Paso (si algo falla)

```bash
# Solo descarga pokémon (si falla)
python -c "from data.fetch_complete_data import fetch_pokemon_complete; fetch_pokemon_complete()"

# Solo movimientos
python -c "from data.fetch_complete_data import fetch_moves_complete; fetch_moves_complete()"

# Etc.
```

## Qué Hace Exactamente

1. **Conecta a PokeAPI** (`https://pokeapi.co/api/v2`)
2. **Descarga todos los pokémon** (IDs 1–1025) con su learnset
3. **Descarga ~900 movimientos** de Gen 9
4. **Descarga ~300 habilidades**
5. **Descarga 45+ items VGC**
6. **Construye tabla de efectividades** de tipos
7. **Guarda todo en JSON** en `data/raw/`

## Archivos Generados

```
data/raw/
├── pokemon.json          (~2.5 MB) — Pokémon + learnset
├── moves.json            (~150 KB) — Movimientos
├── abilities.json        (~50 KB)  — Habilidades
├── items.json            (~20 KB)  — Items
├── type_chart.json       (~3 KB)   — Tabla de tipos
└── natures.json          (~1 KB)   — Nature multipliers
```

## Optimizaciones Incluidas

- ✓ **Caché implícito**: Si la descarga falla a mitad, puedes volver a ejecutar
- ✓ **Barra de progreso**: tqdm te muestra cuánto falta
- ✓ **Manejo de errores**: Si falla un pokémon, continúa con el siguiente
- ✓ **Tabla manual de fallback**: Si PokeAPI falla con tipos, usa tabla hardcodeada

## Problemas Comunes

### "ModuleNotFoundError: No module named 'requests'"
```bash
pip install requests tqdm --break-system-packages
```

### "Connection refused / Timeout"
Significa que PokeAPI está lento. Espera unos minutos e intenta de nuevo:
```bash
python data/fetch_complete_data.py
```

### "¿Cuánto pesa esto?"
Total: ~2.7 MB en JSON. Muy manejable.

## Próximo Paso

Una vez descargues, actualiza `src/utils.py` para que use el **learnset completo**:

```python
# En utils.py, parse_team() puede validar que los movimientos del team
# están en el learnset del pokémon
if move not in pokemon_learnset[poke_name]:
    print(f"⚠ Warning: {poke_name} no puede aprender {move} en Gen 9")
```

## ¿Necesitas Ayuda?

1. Corre el script y pasame la salida completa
2. Si hay errores, te ayudo a debuggear
3. Una vez descargado, podemos ajustar `utils.py` y `damage_calc.py` para usar los nuevos datos
