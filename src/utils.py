"""
utils.py
─────────────────────────────────────────────────────────────────
Módulo base de utilidades del VGC Bot. Provee:

  1. DATA_DIR       — ubicación de los JSON descargados de PokeAPI
  2. Loaders        — carga pokemon, moves, type_chart, items, abilities
  3. Team parser    — convierte team.txt (pokepaste) en dict estructurado
  4. Stat calc      — calcula stats reales de nivel 50 con EVs/IVs/Nature
  5. Helpers        — STAB, efectividad de tipos, búsquedas rápidas

Uso básico:
    from src.utils import load_all_data, parse_team, get_effectiveness

    data = load_all_data()
    team = parse_team("team.txt")
    mult = get_effectiveness("water", ["fire", "rock"], data["type_chart"])
"""

import json
import os
import re
from pathlib import Path
from typing import Optional

# ── Ubicación de los datos ────────────────────────────────────────
# Busca los JSON en este orden:
#   1. Variable de entorno VGC_DATA_DIR
#   2. Escritorio de OneDrive (donde los descargamos)
#   3. data/raw/ relativo al root del proyecto

def _find_data_dir() -> Path:
    # 1. Variable de entorno
    env = os.environ.get("VGC_DATA_DIR")
    if env and Path(env).exists():
        return Path(env)

    # 2. Escritorio de OneDrive (ruta del usuario actual)
    onedrive_desktop = Path.home() / "OneDrive" / "Escritorio" / "vgc_bot_data"
    if onedrive_desktop.exists():
        return onedrive_desktop

    # 3. Downloads
    downloads = Path.home() / "Downloads" / "vgc_bot_data"
    if downloads.exists():
        return downloads

    # 4. Desktop (sin OneDrive)
    desktop = Path.home() / "Desktop" / "vgc_bot_data"
    if desktop.exists():
        return desktop

    # 5. Relativo al proyecto
    project_raw = Path(__file__).resolve().parent.parent / "data" / "raw"
    if project_raw.exists():
        return project_raw

    raise FileNotFoundError(
        "No se encontraron los datos descargados de PokeAPI.\n"
        "Opciones:\n"
        "  a) Corré data/fetch_data.py primero\n"
        "  b) Setear la variable de entorno VGC_DATA_DIR con la ruta correcta"
    )

DATA_DIR = _find_data_dir()

# ── Loaders ───────────────────────────────────────────────────────

def _load_json(filename: str) -> dict:
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def load_pokemon_data() -> dict:
    """
    Retorna dict keyed por ID (str): {
        "id": 382, "name": "kyogre",
        "types": ["water"],
        "stats": {"hp": 100, "attack": 100, ...},
        "abilities": [{"name": "drizzle", "hidden": False, "slot": 1}]
    }
    """
    return _load_json("pokemon.json")

def load_moves_data() -> dict:
    """
    Retorna dict keyed por nombre: {
        "water-spout": {
            "type": "water", "category": "special",
            "power": 150, "accuracy": 100, "priority": 0,
            "target": "all-opponents", "pp": 5, ...
        }
    }
    """
    return _load_json("moves.json")

def load_type_chart() -> dict:
    """
    Retorna dict [attacking_type][defending_type] = multiplicador float
    Ejemplo: chart["water"]["fire"] → 2.0
    """
    return _load_json("type_chart.json")

def load_items_data() -> dict:
    """Retorna dict keyed por nombre de item."""
    return _load_json("items.json")

def load_abilities_data() -> dict:
    """Retorna dict keyed por nombre de habilidad."""
    return _load_json("abilities.json")

def load_all_data() -> dict:
    """
    Carga todos los archivos y los retorna en un único dict.
    Uso: data = load_all_data(); data["moves"]["surf"]
    """
    return {
        "pokemon":    load_pokemon_data(),
        "moves":      load_moves_data(),
        "type_chart": load_type_chart(),
        "items":      load_items_data(),
        "abilities":  load_abilities_data(),
    }

# ── Team parser (formato Pokepaste) ──────────────────────────────

# Tabla de naturalezas → qué stat aumenta (+10%) y cuál baja (-10%)
NATURES: dict[str, tuple[str, str]] = {
    "hardy":   ("attack", "attack"),        # neutro
    "lonely":  ("attack", "defense"),
    "brave":   ("attack", "speed"),
    "adamant": ("attack", "special-attack"),
    "naughty": ("attack", "special-defense"),
    "bold":    ("defense", "attack"),
    "docile":  ("defense", "defense"),      # neutro
    "relaxed": ("defense", "speed"),
    "impish":  ("defense", "special-attack"),
    "lax":     ("defense", "special-defense"),
    "timid":   ("speed", "attack"),
    "hasty":   ("speed", "defense"),
    "serious": ("speed", "speed"),          # neutro
    "jolly":   ("speed", "special-attack"),
    "naive":   ("speed", "special-defense"),
    "modest":  ("special-attack", "attack"),
    "mild":    ("special-attack", "defense"),
    "quiet":   ("special-attack", "speed"),
    "bashful": ("special-attack", "special-attack"),  # neutro
    "rash":    ("special-attack", "special-defense"),
    "calm":    ("special-defense", "attack"),
    "gentle":  ("special-defense", "defense"),
    "sassy":   ("special-defense", "speed"),
    "careful": ("special-defense", "special-attack"),
    "quirky":  ("special-defense", "special-defense"),  # neutro
}

# Mapeo de abreviaciones de stats en pokepaste → nombres de PokeAPI
STAT_ABBR: dict[str, str] = {
    "HP":  "hp",
    "Atk": "attack",
    "Def": "defense",
    "SpA": "special-attack",
    "SpD": "special-defense",
    "Spe": "speed",
}

def _parse_ev_string(ev_str: str) -> dict[str, int]:
    """
    Parsea '140 HP / 68 Def / 156 SpA / 4 SpD / 140 Spe'
    → {"hp": 140, "defense": 68, "special-attack": 156, ...}
    """
    evs = {v: 0 for v in STAT_ABBR.values()}
    for part in ev_str.split("/"):
        part = part.strip()
        match = re.match(r"(\d+)\s+(\w+)", part)
        if match:
            val, abbr = int(match.group(1)), match.group(2)
            if abbr in STAT_ABBR:
                evs[STAT_ABBR[abbr]] = val
    return evs

def _parse_iv_string(iv_str: str) -> dict[str, int]:
    """
    Parsea '0 Atk' o '31 Spe / 0 Atk'
    → {"attack": 0} (resto asume 31)
    """
    ivs = {v: 31 for v in STAT_ABBR.values()}
    for part in iv_str.split("/"):
        part = part.strip()
        match = re.match(r"(\d+)\s+(\w+)", part)
        if match:
            val, abbr = int(match.group(1)), match.group(2)
            if abbr in STAT_ABBR:
                ivs[STAT_ABBR[abbr]] = val
    return ivs

def parse_team(team_path: str | Path) -> list[dict]:
    """
    Parsea un archivo en formato Pokepaste y retorna una lista de dicts.

    Retorna:
        [
            {
                "name":      "kyogre",
                "item":      "mystic-water",
                "ability":   "drizzle",
                "level":     50,
                "tera_type": "grass",
                "nature":    "modest",
                "evs":       {"hp": 140, "attack": 0, "defense": 68, ...},
                "ivs":       {"hp": 31, "attack": 0, ...},
                "moves":     ["water-spout", "origin-pulse", "ice-beam", "protect"],
            },
            ...
        ]
    """
    with open(team_path, encoding="utf-8") as f:
        content = f.read()

    team = []
    # Separar cada Pokémon por línea en blanco
    blocks = [b.strip() for b in content.strip().split("\n\n") if b.strip()]

    for block in blocks:
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        pokemon: dict = {
            "name":      "",
            "item":      None,
            "ability":   "",
            "level":     50,
            "tera_type": None,
            "nature":    "",
            "evs":       {v: 0 for v in STAT_ABBR.values()},
            "ivs":       {v: 31 for v in STAT_ABBR.values()},
            "moves":     [],
        }

        for line in lines:
            # Primera línea: "Nombre @ Item" o solo "Nombre"
            if not pokemon["name"]:
                if "@" in line:
                    parts = line.split("@")
                    raw_name = parts[0].strip()
                    raw_item = parts[1].strip()
                    # Normalizar a formato PokeAPI (lowercase, espacios → guiones)
                    pokemon["name"] = _normalize(raw_name)
                    pokemon["item"] = _normalize(raw_item)
                else:
                    pokemon["name"] = _normalize(line)
                continue

            if line.startswith("Ability:"):
                pokemon["ability"] = _normalize(line.replace("Ability:", "").strip())
            elif line.startswith("Level:"):
                pokemon["level"] = int(line.replace("Level:", "").strip())
            elif line.startswith("Tera Type:"):
                pokemon["tera_type"] = line.replace("Tera Type:", "").strip().lower()
            elif line.startswith("EVs:"):
                pokemon["evs"] = _parse_ev_string(line.replace("EVs:", "").strip())
            elif line.startswith("IVs:"):
                pokemon["ivs"] = _parse_iv_string(line.replace("IVs:", "").strip())
            elif line.endswith("Nature"):
                pokemon["nature"] = line.replace("Nature", "").strip().lower()
            elif line.startswith("- "):
                move = _normalize(line[2:].strip())
                pokemon["moves"].append(move)

        if pokemon["name"]:
            team.append(pokemon)

    return team

def _normalize(name: str) -> str:
    """
    Convierte nombres de Pokepaste al formato PokeAPI:
    'Water Spout'     → 'water-spout'
    'Calyrex-Shadow'  → 'calyrex-shadow'
    'Urshifu-Rapid-Strike' → 'urshifu-rapid-strike'
    """
    return name.lower().strip().replace(" ", "-")

# ── Cálculo de stats reales (nivel 50, EVs, IVs, naturaleza) ─────

def calc_stat(
    stat_name: str,
    base: int,
    ev: int,
    iv: int,
    nature: str,
    level: int = 50,
) -> int:
    """
    Calcula el stat final de un Pokémon según las fórmulas oficiales de gen 3+.

    Para HP:
        floor((2*base + iv + floor(ev/4)) * level/100 + level + 10)
    Para los demás:
        floor((floor((2*base + iv + floor(ev/4)) * level/100) + 5) * nature_mult)

    Args:
        stat_name: nombre en formato PokeAPI ('hp', 'attack', 'speed', etc.)
        base: stat base del Pokémon
        ev: EVs en ese stat (0–252)
        iv: IVs en ese stat (0–31)
        nature: nombre de la naturaleza en minúsculas ('modest', 'jolly', etc.)
        level: nivel del Pokémon (default 50 para VGC)

    Returns:
        Stat final calculado (int)
    """
    ev_contrib = ev // 4
    base_calc  = (2 * base + iv + ev_contrib) * level // 100

    if stat_name == "hp":
        return base_calc + level + 10

    nature_mult = _nature_multiplier(nature, stat_name)
    return int((base_calc + 5) * nature_mult)

def _nature_multiplier(nature: str, stat_name: str) -> float:
    """Retorna el multiplicador de naturaleza (0.9, 1.0 o 1.1) para un stat."""
    if nature not in NATURES:
        return 1.0
    boosts, drops = NATURES[nature]
    if boosts == drops:
        return 1.0   # naturaleza neutra
    if stat_name == boosts:
        return 1.1
    if stat_name == drops:
        return 0.9
    return 1.0

def calc_all_stats(pokemon_entry: dict, pokemon_data: dict) -> dict[str, int]:
    """
    Calcula los 6 stats finales de un Pokémon del equipo.

    Args:
        pokemon_entry: dict retornado por parse_team() para un Pokémon
        pokemon_data:  datos de PokeAPI (load_pokemon_data())

    Returns:
        {"hp": 175, "attack": 80, "defense": 95, "special-attack": 150,
         "special-defense": 110, "speed": 115}
    """
    # Buscar por nombre (PokeAPI usa nombres en minúsculas con guiones)
    poke_info = _find_pokemon_by_name(pokemon_entry["name"], pokemon_data)
    if poke_info is None:
        raise ValueError(f"Pokémon '{pokemon_entry['name']}' no encontrado en los datos")

    base_stats = poke_info["stats"]
    result = {}

    for stat_name, base in base_stats.items():
        ev  = pokemon_entry["evs"].get(stat_name, 0)
        iv  = pokemon_entry["ivs"].get(stat_name, 31)
        result[stat_name] = calc_stat(
            stat_name, base, ev, iv, pokemon_entry["nature"], pokemon_entry["level"]
        )
    return result

# ── Helpers de tipo y efectividad ────────────────────────────────

def get_effectiveness(
    attacking_type: str,
    defending_types: list[str],
    type_chart: dict,
) -> float:
    """
    Calcula el multiplicador de efectividad total de un tipo de ataque
    contra uno o dos tipos defensivos.

    Args:
        attacking_type: tipo del movimiento en minúsculas ('water', 'fire', etc.)
        defending_types: lista de tipos del defensor (['fire', 'rock'])
        type_chart: dict retornado por load_type_chart()

    Returns:
        Multiplicador total (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)

    Ejemplo:
        get_effectiveness("water", ["fire", "rock"], chart) → 4.0
        get_effectiveness("normal", ["ghost"], chart) → 0.0
    """
    chart = type_chart.get(attacking_type, {})
    mult  = 1.0
    for def_type in defending_types:
        mult *= chart.get(def_type, 1.0)
    return mult

def is_stab(move_type: str, pokemon_types: list[str]) -> bool:
    """True si el movimiento es del mismo tipo que el Pokémon (STAB)."""
    return move_type in pokemon_types

def get_nature_multiplier(nature: str, stat_name: str) -> float:
    """Exporta el multiplicador de naturaleza para uso externo."""
    return _nature_multiplier(nature, stat_name)

# ── Overrides para formas alternativas no incluidas en el dataset ──
# PokeAPI solo descargó IDs 1-1025 (formas base). Estos overrides
# corrigen las formas alternativas usadas en VGC competitivo.

FORM_OVERRIDES: dict[str, dict] = {
    # Calyrex-Shadow Rider (Psychic/Ghost, SpA=145, Spe=150)
    "calyrex-shadow": {
        "id": 898, "name": "calyrex-shadow",
        "types": ["psychic", "ghost"],
        "stats": {
            "hp": 100, "attack": 85, "defense": 80,
            "special-attack": 145, "special-defense": 80, "speed": 150,
        },
        "abilities": ["as-one"],
    },
    # Calyrex-Ice Rider (Psychic/Ice, Atk=165, Spe=50)
    "calyrex-ice": {
        "id": 898, "name": "calyrex-ice",
        "types": ["psychic", "ice"],
        "stats": {
            "hp": 100, "attack": 165, "defense": 150,
            "special-attack": 85, "special-defense": 130, "speed": 50,
        },
        "abilities": ["as-one"],
    },
    # Urshifu-Rapid-Strike (Fighting/Water)
    "urshifu-rapid-strike": {
        "id": 892, "name": "urshifu-rapid-strike",
        "types": ["fighting", "water"],
        "stats": {
            "hp": 100, "attack": 130, "defense": 100,
            "special-attack": 63, "special-defense": 60, "speed": 97,
        },
        "abilities": ["unseen-fist"],
    },
    # Kyogre (base está bien pero añadimos por completitud)
    # Necrozma-Dawn-Wings, Necrozma-Dusk-Mane, etc. por si acaso
    "incineroar": None,   # sentinel — usar la del dataset
}


# ── Búsquedas ─────────────────────────────────────────────────────

def _find_pokemon_by_name(name: str, pokemon_data: dict) -> Optional[dict]:
    """
    Busca un Pokémon en el dataset por nombre.
    Primero revisa FORM_OVERRIDES para formas alternativas que no
    están en el dataset de PokeAPI (IDs 1-1025).
    """
    name = name.lower()

    # 1. Overrides explícitos para formas alternativas
    if name in FORM_OVERRIDES:
        override = FORM_OVERRIDES[name]
        if override is not None:   # None = sentinel "usar dataset normal"
            return override

    # 2. Búsqueda normal en el dataset
    for poke in pokemon_data.values():
        if poke["name"] == name:
            return poke

    # 3. Fallback: intentar con el nombre base (sin el sufijo de forma)
    #    ej: "kyogre-primal" → intentar "kyogre"
    base_name = name.split("-")[0]
    if base_name != name:
        for poke in pokemon_data.values():
            if poke["name"] == base_name:
                # Retornar una copia con el nombre correcto
                return dict(poke, name=name)

    return None

def get_pokemon(name: str, pokemon_data: dict) -> Optional[dict]:
    """Retorna el dict completo de un Pokémon buscando por nombre."""
    return _find_pokemon_by_name(name.lower(), pokemon_data)

def get_move(name: str, moves_data: dict) -> Optional[dict]:
    """Retorna el dict de un movimiento buscando por nombre."""
    return moves_data.get(name.lower())

def get_item(name: str, items_data: dict) -> Optional[dict]:
    """Retorna el dict de un objeto buscando por nombre."""
    return items_data.get(name.lower())

def get_ability(name: str, abilities_data: dict) -> Optional[dict]:
    """Retorna el dict de una habilidad buscando por nombre."""
    return abilities_data.get(name.lower())

# ── Quick summary del equipo ──────────────────────────────────────

def summarize_team(team: list[dict], pokemon_data: dict) -> None:
    """
    Imprime un resumen legible del equipo con sus stats calculados.
    Útil para verificar que el parser y calc_stat funcionan bien.
    """
    print("═" * 60)
    print("  Resumen del equipo")
    print("═" * 60)
    for p in team:
        poke_info = _find_pokemon_by_name(p["name"], pokemon_data)
        types_str = "/".join(poke_info["types"]) if poke_info else "???"
        stats     = calc_all_stats(p, pokemon_data) if poke_info else {}
        print(f"\n  {p['name'].upper()}  ({types_str})")
        print(f"    Item:    {p['item']}  |  Ability: {p['ability']}")
        print(f"    Nature:  {p['nature']}  |  Tera: {p['tera_type']}")
        print(f"    Moves:   {', '.join(p['moves'])}")
        if stats:
            stat_line = "  ".join(
                f"{k[:3].upper()}: {v}" for k, v in stats.items()
            )
            print(f"    Stats:   {stat_line}")
    print("\n" + "═" * 60)


# ── Entry point de prueba ─────────────────────────────────────────

if __name__ == "__main__":
    print(f"Cargando datos desde: {DATA_DIR}\n")

    data      = load_all_data()
    team_path = Path(__file__).resolve().parent.parent / "team.txt"
    team      = parse_team(team_path)

    print(f"Pokémon cargados:    {len(data['pokemon'])}")
    print(f"Movimientos cargados: {len(data['moves'])}")
    print(f"Items cargados:      {len(data['items'])}")
    print(f"Habilidades cargadas: {len(data['abilities'])}")
    print(f"Pokémon en el equipo: {len(team)}")

    summarize_team(team, data["pokemon"])

    # Test de efectividad
    chart = data["type_chart"]
    print("\nTest de efectividad:")
    print(f"  Agua → Fuego/Roca:  x{get_effectiveness('water', ['fire', 'rock'], chart)}")
    print(f"  Normal → Fantasma:  x{get_effectiveness('normal', ['ghost'], chart)}")
    print(f"  Hielo → Dragón:     x{get_effectiveness('ice', ['dragon'], chart)}")
    print(f"  Lucha → Normal:     x{get_effectiveness('fighting', ['normal'], chart)}")
