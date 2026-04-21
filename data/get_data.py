"""
get_data.py  —  Descarga todos los datos de PokeAPI a data/raw/
Uso: python data/get_data.py
"""

import json
import os
import requests
from tqdm import tqdm

# ── directorio destino ─────────────────────────────────────────────
RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
os.makedirs(RAW, exist_ok=True)

BASE = "https://pokeapi.co/api/v2"
RAW = r"C:\PKMNData"
os.makedirs(RAW, exist_ok=True)

def guardar(nombre, data):
    ruta = os.path.join(RAW, nombre)
    texto = json.dumps(data, indent=2, ensure_ascii=False)
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(texto)
    kb = os.path.getsize(ruta) // 1024
    print(f"  guardado: {nombre}  ({kb} KB)")
     
print(f"\nGuardando en: {RAW}\n")

# ── 1. POKÉMON ─────────────────────────────────────────────────────
print("[1/5] Pokémon (1-1025)...")
pokemon = {}
for pid in tqdm(range(1, 1026), unit="pkmn"):
    try:
        d = requests.get(f"{BASE}/pokemon/{pid}", timeout=10).json()
        learnset = {}
        for m in d.get("moves", []):
            mname = m["move"]["name"]
            methods = {}
            for vd in m.get("version_group_details", []):
                methods[vd["move_learn_method"]["name"]] = vd["level_learned_at"]
            learnset[mname] = methods
        pokemon[d["name"]] = {
            "id":        d["id"],
            "name":      d["name"],
            "types":     [t["type"]["name"] for t in d["types"]],
            "stats":     {s["stat"]["name"]: s["base_stat"] for s in d["stats"]},
            "abilities": [a["ability"]["name"] for a in d["abilities"]],
            "learnset":  learnset,
            "height":    d.get("height", 0),
            "weight":    d.get("weight", 0),
        }
    except Exception as e:
        tqdm.write(f"skip #{pid}: {e}")
guardar("pokemon.json", pokemon)
print(f"  {len(pokemon)} pokémon")

# ── 2. MOVIMIENTOS ─────────────────────────────────────────────────
print("\n[2/5] Movimientos...")
moves = {}
for item in tqdm(requests.get(f"{BASE}/move?limit=2000", timeout=10).json()["results"], unit="move"):
    try:
        d = requests.get(item["url"], timeout=10).json()
        moves[d["name"]] = {
            "id":       d["id"],
            "name":     d["name"],
            "type":     d["type"]["name"],
            "power":    d.get("power") or 0,
            "accuracy": d.get("accuracy") or 100,
            "pp":       d.get("pp", 0),
            "priority": d.get("priority", 0),
            "category": d["damage_class"]["name"],
            "target":   d["target"]["name"],
        }
    except Exception as e:
        tqdm.write(f"skip {item['name']}: {e}")
guardar("moves.json", moves)
print(f"  {len(moves)} movimientos")

# ── 3. HABILIDADES ─────────────────────────────────────────────────
print("\n[3/5] Habilidades...")
abilities = {}
for item in tqdm(requests.get(f"{BASE}/ability?limit=400", timeout=10).json()["results"], unit="ab"):
    try:
        d = requests.get(item["url"], timeout=10).json()
        en = next((e["effect"] for e in d.get("effect_entries", []) if e["language"]["name"] == "en"), "")
        abilities[d["name"]] = {"id": d["id"], "name": d["name"], "effect": en}
    except Exception as e:
        tqdm.write(f"skip {item['name']}: {e}")
guardar("abilities.json", abilities)
print(f"  {len(abilities)} habilidades")

# ── 4. TIPOS ───────────────────────────────────────────────────────
print("\n[4/5] Tipos...")
type_chart = {}
for item in tqdm(requests.get(f"{BASE}/type?limit=30", timeout=10).json()["results"], unit="type"):
    try:
        d   = requests.get(item["url"], timeout=10).json()
        rel = d["damage_relations"]
        row = {}
        for t in rel.get("double_damage_to", []):  row[t["name"]] = 2.0
        for t in rel.get("half_damage_to",   []):  row[t["name"]] = 0.5
        for t in rel.get("no_damage_to",     []):  row[t["name"]] = 0.0
        type_chart[d["name"]] = row
    except Exception as e:
        tqdm.write(f"skip {item['name']}: {e}")
guardar("type_chart.json", type_chart)
print(f"  {len(type_chart)} tipos")

# ── 5. NATURES ─────────────────────────────────────────────────────
print("\n[5/5] Natures...")
natures = {
    "hardy":   {},  "docile":  {},  "serious": {},  "bashful": {},  "quirky":  {},
    "lonely":  {"attack": 1.1, "defense": 0.9},
    "brave":   {"attack": 1.1, "speed": 0.9},
    "adamant": {"attack": 1.1, "special-attack": 0.9},
    "naughty": {"attack": 1.1, "special-defense": 0.9},
    "bold":    {"defense": 1.1, "attack": 0.9},
    "relaxed": {"defense": 1.1, "speed": 0.9},
    "impish":  {"defense": 1.1, "special-attack": 0.9},
    "lax":     {"defense": 1.1, "special-defense": 0.9},
    "timid":   {"speed": 1.1, "attack": 0.9},
    "hasty":   {"speed": 1.1, "defense": 0.9},
    "jolly":   {"speed": 1.1, "special-attack": 0.9},
    "naive":   {"speed": 1.1, "special-defense": 0.9},
    "modest":  {"special-attack": 1.1, "attack": 0.9},
    "mild":    {"special-attack": 1.1, "defense": 0.9},
    "quiet":   {"special-attack": 1.1, "speed": 0.9},
    "rash":    {"special-attack": 1.1, "special-defense": 0.9},
    "calm":    {"special-defense": 1.1, "attack": 0.9},
    "gentle":  {"special-defense": 1.1, "defense": 0.9},
    "sassy":   {"special-defense": 1.1, "speed": 0.9},
    "careful": {"special-defense": 1.1, "special-attack": 0.9},
}
guardar("natures.json", natures)
print(f"  {len(natures)} natures")

# ── RESUMEN ────────────────────────────────────────────────────────
print("\n" + "=" * 50)
for fname in ["pokemon.json", "moves.json", "abilities.json", "type_chart.json", "natures.json"]:
    ruta = os.path.join(RAW, fname)
    if os.path.exists(ruta):
        print(f"  OK  {fname:22} {os.path.getsize(ruta)//1024:6} KB")
    else:
        print(f"  FALTA  {fname}")
print("=" * 50)
