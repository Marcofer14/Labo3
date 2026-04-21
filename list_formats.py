"""
list_formats.py — Lista los formatos VGC disponibles en el servidor local.
Uso: python list_formats.py
"""

import asyncio
import os
import re
import argparse
import websockets


async def get_formats(server_host: str, show_raw: bool = False) -> list[dict]:
    uri = f"ws://{server_host}/showdown/websocket"
    print(f"Conectando a {uri} ...")

    raw_messages = []
    try:
        async with websockets.connect(uri, open_timeout=8) as ws:
            # Leer mensajes durante 6 segundos
            end = asyncio.get_event_loop().time() + 6
            while asyncio.get_event_loop().time() < end:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=2)
                    raw_messages.append(msg)
                    if "|formats|" in msg:
                        break
                except asyncio.TimeoutError:
                    break
    except OSError as e:
        print(f"\n❌ No se pudo conectar: {e}")
        return []

    if show_raw:
        print("\n─── MENSAJES RAW ───")
        for i, m in enumerate(raw_messages):
            print(f"\n[Mensaje {i}]:\n{m[:2000]}")
        print("─── FIN RAW ───\n")

    # Buscar el mensaje que contiene los formatos
    formats_msg = next((m for m in raw_messages if "|formats|" in m), None)
    if not formats_msg:
        print("⚠ No se recibió el mensaje de formatos.")
        print("  Mensajes recibidos:", len(raw_messages))
        if raw_messages:
            print("\n─── PRIMER MENSAJE ───")
            print(raw_messages[0][:1000])
        return []

    # ── Parsear el mensaje de formatos ───────────────────────────
    # Formato real del mensaje (separado por |):
    #   |formats|,LL|,1|Nombre Sección|DisplayName,flags|DisplayName,flags|...
    #
    # Cada token puede ser:
    #   - ",LL" o ",1" etc.  → flags globales (ignorar)
    #   - "Sección Name"     → nombre de sección (sin corchetes, sin coma al inicio)
    #   - "[Gen X] Nombre,5c" → formato: DisplayName ANTES de la coma, flags después
    #
    # El ID interno de Showdown = display name en minúsculas sin nada especial
    # Ej: "[Gen 9] VGC 2026 Reg I" → "gen9vgc2026regi"

    formats_start = formats_msg.index("|formats|") + len("|formats|")
    content = formats_msg[formats_start:]
    tokens = content.split("|")

    formats = []
    current_section = ""

    for token in tokens:
        token = token.strip()
        if not token:
            continue

        # Separar nombre del formato de sus flags: "DisplayName,5c" → ("DisplayName", "5c")
        # Los flags son siempre alfanuméricos cortos al final después de una coma
        name_part = token.split(",")[0].strip()

        # Detectar sección: empieza con letra mayúscula, no tiene corchetes
        if name_part and not name_part.startswith("[") and re.match(r'^[A-Z]', name_part):
            current_section = name_part
            continue

        # Detectar formato: el nombre tiene corchetes "[Gen X]"
        if "[Gen" in name_part:
            display_name = name_part
            # ID = lowercase, solo letras y números
            fmt_id = re.sub(r'[^a-z0-9]', '', display_name.lower())
            formats.append({
                "name":    display_name,
                "id":      fmt_id,
                "section": current_section,
            })

    return formats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default=os.environ.get("SHOWDOWN_SERVER", "localhost:8000"))
    parser.add_argument("--all", action="store_true", help="Mostrar todos los formatos")
    parser.add_argument("--raw", action="store_true", help="Mostrar mensajes WebSocket raw")
    args = parser.parse_args()

    formats = asyncio.run(get_formats(args.server, show_raw=args.raw))

    if not formats:
        print("\nNo se encontraron formatos. Probá con --raw para ver los mensajes crudos:")
        print(f"  python list_formats.py --raw")
        return

    vgc_kw = ["vgc", "doubles", "double"]
    vgc = [f for f in formats if any(k in f["name"].lower() for k in vgc_kw)]
    other = [f for f in formats if f not in vgc]

    print(f"\n✓ {len(formats)} formatos encontrados\n")

    if vgc:
        print("═" * 55)
        print("  FORMATOS VGC / DOBLES")
        print("═" * 55)
        for f in vgc:
            print(f"  --format {f['id']}")
            print(f"           {f['name']}")
            print()
    else:
        print("⚠ No se encontraron formatos VGC. Mostrando todos:\n")
        args.all = True

    if args.all:
        print("─" * 55)
        print("  TODOS LOS FORMATOS")
        print("─" * 55)
        for f in other:
            print(f"  {f['id']:<40} {f['name']}")

    if vgc:
        print("─" * 55)
        best = vgc[0]['id']
        print(f"  Usar con: python battle.py --format {best}")
        print("─" * 55)


if __name__ == "__main__":
    main()
