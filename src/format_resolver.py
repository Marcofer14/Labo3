"""
src/format_resolver.py

Resuelve qué formato de Pokémon Showdown usar.

Prioridad:
1. Formato pasado por comando, por ejemplo:
   --format gen9vgc2026regi

2. Variable de entorno:
   VGC_FORMAT=gen9vgc2026regi

3. Fallback automático:
   gen9vgc2026regi
"""

from __future__ import annotations

import os


DEFAULT_VGC_FORMAT = "gen9vgc2026regi"


def resolve_format(cli_format: str | None = None) -> str:
    """
    Decide qué formato usar.

    Si cli_format viene informado, se respeta.
    Si no, usa VGC_FORMAT desde variables de entorno.
    Si tampoco existe, usa DEFAULT_VGC_FORMAT.
    """

    if cli_format:
        return cli_format

    env_format = os.environ.get("VGC_FORMAT")
    if env_format:
        return env_format

    return DEFAULT_VGC_FORMAT