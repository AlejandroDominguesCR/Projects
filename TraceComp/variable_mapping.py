from __future__ import annotations
import json
import os
import re
from typing import Dict, List, Sequence, Optional, Tuple

import numpy as np

# === CANONICAL MAPPING START ===
# regex patterns applied on slugified header names
ALIAS_PATTERNS: Dict[str, List[str]] = {
    "speed":    [r"^carspeed$", r"^vcar$", r"^speed$", r"^veh.*speed"],
    "lat_acc":  [r"^ay($|_)", r"^gextaccy$", r"^lat.*acc", r"^lat.*g"],
    "long_acc": [r"^ax($|_)", r"^glong$", r"^long.*acc"],
    "brake":    [r"^brake(_press)?$", r"^braketotal$", r"^brake.*"],
    "throttle": [r"^throttle$", r"^rthrottlepedal$", r"^gas.*", r".*pedal.*throttle"],
}

ALIASES_FILE = os.path.join(os.path.dirname(__file__), "variable_aliases.json")


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def to_float(val, default: float = np.nan) -> float:
    try:
        return float(str(val).strip())
    except Exception:
        return default


def load_alias_db(path: str = ALIASES_FILE) -> Dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"global": {}, "per_file": {}}


def save_alias_db(db: Dict, path: str = ALIASES_FILE) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2)


def suggest_mapping(headers: Sequence[str], alias_db: Dict) -> Dict[str, List[str]]:
    slugs = {h: slug(h) for h in headers}
    result: Dict[str, List[str]] = {r: [] for r in ALIAS_PATTERNS}
    for role, patterns in ALIAS_PATTERNS.items():
        for h, s in slugs.items():
            for p in patterns:
                if re.search(p, s):
                    result[role].append(h)
                    break
    for role, names in alias_db.get("global", {}).items():
        if role in result:
            ordered = [h for h in names if h in headers]
            result[role] = ordered + [h for h in result[role] if h not in ordered]
    return result


def normalize_data(
    headers: Sequence[str],
    data_dict: Dict[str, Sequence],
    units_map: Optional[Dict[str, str]] = None,
    alias_db: Optional[Dict] = None,
    overrides: Optional[Dict[str, str]] = None,
    platform: Optional[str] = None,
) -> Tuple[Dict[str, Optional[str]], Dict[str, List[float]], List[str]]:
    alias_db = alias_db or {}
    suggested = suggest_mapping(headers, alias_db)
    mapping: Dict[str, Optional[str]] = {}
    for role in ALIAS_PATTERNS:
        header = None
        if overrides and overrides.get(role):
            header = overrides[role]
        elif suggested.get(role):
            header = suggested[role][0]
        if header not in headers:
            header = None
        mapping[role] = header
    datos_can: Dict[str, List[float]] = {}
    missing: List[str] = []
    for role, head in mapping.items():
        if head and head in data_dict:
            vals = [to_float(v) for v in data_dict[head]]
            unit_hint = units_map.get(head, "").lower() if units_map else ""
            if role == "speed" and (platform == "canopy" or "m/s" in unit_hint):
                vals = [v * 3.6 for v in vals]
            datos_can[role] = vals
        else:
            missing.append(role)
    # === CANONICAL MAPPING END ===
    return mapping, datos_can, missing