"""Build the standalone kernel using the installed CP Only sight/targeting code.

Original game/CP sources remain read-only and generated copies stay in build/.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools._build_common import compile_shared, extract_cpp_function


def build(cp, game):
    out = ROOT / "build"
    out.mkdir(exist_ok=True)
    sdk = cp / "CvGameCoreDLL_Expansion2"
    mod = cp / "(1) Community Patch"
    sources = {}

    def read(path):
        data = path.read_bytes()
        sources[
            path.relative_to(cp if path.is_relative_to(cp) else game).as_posix()
        ] = hashlib.sha256(data).hexdigest()
        return data.decode("utf-8-sig", errors="replace").replace("\r\n", "\n")

    plot = read(sdk / "CvPlot.cpp")
    read(sdk / "CvPlot.h")  # getVisibilityCount: CORE_DELAYED_VISIBILITY.
    targeting = read(sdk / "CvTargeting.cpp")
    sight = extract_cpp_function(plot, "void CvPlot::changeAdjacentSight(")
    ray = targeting[targeting.index("static int ms_LookBackX") :]
    (out / "cp_geometry.inc").write_text(sight + "\n" + ray, encoding="utf-8")
    for name in (
        "CvUnit.cpp",
        "CvUnitCombat.cpp",
        "CvUnitMovement.cpp",
        "CvPlayer.cpp",
        "CvUnitMission.cpp",
    ):
        read(sdk / name)

    # The selected CP-only profile retains BNW unit/terrain data. Record every
    # XML layer used, plus CP's database changes and option values for auditing.
    defines = {}
    layers = [game / "Assets/Gameplay/XML"] + [
        game / f"Assets/DLC/{exp}/Gameplay/XML" for exp in ("Expansion", "Expansion2")
    ]
    units = {}
    selected = ("WARRIOR", "ARCHER", "CHARIOT_ARCHER", "SPEARMAN", "HORSEMAN")
    for layer in layers:
        root = ET.fromstring(read(layer / "GlobalDefines.xml"))
        for row in root.findall("./Defines/Row"):
            defines[row.get("Name")] = row.findtext("Value")
        for path in sorted((layer / "Units").glob("*.xml")):
            if "Promotions" in path.name:
                read(path)
                continue
            if "Inherited" in path.name:
                # None of the five selected types has a BNW inherited override.
                root = ET.fromstring(read(path))
                for row in root.findall("./Units/Row"):
                    if row.findtext("Type", "").removeprefix("UNIT_") in selected:
                        raise ValueError(
                            f"Unexpected inherited unit definition: {path}"
                        )
                continue
            if path.name.lower() != "civ5units.xml":
                continue
            root = ET.fromstring(read(path))
            for row in root.findall("./Units/Row"):
                name = row.findtext("Type", "").removeprefix("UNIT_")
                if name in selected:
                    units[name] = {c.tag: c.text for c in row}
        for name in ("CIV5Terrains.xml", "CIV5Features.xml"):
            path = layer / "Terrain" / name
            if path.exists():
                read(path)
    changes = read(mod / "Database Changes/Defines/CoreDefineChanges.sql")
    defines.update(dict(re.findall(r"\('([A-Z0-9_]+)',\s*(-?\d+)\)", changes)))
    for directory in ("Units", "UnitPromotions"):
        for path in sorted((mod / "Database Changes" / directory).glob("*.sql")):
            read(path)
    for directory in ("Terrains", "Features"):
        for path in sorted(
            (mod / "Database Changes/WorldMap" / directory).glob("*.sql")
        ):
            read(path)
    options_xml = read(mod / "Database Changes/NewCustomModOptions.xml")
    options = {
        row.get("Name"): int(row.get("Value"))
        for row in ET.fromstring(options_xml).findall("./CustomModOptions/Row")
    }
    if (
        options.get("BALANCE_VP") != 0
        or options.get("BALANCE_CORE_RANGED_ATTACK_PENALTY") != 0
        or options.get("CORE_DELAYED_VISIBILITY") != 1
    ):
        raise ValueError(
            "Expected the CP Only profile without the VP ranged-defense rule"
        )
    constants = {
        name: int(defines[name])
        for name in (
            "FORTIFY_MODIFIER_PER_TURN",
            "NEUTRAL_HEAL_RATE",
            "WOUNDED_DAMAGE_MULTIPLIER",
            "ATTACK_SAME_STRENGTH_MIN_DAMAGE",
            "ATTACK_SAME_STRENGTH_POSSIBLE_EXTRA_DAMAGE",
            "HILLS_EXTRA_DEFENSE",
            "RIVER_ATTACK_MODIFIER",
            "BONUS_PER_ADJACENT_FRIEND",
        )
    }
    stats = [
        [
            int(units[n].get(k, "0"))
            for k in ("Combat", "RangedCombat", "Moves", "Range")
        ]
        for n in selected
    ]
    if stats != [
        [8, 0, 2, 0],
        [5, 7, 2, 2],
        [6, 10, 4, 2],
        [11, 0, 2, 0],
        [12, 0, 4, 0],
    ]:
        raise ValueError(f"Unsupported unit data: {stats}")
    commit = subprocess.check_output(
        ["git", "-C", str(cp), "rev-parse", "HEAD"], text=True
    ).strip()
    manifest = {
        "profile": "CP Only 4.22 (v144)",
        "balance_vp": False,
        "commit": commit,
        "sources": sources,
        "constants": constants,
        "unit_stats": dict(zip(selected, stats)),
        "options": {
            k: options[k]
            for k in (
                "BALANCE_VP",
                "BALANCE_CORE_RANGED_ATTACK_PENALTY",
                "SANE_UNIT_MOVEMENT_COST",
                "CORE_DELAYED_VISIBILITY",
            )
        },
    }
    manifest["kernel_sources"] = {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((ROOT / "native").glob("*"))
        if p.is_file()
    }
    fingerprint = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()
    ).hexdigest()
    manifest["fingerprint"] = fingerprint
    (out / "combat_rules.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    header = "\n".join(f"constexpr int {k} = {v};" for k, v in constants.items())
    header += '\nconstexpr const char* RULES_FINGERPRINT = "' + fingerprint + '";\n'
    (out / "combat_rules.h").write_text(header, encoding="utf-8")
    # An isolated oracle compiles the actual CP damage function, independently
    # of the standalone implementation. This is a test export, never in hot paths.
    reference = extract_cpp_function(
        read(sdk / "CvUnitCombat.cpp"), "int CvUnitCombat::DoDamageMath("
    )
    reference = reference.replace("CvUnitCombat::DoDamageMath", "reference_damage")
    reference = reference.replace(
        "GC.getGame().randRangeExclusive(0, iMaxRandomDamage100, randomSeed)",
        "randomSeed",
    )
    reference = reference.replace("const CvSeeder& randomSeed", "int randomSeed")
    oracle = "#include <algorithm>\n#include <cmath>\nusing std::min; using std::max;\n#define MIN std::min\n"
    oracle += reference
    oracle += '\n#ifdef _WIN32\n#define API extern "C" __declspec(dllexport)\n#else\n#define API extern "C"\n#endif\n'
    oracle += "API int cp_reference_damage(int a,int d,int roll) { return reference_damage(a,d,2400,1200,true,roll,0)/100; }\n"
    (out / "combat_reference.cpp").write_text(oracle, encoding="utf-8")
    library = "combat.dll" if os.name == "nt" else "libcombat.so"
    compile_shared(
        [ROOT / "native/combat.cpp", ROOT / "native/combat_state.cpp",
         ROOT / "native/geometry.cpp", out / "combat_reference.cpp"],
        out / library, include_dirs=[out], script_name="compile_combat.cmd",
    )
    print(f"Built combat kernel; rules {fingerprint[:16]}; constants {constants}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cp", type=Path, default=Path("E:/civ5/Community-Patch-DLL"))
    parser.add_argument(
        "--game", type=Path, default=Path("E:/civ5/Sid Meier's Civilization V")
    )
    args = parser.parse_args()
    build(args.cp.resolve(), args.game.resolve())
