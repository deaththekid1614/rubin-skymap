"""
build_sky_assets.py
===================
Generate static JSON data files for the rubin-skymap frontend sky renderer.

Outputs (written to frontend/data/):
  stars.json          ~9,000 Hipparcos-catalog bright stars (via HYG database)
  constellations.json 88 IAU constellation line segments (via Stellarium)
  milkyway.json       Galactic-plane centre-line in equatorial coords

Run once, commit the output files.  Safe to re-run (idempotent).

Usage:
    python scripts/build_sky_assets.py

Requirements (all already in requirements.txt / stdlib):
    requests, astropy, json, gzip, io, math
"""

import gzip
import io
import json
import math
import os
import sys
from pathlib import Path

# ── Output directory ─────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parent.parent
OUT_DIR    = REPO_ROOT / "frontend" / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STARS_OUT    = OUT_DIR / "stars.json"
CONST_OUT    = OUT_DIR / "constellations.json"
MILKYWAY_OUT = OUT_DIR / "milkyway.json"

N_STARS     = 9000          # brightest N stars to keep
MAG_LIMIT   = 7.0           # absolute cut-off magnitude (skip fainter)

# ═════════════════════════════════════════════════════════════════════════════
#  FALLBACK EMBEDDED DATA (used when network is unavailable)
# ═════════════════════════════════════════════════════════════════════════════

FALLBACK_STARS = [
    # name,           ra_h,    dec_deg,  mag
    ("Sirius",        6.7525,  -16.7161, -1.46),
    ("Canopus",       6.3992,  -52.6957, -0.74),
    ("Rigil Kent",   14.6600,  -60.8333, -0.27),
    ("Arcturus",     14.2610,   19.1822, -0.05),
    ("Vega",         18.6157,   38.7836,  0.03),
    ("Capella",       5.2781,   45.9980,  0.08),
    ("Rigel",         5.2423,   -8.2017,  0.12),
    ("Procyon",       7.6553,    5.2250,  0.38),
    ("Achernar",      1.6286,  -57.2367,  0.46),
    ("Betelgeuse",    5.9194,    7.4069,  0.50),
    ("Hadar",        14.0639,  -60.3731,  0.61),
    ("Altair",       19.8463,    8.8683,  0.77),
    ("Aldebaran",     4.5986,   16.5093,  0.85),
    ("Acrux",        12.4433,  -63.0991,  0.87),
    ("Antares",      16.4901,  -26.4319,  1.09),
    ("Spica",        13.4199,  -11.1613,  1.04),
    ("Pollux",        7.7553,   28.0262,  1.14),
    ("Fomalhaut",    22.9608,  -29.6223,  1.16),
    ("Deneb",        20.6905,   45.2803,  1.25),
    ("Mimosa",       12.7954,  -59.6888,  1.25),
    ("Regulus",      10.1395,   11.9672,  1.36),
    ("Adhara",        6.9771,  -28.9722,  1.50),
    ("Castor",        7.5766,   31.8883,  1.58),
    ("Shaula",       17.5604,  -37.1036,  1.63),
    ("Gacrux",       12.5194,  -57.1133,  1.64),
    ("Bellatrix",     5.4189,    6.3497,  1.64),
    ("Elnath",        5.4382,   28.6075,  1.65),
    ("Miaplacidus",   9.2200,  -69.7172,  1.68),
    ("Alnilam",       5.6036,   -1.2019,  1.70),
    ("Alnitak",       5.6796,   -1.9425,  1.74),
    ("Gamma Vel",     8.1586,  -47.3366,  1.78),
    ("Alioth",       12.9004,   55.9598,  1.77),
    ("Mirfak",        3.4054,   49.8612,  1.79),
    ("Dubhe",        11.0621,   61.7511,  1.79),
    ("Wezen",         7.1397,  -26.3933,  1.84),
    ("Kaus Australis",18.4028, -34.3846,  1.85),
    ("Avior",         8.3750,  -59.5092,  1.86),
    ("Alkaid",       13.7924,   49.3133,  1.86),
    ("Sargas",       17.6219,  -42.9979,  1.87),
    ("Menkent",      14.1114,  -36.3700,  2.06),
    ("Atria",        16.8111,  -69.0278,  1.91),
    ("Alhena",        6.6285,   16.3993,  1.93),
    ("Peacock",      20.4275,  -56.7350,  1.94),
    ("Mirzam",        6.3783,  -17.9556,  1.98),
    ("Alphard",       9.4598,   -8.6586,  1.98),
    ("Polaris",       2.5300,   89.2641,  1.97),
    ("Hamal",         2.1199,   23.4624,  2.00),
    ("Algieba",      10.3320,   19.8414,  2.01),
    ("Nunki",        18.9211,  -26.2967,  2.02),
    ("Menkib",        3.9080,   31.8836,  2.03),
    ("Saiph",         5.7955,   -9.6697,  2.06),
]

FALLBACK_CONSTELLATIONS = [
    {
        "name": "Orion",
        "segments": [
            [5.5203, 9.9342, 5.2423, -8.2017],   # Betelgeuse-Rigel
            [5.5203, 9.9342, 5.9194, 7.4069],    # Betelgeuse-Bellatrix approx
            [5.6036, -1.2019, 5.6796, -1.9425],  # Alnilam-Alnitak
            [5.5332, -0.2991, 5.6036, -1.2019],  # Mintaka-Alnilam
        ],
    },
    {
        "name": "Ursa Major",
        "segments": [
            [11.0621, 61.7511, 11.8971, 53.6947], # Dubhe-Merak
            [11.8971, 53.6947, 12.2572, 57.0326], # Merak-Phecda
            [12.2572, 57.0326, 12.9004, 55.9598], # Phecda-Alioth
            [12.9004, 55.9598, 13.3988, 54.9253], # Alioth-Mizar
            [13.3988, 54.9253, 13.7924, 49.3133], # Mizar-Alkaid
        ],
    },
    {
        "name": "Scorpius",
        "segments": [
            [16.4901, -26.4319, 16.0053, -22.6217],
            [16.4901, -26.4319, 16.8356, -34.2928],
            [17.5604, -37.1036, 17.6219, -42.9979],
        ],
    },
    {
        "name": "Leo",
        "segments": [
            [10.1395, 11.9672, 10.3320, 19.8414],
            [10.3320, 19.8414, 11.2352, 20.5236],
            [10.1395, 11.9672, 9.7642, 23.7742],
        ],
    },
    {
        "name": "Crux",
        "segments": [
            [12.4433, -63.0991, 12.7954, -59.6888],
            [12.5194, -57.1133, 12.3564, -60.4017],
        ],
    },
]

# ═════════════════════════════════════════════════════════════════════════════
#  HIPPARCOS / HYG STAR CATALOG
# ═════════════════════════════════════════════════════════════════════════════

def _bright_star_names():
    """Return a dict of {hip_id: common_name} for the very brightest stars."""
    return {
        32349: "Sirius",       30438: "Canopus",      71683: "Rigil Kent",
        69673: "Arcturus",     91262: "Vega",         24608: "Capella",
        24436: "Rigel",        37279: "Procyon",       7588: "Achernar",
        27989: "Betelgeuse",   68702: "Hadar",        97649: "Altair",
        21421: "Aldebaran",    60718: "Acrux",        80763: "Antares",
        65474: "Spica",        37826: "Pollux",       113368: "Fomalhaut",
       102098: "Deneb",        62434: "Mimosa",       49669: "Regulus",
        33579: "Adhara",       36850: "Castor",       85927: "Shaula",
        61084: "Gacrux",       25336: "Bellatrix",    25428: "Elnath",
        45238: "Miaplacidus",  26311: "Alnilam",      26727: "Alnitak",
        39953: "Gamma Vel",    62956: "Alioth",       15863: "Mirfak",
        54061: "Dubhe",        35904: "Wezen",        90185: "Kaus Australis",
        41037: "Avior",        67301: "Alkaid",       85696: "Sargas",
        68933: "Menkent",      82273: "Atria",        31681: "Alhena",
       100751: "Peacock",      31685: "Mirzam",       46390: "Alphard",
        11767: "Polaris",       9884: "Hamal",        50583: "Algieba",
        92855: "Nunki",        17851: "Menkib",       27366: "Saiph",
    }


def fetch_hipparcos():
    """Download the HYG database (includes Hipparcos IDs and magnitudes).

    Source: astronexus/HYG-Database hygdata_v41.csv on GitHub.
    HYG stores RA already in decimal hours [0, 24) and Dec in degrees.
    Returns list of dicts: {ra_h, dec_deg, mag, name}.
    Falls back to FALLBACK_STARS on any network or parse error.
    """
    try:
        import requests
    except ImportError:
        print("[warn] requests not available — using fallback star data")
        return _fallback_stars()

    # HYG v41: plain CSV, ~120 KB, includes hip, ra (decimal hours), dec, mag, proper
    URL = (
        "https://raw.githubusercontent.com/astronexus/"
        "HYG-Database/main/hyg/CURRENT/hygdata_v41.csv"
    )
    print(f"[stars] Fetching HYG database …")
    try:
        resp = requests.get(URL, timeout=120)
        resp.raise_for_status()
    except Exception as exc:
        print(f"[warn] HYG fetch failed ({exc}) — using fallback star data")
        return _fallback_stars()

    print(f"[stars]   downloaded {len(resp.content) / 1024:.0f} KB")

    names = _bright_star_names()
    stars = []
    i_hip = i_ra = i_dec = i_mag = i_prop = -1

    for i, line in enumerate(resp.text.splitlines()):
        if i == 0:
            # Strip surrounding quotes (HYG v41 uses quoted headers)
            headers = [h.strip().strip('"').lower() for h in line.split(",")]
            try:
                i_hip  = headers.index("hip")
                i_ra   = headers.index("ra")
                i_dec  = headers.index("dec")
                i_mag  = headers.index("mag")
            except ValueError as e:
                print(f"[warn] HYG CSV missing column ({e}) — using fallback")
                return _fallback_stars()
            i_prop = headers.index("proper") if "proper" in headers else -1
            continue

        parts = line.split(",")
        if len(parts) <= max(i_hip, i_ra, i_dec, i_mag):
            continue
        try:
            vmag    = float(parts[i_mag].strip().strip('"'))
            ra_h    = float(parts[i_ra].strip().strip('"'))   # decimal hours
            dec_deg = float(parts[i_dec].strip().strip('"'))
            hip_raw = parts[i_hip].strip().strip('"')
            hip_id  = int(hip_raw) if hip_raw else 0
        except (ValueError, IndexError):
            continue

        if vmag > MAG_LIMIT:
            continue

        proper = names.get(hip_id)
        if proper is None and i_prop >= 0:
            raw_name = parts[i_prop].strip()
            proper = raw_name if raw_name else None

        stars.append({
            "ra_h":    round(ra_h, 5),
            "dec_deg": round(dec_deg, 5),
            "mag":     round(vmag, 2),
            "name":    proper,
        })

    if len(stars) < 100:
        print("[warn] Too few stars parsed — using fallback star data")
        return _fallback_stars()

    # Sort by magnitude (brightest first), keep top N
    stars.sort(key=lambda s: s["mag"])
    stars = stars[:N_STARS]
    print(f"[stars] Parsed {len(stars)} stars (mag ≤ {MAG_LIMIT})")
    return stars


def _fallback_stars():
    stars = []
    for row in FALLBACK_STARS:
        name, ra_h, dec_deg, mag = row
        stars.append({"ra_h": ra_h, "dec_deg": dec_deg, "mag": mag, "name": name})
    print(f"[stars] Using {len(stars)} fallback stars")
    return stars


# ═════════════════════════════════════════════════════════════════════════════
#  CONSTELLATION LINES
# ═════════════════════════════════════════════════════════════════════════════

def fetch_constellations():
    """Fetch constellation lines from Stellarium + HYG star positions.

    Uses:
      - constellationship.fab from Stellarium tag v23.2 (88 IAU constellations,
        each line lists HIP pairs)
      - HYG v41 CSV for HIP-id → (ra_h, dec_deg) lookup

    Returns list of {name, segments: [[ra1,dec1,ra2,dec2], ...]}.
    Falls back to FALLBACK_CONSTELLATIONS on any error.
    """
    try:
        import requests
    except ImportError:
        print("[warn] requests not available — using fallback constellation data")
        return FALLBACK_CONSTELLATIONS

    URL_LINES = (
        "https://raw.githubusercontent.com/Stellarium/"
        "stellarium/v23.2/skycultures/modern/constellationship.fab"
    )
    URL_HYG = (
        "https://raw.githubusercontent.com/astronexus/"
        "HYG-Database/main/hyg/CURRENT/hygdata_v41.csv"
    )

    print("[consts] Fetching constellation lines from Stellarium …")
    try:
        r_lines = requests.get(URL_LINES, timeout=60)
        r_lines.raise_for_status()
    except Exception as exc:
        print(f"[warn] Constellation lines fetch failed ({exc}) — using fallback")
        return FALLBACK_CONSTELLATIONS

    print("[consts] Fetching HYG star positions …")
    try:
        r_hyg = requests.get(URL_HYG, timeout=120)
        r_hyg.raise_for_status()
    except Exception as exc:
        print(f"[warn] HYG catalog fetch failed ({exc}) — using fallback")
        return FALLBACK_CONSTELLATIONS

    # Build HIP → (ra_h, dec_deg) lookup from HYG CSV
    hip_pos = {}
    i_hip = i_ra = i_dec = -1
    for idx, line in enumerate(r_hyg.text.splitlines()):
        if idx == 0:
            # Strip surrounding quotes (HYG v41 uses quoted headers)
            headers = [h.strip().strip('"').lower() for h in line.split(",")]
            try:
                i_hip = headers.index("hip")
                i_ra  = headers.index("ra")
                i_dec = headers.index("dec")
            except ValueError:
                print("[warn] HYG CSV headers unexpected — using fallback")
                return FALLBACK_CONSTELLATIONS
            continue
        parts = line.split(",")
        if len(parts) <= max(i_hip, i_ra, i_dec):
            continue
        try:
            hip_raw = parts[i_hip].strip().strip('"')
            if not hip_raw:
                continue
            hip = int(hip_raw)
            ra  = float(parts[i_ra].strip().strip('"'))
            dec = float(parts[i_dec].strip().strip('"'))
            hip_pos[hip] = (round(ra, 5), round(dec, 5))
        except (ValueError, IndexError):
            continue

    # IAU abbreviation → full name mapping
    IAU_NAMES = {
        "And":"Andromeda","Ant":"Antlia","Aps":"Apus","Aqr":"Aquarius",
        "Aql":"Aquila","Ara":"Ara","Ari":"Aries","Aur":"Auriga",
        "Boo":"Bootes","Cae":"Caelum","Cam":"Camelopardalis","Cnc":"Cancer",
        "CVn":"Canes Venatici","CMa":"Canis Major","CMi":"Canis Minor",
        "Cap":"Capricornus","Car":"Carina","Cas":"Cassiopeia","Cen":"Centaurus",
        "Cep":"Cepheus","Cet":"Cetus","Cha":"Chamaeleon","Cir":"Circinus",
        "Col":"Columba","Com":"Coma Berenices","CrA":"Corona Australis",
        "CrB":"Corona Borealis","Crv":"Corvus","Crt":"Crater","Cru":"Crux",
        "Cyg":"Cygnus","Del":"Delphinus","Dor":"Dorado","Dra":"Draco",
        "Equ":"Equuleus","Eri":"Eridanus","For":"Fornax","Gem":"Gemini",
        "Gru":"Grus","Her":"Hercules","Hor":"Horologium","Hya":"Hydra",
        "Hyi":"Hydrus","Ind":"Indus","Lac":"Lacerta","Leo":"Leo",
        "LMi":"Leo Minor","Lep":"Lepus","Lib":"Libra","Lup":"Lupus",
        "Lyn":"Lynx","Lyr":"Lyra","Men":"Mensa","Mic":"Microscopium",
        "Mon":"Monoceros","Mus":"Musca","Nor":"Norma","Oct":"Octans",
        "Oph":"Ophiuchus","Ori":"Orion","Pav":"Pavo","Peg":"Pegasus",
        "Per":"Perseus","Phe":"Phoenix","Pic":"Pictor","Psc":"Pisces",
        "PsA":"Piscis Austrinus","Pup":"Puppis","Pyx":"Pyxis","Ret":"Reticulum",
        "Sge":"Sagitta","Sgr":"Sagittarius","Sco":"Scorpius","Scl":"Sculptor",
        "Sct":"Scutum","Ser":"Serpens","Sex":"Sextans","Tau":"Taurus",
        "Tel":"Telescopium","Tri":"Triangulum","TrA":"Triangulum Australe",
        "Tuc":"Tucana","UMa":"Ursa Major","UMi":"Ursa Minor","Vel":"Vela",
        "Vir":"Virgo","Vol":"Volans","Vul":"Vulpecula",
    }

    constellations = []
    for line in r_lines.text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        abbr  = parts[0]
        try:
            n_seg = int(parts[1])
        except ValueError:
            continue
        hips  = parts[2:]
        if len(hips) < n_seg * 2:
            continue

        segments = []
        for i in range(n_seg):
            h1, h2 = int(hips[i * 2]), int(hips[i * 2 + 1])
            p1 = hip_pos.get(h1)
            p2 = hip_pos.get(h2)
            if p1 and p2:
                segments.append([p1[0], p1[1], p2[0], p2[1]])

        if segments:
            full_name = IAU_NAMES.get(abbr, abbr)
            constellations.append({"name": full_name, "segments": segments})

    if len(constellations) < 10:
        print("[warn] Too few constellations parsed — using fallback")
        return FALLBACK_CONSTELLATIONS

    print(f"[consts] Parsed {len(constellations)} constellations")
    return constellations


# ═════════════════════════════════════════════════════════════════════════════
#  MILKY WAY PLANE
# ═════════════════════════════════════════════════════════════════════════════

def build_milkyway():
    """Compute the galactic plane centre-line in equatorial coords.

    Uses astropy to transform galactic (l, b=0) → ICRS (RA, Dec).
    Returns {"points": [[ra_h, dec_deg], ...], "width_deg": 20}.
    Falls back to a pure-math great-circle approximation on failure.
    """
    try:
        from astropy.coordinates import Galactic, ICRS
        import astropy.units as u

        print("[milkyway] Computing galactic plane via astropy …")
        N = 360
        points = []
        for i in range(N + 1):
            l = i * (360.0 / N)
            gc = Galactic(l=l * u.deg, b=0 * u.deg)
            eq = gc.transform_to(ICRS())
            ra_h  = round(float(eq.ra.deg) / 15.0, 4)
            dec   = round(float(eq.dec.deg), 4)
            points.append([ra_h, dec])
        print(f"[milkyway] Generated {len(points)} galactic plane points")
        return {"points": points, "width_deg": 20}

    except Exception as exc:
        print(f"[warn] astropy galactic transform failed ({exc}) — using analytic approx")
        return _fallback_milkyway()


def _fallback_milkyway():
    """Analytic great-circle approximation of the galactic plane.

    The galactic North Pole is at RA=192.859°, Dec=+27.128° (J2000).
    The Galactic Centre is at RA=266.405°, Dec=-28.936° (J2000).
    We parameterise the great circle through the GC perpendicular to GNP.
    """
    gnp_ra  = math.radians(192.859)
    gnp_dec = math.radians(27.128)
    nx = math.cos(gnp_dec) * math.cos(gnp_ra)
    ny = math.cos(gnp_dec) * math.sin(gnp_ra)
    nz = math.sin(gnp_dec)

    # Galactic Centre as start vector
    gc_ra  = math.radians(266.405)
    gc_dec = math.radians(-28.936)
    vx = math.cos(gc_dec) * math.cos(gc_ra)
    vy = math.cos(gc_dec) * math.sin(gc_ra)
    vz = math.sin(gc_dec)

    # Perpendicular vector in the plane via n × v
    wx = ny * vz - nz * vy
    wy = nz * vx - nx * vz
    wz = nx * vy - ny * vx
    wlen = math.sqrt(wx**2 + wy**2 + wz**2)
    wx /= wlen; wy /= wlen; wz /= wlen

    N = 360
    points = []
    for i in range(N + 1):
        t = 2 * math.pi * i / N
        px = math.cos(t) * vx + math.sin(t) * wx
        py = math.cos(t) * vy + math.sin(t) * wy
        pz = math.cos(t) * vz + math.sin(t) * wz
        dec_r = math.asin(max(-1.0, min(1.0, pz)))
        ra_r  = math.atan2(py, px)
        ra_h  = round((math.degrees(ra_r) % 360) / 15.0, 4)
        dec   = round(math.degrees(dec_r), 4)
        points.append([ra_h, dec])

    print(f"[milkyway] Generated {len(points)} galactic plane points (analytic fallback)")
    return {"points": points, "width_deg": 20}


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  build_sky_assets.py — rubin-skymap frontend data builder")
    print("=" * 60)

    # ── Stars ────────────────────────────────────────────────────────────────
    stars = fetch_hipparcos()
    with open(STARS_OUT, "w") as f:
        json.dump(stars, f, separators=(",", ":"))
    size_kb = STARS_OUT.stat().st_size / 1024
    print(f"[stars] Wrote {STARS_OUT}  ({len(stars)} stars, {size_kb:.1f} KB)")
    if size_kb > 1024:
        print("[warn]  stars.json > 1 MB — consider reducing N_STARS")

    # ── Constellations ───────────────────────────────────────────────────────
    constellations = fetch_constellations()
    with open(CONST_OUT, "w") as f:
        json.dump(constellations, f, separators=(",", ":"))
    size_kb = CONST_OUT.stat().st_size / 1024
    print(f"[consts] Wrote {CONST_OUT}  ({len(constellations)} constellations, {size_kb:.1f} KB)")
    if size_kb > 512:
        print("[warn]   constellations.json > 512 KB")

    # ── Milky Way ────────────────────────────────────────────────────────────
    milkyway = build_milkyway()
    with open(MILKYWAY_OUT, "w") as f:
        json.dump(milkyway, f, separators=(",", ":"))
    size_kb = MILKYWAY_OUT.stat().st_size / 1024
    n_pts = len(milkyway["points"])
    print(f"[milkyway] Wrote {MILKYWAY_OUT}  ({n_pts} points, {size_kb:.1f} KB)")

    total_kb = sum(p.stat().st_size for p in [STARS_OUT, CONST_OUT, MILKYWAY_OUT]) / 1024
    print()
    print(f"[done] Total asset size: {total_kb:.1f} KB")
    print(f"[done] Output directory: {OUT_DIR}")
    print()
    print("Next step: open the dashboard and the sky should show stars + constellations.")


if __name__ == "__main__":
    main()
