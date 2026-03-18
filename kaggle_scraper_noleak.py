"""
Build pre-tournament KenPom ratings from two sources:

1. Local CSVs (data/kenpom_pretourney/*.csv) for 2002-2018 — high-precision
   pre-tournament snapshots.
2. Wayback Machine archives for 2019-2025 — scraped before Selection Sunday.
3. Live kenpom.com for the current season (tournament hasn't happened yet).

Output: data/kenpom_pretourney.csv with the same schema consumed by
tune_xgb_features.py and 2026_notebook.ipynb.
"""

import re
from pathlib import Path
from time import sleep

import cloudscraper
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

# ── Local CSV directory (2002-2018) ─────────────────────────────────────
CSV_DIR = Path("data/kenpom_pretourney")
CSV_YEARS = range(2002, 2019)  # 2002 through 2018 inclusive

# ── Archive URLs ────────────────────────────────────────────────────────
# 2004-2010: legacy <pre> format (fallback if CSVs missing)
# 2011-2018: supplementary columns (Luck, SOS, NCSOS) merged into CSVs
# 2019-2025: sole source (full data)
ARCHIVE_URLS: dict[int, str] = {
    2004: "https://web.archive.org/web/20040202143610/http://kenpom.com/rate.php",
    2005: "https://web.archive.org/web/20050302021451/http://kenpom.com/rate.php",
    2006: "https://web.archive.org/web/20060318072421/http://kenpom.com/rate.php",
    2007: "https://web.archive.org/web/20070313232824/http://kenpom.com/rate.php",
    2009: "https://web.archive.org/web/20090315085050/http://kenpom.com/rate.php",
    2010: "https://web.archive.org/web/20100304023540/http://kenpom.com/rate.php",
    2011: "https://web.archive.org/web/20110311233233/http://www.kenpom.com/",
    2012: "https://web.archive.org/web/20120311165019/http://kenpom.com/",
    2013: "https://web.archive.org/web/20130318221134/http://kenpom.com/",
    2014: "https://web.archive.org/web/20140318100454/http://kenpom.com/",
    2015: "https://web.archive.org/web/20150316212936/http://kenpom.com/",
    2016: "https://web.archive.org/web/20160314134726/http://kenpom.com/",
    2017: "https://web.archive.org/web/20170312131016/http://kenpom.com/",
    2018: "https://web.archive.org/web/20180311122559/https://kenpom.com/",
    2019: "https://web.archive.org/web/20190317211809/https://kenpom.com/",
    # 2020: COVID — no tournament
    2021: "https://web.archive.org/web/20210314233855/https://kenpom.com/",
    2022: "https://web.archive.org/web/20220313232423/https://kenpom.com/",
    2023: "https://web.archive.org/web/20230312230828/https://kenpom.com/",
    2024: "https://web.archive.org/web/20240317150142/https://kenpom.com/",
    2025: "https://web.archive.org/web/20250316235356/https://kenpom.com/",
}

# Columns that archives can supplement into CSV data.
SUPPLEMENT_COLS = [
    "Conf", "W", "L", "Luck",
    "SOS_NetRtg", "SOS_ORtg", "SOS_DRtg", "NCSOS_NetRtg",
]

# Current season — live kenpom.com (tournament hasn't happened yet).
LIVE_URL = "https://kenpom.com/index.php"
CURRENT_YEAR = 2026

# ── Team-name normalisation ────────────────────────────────────────────
TEAM_NAME_MAP: dict[str, str] = {
    "arkansas little rock": "arkansas-little-rock",
    "louisiana lafayette": "louisiana-lafayette",
    "southwest missouri st.": "southwest missouri state",
    "illinois chicago": "illinois-chicago",
    "texas pan american": "texas rio grande valley",
    "louisiana monroe": "louisiana-monroe",
    "southwest texas st.": "texas-st",
    "tennessee martin": "tennessee-martin",
    "texas a&m corpus chris": "texas a&m-corpus christi",
    "st. francis ny": "st. francis-ny",
    "southeast missouri st.": "southeast-missouri-state",
    "mississippi valley st.": "mississippi-valley-state",
    "bethune cookman": "bethune-cookman",
    "st. francis pa": "saint-francis-pa",
    "arkansas pine bluff": "arkansas-pine bluff",
    "winston salem st.": "winston-salem-state",
    "cal st. bakersfield": "cal state bakersfield",
    "ut rio grande valley": "texas rio grande valley",
    "tarleton st.": "tarleton st",
    "dixie st.": "dixie st",
    "queens": "queens nc",
    "texas a&m commerce": "texas a&m-commerce",
    "saint francis": "saint-francis-pa",
    "southeast missouri": "southeast-missouri-state",
    "liu": "liu brooklyn",
    "virginia military inst": "vmi",
    "nj inst of technology": "njit",
    "md baltimore county": "umbc",
    "alabama a&m;": "alabama a&m",
    "east texas a&m;": "east texas a&m",
    "florida a&m;": "florida a&m",
    "north carolina a&t;": "north carolina a&t",
    "prairie view a&m;": "prairie view a&m",
    "texas a&m;": "texas a&m",
    "texas a&m; commerce": "texas a&m-commerce",
    "texas a&m; corpus chris": "texas a&m-corpus christi",
    "cal davis": "uc davis",
    "cal irvine": "uc irvine",
    "cal santa barbara": "uc santa barbara",
    "iupu fort wayne": "ipfw",
    "louisiana st.": "lsu",
    "maryland baltimore county": "umbc",
    "missouri kansas city": "umkc",
    "nevada las vegas": "unlv",
    "st. louis": "saint louis",
    "texas el paso": "utep",
    "wisconsin green bay": "green bay",
    "wisconsin milwaukee": "milwaukee",
    # Legacy archive full-name → abbreviated canonical forms
    "brigham young": "byu",
    "central florida": "ucf",
    "florida international": "fiu",
    "grambling": "grambling st.",
    "long island": "liu brooklyn",
    "loyola chicago": "loyola-chicago",
    "mount st. mary's": "mt st mary's",
    "nc asheville": "unc asheville",
    "nc greensboro": "unc greensboro",
    "nc wilmington": "unc wilmington",
    "north carolina central": "nc central",
    "pennsylvania": "penn",
    "south carolina upstate": "usc upstate",
    "southern california": "usc",
    "southern methodist": "smu",
    "southern mississippi": "southern miss",
    "se louisiana": "southeastern louisiana",
    "st. bonaventure": "st bonaventure",
    "st. john's": "st john's",
    "st. joseph's": "saint joseph's",
    "st. mary's": "saint mary's",
    "st. peter's": "saint peter's",
    "texas arlington": "ut arlington",
    "texas christian": "tcu",
    "texas san antonio": "utsa",
    "virginia commonwealth": "vcu",
    "virginia military": "vmi",
    "alabama birmingham": "uab",
    "cal riverside": "uc riverside",
    "fort wayne": "ipfw",
}

# ── CSV name normalisation ─────────────────────────────────────────────
# The CSVs use different team name conventions than MTeamSpellings.
CSV_TEAM_NAME_MAP: dict[str, str] = {
    "Alabama A&M": "alabama a&m",
    "Abilene Christian": "abilene christian",
    "Arkansas Pine Bluff": "arkansas-pine bluff",
    "Bethune Cookman": "bethune-cookman",
    "Cal St. Bakersfield": "cal state bakersfield",
    "Cal St. Fullerton": "cal st fullerton",
    "Cal St. Northridge": "cal st northridge",
    "East Tennessee St.": "east tennessee st",
    "Florida A&M": "florida a&m",
    "Florida Atlantic": "fau",
    "Florida Gulf Coast": "florida gulf coast",
    "Fort Wayne": "ipfw",
    "Houston Baptist": "houston baptist",
    "Illinois Chicago": "illinois-chicago",
    "LIU Brooklyn": "liu brooklyn",
    "Louisiana Lafayette": "louisiana-lafayette",
    "Louisiana Monroe": "louisiana-monroe",
    "Loyola Chicago": "loyola-chicago",
    "Loyola MD": "loyola md",
    "Mississippi Valley St.": "mississippi-valley-state",
    "Mount St. Mary's": "mt st mary's",
    "NC State": "north carolina st",
    "North Carolina A&T": "north carolina a&t",
    "North Carolina Central": "nc central",
    "Prairie View A&M": "prairie view a&m",
    "Southeast Missouri St.": "southeast-missouri-state",
    "Southern Miss": "southern miss",
    "St. Bonaventure": "st bonaventure",
    "St. Francis NY": "st. francis-ny",
    "St. Francis PA": "saint-francis-pa",
    "St. John's": "st john's",
    "Tennessee Martin": "tennessee-martin",
    "Texas A&M": "texas a&m",
    "Texas A&M Corpus Chris": "texas a&m-corpus christi",
    "Texas Southern": "texas southern",
    "UC Davis": "uc davis",
    "UC Irvine": "uc irvine",
    "UC Riverside": "uc riverside",
    "UC Santa Barbara": "uc santa barbara",
    "UMass Lowell": "umass lowell",
    "UMBC": "umbc",
    "UMKC": "umkc",
    "UNC Asheville": "unc asheville",
    "UNC Greensboro": "unc greensboro",
    "UNC Wilmington": "unc wilmington",
    "UNLV": "unlv",
    "USC Upstate": "usc upstate",
    "UT Arlington": "ut arlington",
    "UT Rio Grande Valley": "texas rio grande valley",
    "UTEP": "utep",
    "UTSA": "utsa",
    "VMI": "vmi",
    "Winston Salem St.": "winston-salem-state",
    "Wisconsin Green Bay": "green bay",
    "Wisconsin Milwaukee": "milwaukee",
    "Cal St. Fullerton": "cal st. fullerton",
    "Cal St. Northridge": "cal st. northridge",
    "East Tennessee St.": "east tennessee st.",
    "Florida Atlantic": "florida atlantic",
}


# ── CSV loader ──────────────────────────────────────────────────────────

def load_csv_year(year: int) -> pd.DataFrame:
    """Load a local pre-tournament KenPom CSV and normalise to our schema."""
    path = CSV_DIR / f"kenpom_pretourney_{year}.csv"
    if not path.exists():
        raise FileNotFoundError(f"No CSV for {year}: {path}")

    df = pd.read_csv(path)

    # The CSVs have two schemas:
    #   2002-2016: AdjOE, AdjDE, Pythag/RankPythag, Tempo/AdjTempo
    #   2017-2018: AdjOE, AdjDE, AdjEM/RankAdjEM (no Tempo)
    out = pd.DataFrame()
    out["Team"] = df["TeamName"]

    # Map to our standard column names.
    out["ORtg"] = pd.to_numeric(df["AdjOE"], errors="coerce")
    out["DRtg"] = pd.to_numeric(df["AdjDE"], errors="coerce")

    if "AdjEM" in df.columns:
        out["NetRtg"] = pd.to_numeric(df["AdjEM"], errors="coerce")
        out["Rk"] = pd.to_numeric(df["RankAdjEM"], errors="coerce")
    elif "Pythag" in df.columns:
        out["NetRtg"] = out["ORtg"] - out["DRtg"]  # AdjEM = AdjOE - AdjDE
        out["Rk"] = pd.to_numeric(df["RankPythag"], errors="coerce")

    if "AdjTempo" in df.columns:
        out["AdjT"] = pd.to_numeric(df["AdjTempo"], errors="coerce")

    out["Season"] = year
    out = out.dropna(subset=["Team", "ORtg", "DRtg"]).copy()

    return out


# ── Web scraping helpers ────────────────────────────────────────────────

def _parse_modern_table(html: str) -> pd.DataFrame:
    """Parse the modern kenpom.com ratings-table (2011+)."""
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", {"id": "ratings-table"})
    if table is None:
        raise ValueError("No ratings-table found in HTML")

    rows = []
    for tr in table.find("tbody").find_all("tr"):
        if tr.find("th"):
            continue
        tds = tr.find_all("td")
        if len(tds) < 5:
            continue

        row: dict = {}
        row["Rk"] = int(tds[0].get_text(strip=True))

        team_td = tds[1]
        seed_span = team_td.find("span", class_="seed")
        if seed_span:
            seed_span.decompose()
        row["Team"] = team_td.get_text(strip=True)

        row["Conf"] = tds[2].get_text(strip=True)

        wl = tds[3].get_text(strip=True)
        parts = wl.split("-")
        row["W"] = int(parts[0])
        row["L"] = int(parts[1])

        row["NetRtg"] = float(tds[4].get_text(strip=True))

        paired = [
            ("ORtg", 5), ("DRtg", 7), ("AdjT", 9), ("Luck", 11),
            ("SOS_NetRtg", 13), ("SOS_ORtg", 15), ("SOS_DRtg", 17),
            ("NCSOS_NetRtg", 19),
        ]
        for col_name, td_idx in paired:
            if td_idx >= len(tds):
                break
            row[col_name] = float(tds[td_idx].get_text(strip=True))

        rows.append(row)

    return pd.DataFrame(rows)


def _parse_legacy_pre(html: str) -> pd.DataFrame:
    """Parse legacy kenpom.com <pre>-formatted ratings (2004-2010).

    Two sub-formats:
      2004-2006 (simple): Rnk Team Conf W-L Rating SOS Rnk SOS_NC Rnk ...
      2007-2010 (rich):   Rnk Team Conf W-L Pyth AdjO/Rk AdjD/Rk Cons/Rk
                          Luck/Rk SOS_Pyth/Rk SOS_OppO/Rk SOS_OppD/Rk NCSOS/Rk
    """
    soup = BeautifulSoup(html, "lxml")
    pre = soup.find("pre")
    if pre is None:
        raise ValueError("No <pre> block found in legacy HTML")

    lines = pre.get_text().splitlines()
    # Detect format by checking for "/" in header or first data line
    is_rich = any("AdjO/Rnk" in l or "AdjO/" in l for l in lines[:5])
    if not is_rich:
        # Check first data lines too
        for l in lines:
            if l.strip() and l.strip()[0].isdigit() and "/" in l:
                is_rich = True
                break

    rows = []
    for line in lines:
        line = line.strip()
        if not line or not line[0].isdigit():
            continue

        # Extract rank
        m = re.match(r"(\d+)\s+(.+)", line)
        if not m:
            continue
        rk = int(m.group(1))
        rest = m.group(2)

        # Find W-L pattern: either "W-L" or "W  L" (2004 uses spaces)
        wl_match = re.search(r"(\d+)\s*-\s*(\d+)\s+", rest)
        if not wl_match:
            # Try "W  L" format (2004)
            wl_match = re.search(r"(\d+)\s+(\d+)\s+", rest)
            if not wl_match:
                continue

        w, l = int(wl_match.group(1)), int(wl_match.group(2))
        team_conf = rest[: wl_match.start()].strip()
        after_wl = rest[wl_match.end() :].strip()

        # Team is everything except the last token (conference)
        parts = team_conf.rsplit(None, 1)
        if len(parts) == 2:
            team, conf = parts
        else:
            team = team_conf
            conf = ""

        row: dict = {"Rk": rk, "Team": team.strip(), "Conf": conf, "W": w, "L": l}

        if is_rich:
            # 2007-2010: Pyth  AdjO/Rk  AdjD/Rk  Cons/Rk  Luck/Rk
            #            SOS_Pyth/Rk  SOS_OppO/Rk  SOS_OppD/Rk  NCSOS_Pyth/Rk
            vals = re.findall(r"([+\-]?[\d.]+)\s*/\s*\d+", after_wl)
            if len(vals) >= 2:
                row["ORtg"] = float(vals[0])
                row["DRtg"] = float(vals[1])
                row["NetRtg"] = row["ORtg"] - row["DRtg"]
            if len(vals) >= 4:
                # vals[2] = Consistency, vals[3] = Luck
                row["Luck"] = float(vals[3])
            if len(vals) >= 5:
                row["SOS_NetRtg"] = float(vals[4])
            if len(vals) >= 6:
                row["SOS_ORtg"] = float(vals[5])
            if len(vals) >= 7:
                row["SOS_DRtg"] = float(vals[6])
            if len(vals) >= 8:
                row["NCSOS_NetRtg"] = float(vals[7])
        else:
            # 2004-2006: Rating  SOS_Overall  Rnk  SOS_NC  Rnk  ...
            # No AdjO/AdjD/Luck in this format — only composite Rating + SOS
            # We already have ORtg/DRtg from CSVs, so just extract what's unique
            vals = re.findall(r"[+\-]?[\d.]+", after_wl)
            # vals[0] = Rating, vals[1] = SOS_Overall, vals[2] = SOS_Rnk,
            # vals[3] = SOS_NonConf, vals[4] = NC_Rnk, ...
            if len(vals) >= 1:
                row["NetRtg"] = float(vals[0])
            if len(vals) >= 2:
                row["SOS_NetRtg"] = float(vals[1])

        if "NetRtg" in row:
            rows.append(row)

    return pd.DataFrame(rows)


def scrape_year(scraper, url: str, year: int) -> pd.DataFrame:
    """Fetch one year's KenPom page and return a parsed DataFrame."""
    response = scraper.get(url)
    if response.status_code != 200:
        print(f"  WARNING: HTTP {response.status_code} for {year}")
        return pd.DataFrame()

    # Legacy pages (2004-2010) use <pre> text; modern pages (2011+) use HTML table.
    try:
        df = _parse_modern_table(response.text)
    except ValueError:
        df = _parse_legacy_pre(response.text)

    df["Season"] = year
    return df


def normalise_team_names(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase, strip digits, apply name mappings."""
    df["Team"] = (
        df["Team"]
        .apply(lambda x: re.sub(r"\d", "", str(x)).strip().lower())
    )
    df["Team"] = df["Team"].replace(TEAM_NAME_MAP)
    return df


def join_team_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Merge with MTeamSpellings to get TeamID."""
    spellings = pd.read_csv("data/MTeamSpellings.csv")
    df = df.merge(
        spellings,
        left_on="Team",
        right_on="TeamNameSpelling",
        how="left",
    ).drop(columns=["TeamNameSpelling"])
    return df


# ── Main ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    all_frames: list[pd.DataFrame] = []

    # ── Phase 1: Load local CSVs (2002-2018) ───────────────────────────
    csv_frames: dict[int, pd.DataFrame] = {}
    for year in CSV_YEARS:
        if year == 2020:
            continue
        try:
            df = load_csv_year(year)
            df["Team"] = df["Team"].replace(CSV_TEAM_NAME_MAP)
            csv_frames[year] = df
            print(f"Loaded {year} (CSV) → {len(df)} teams")
        except Exception as exc:
            print(f"  ERROR loading {year} CSV: {exc}")

    # ── Phase 2: Scrape archives ──────────────────────────────────────
    #   2011-2018: supplementary columns only (Luck, SOS, NCSOS) → merge into CSVs
    #   2019-2025: sole source (full data)
    scraper = cloudscraper.create_scraper(
        browser={"browser": "chrome", "platform": "windows", "mobile": False},
    )

    # Any year that has a CSV — archive data supplements (or is ignored if no new cols)
    supplement_years = set(csv_frames.keys())
    archive_frames: list[pd.DataFrame] = []

    for year in sorted(ARCHIVE_URLS):
        url = ARCHIVE_URLS[year]
        print(f"Scraping {year} (archive)…")
        try:
            df = scrape_year(scraper, url, year)
            if not df.empty:
                if year in supplement_years:
                    # Merge supplementary columns from archive into CSV data.
                    # Both sides need normalised names for matching.
                    df_norm = normalise_team_names(df.copy())
                    csv_df = csv_frames.get(year)
                    if csv_df is not None:
                        csv_norm = normalise_team_names(csv_df.copy())
                        avail = [c for c in SUPPLEMENT_COLS if c in df_norm.columns]
                        if avail:
                            supplement = df_norm[["Team", "Season"] + avail].copy()
                            merged = csv_norm.merge(
                                supplement, on=["Team", "Season"], how="left",
                            )
                            # Store already-normalised merged frame
                            csv_frames[year] = merged
                            filled = merged[avail[0]].notna().sum()
                            print(f"  → merged {len(avail)} supplement cols ({filled}/{len(merged)} matched)")
                        else:
                            print(f"  → no supplement cols found")
                    else:
                        print(f"  → no CSV for {year}, using archive as sole source")
                        archive_frames.append(df)
                else:
                    # 2019-2025: archive is the sole source
                    archive_frames.append(df)
                    print(f"  → {len(df)} teams")
        except Exception as exc:
            print(f"  ERROR: {exc}")
        sleep(np.random.random() * 2 + 1)

    # Collect CSV frames (now enriched with supplement cols for 2011-2018)
    for year in sorted(csv_frames):
        all_frames.append(csv_frames[year])

    # Add archive-only frames (2019-2025)
    all_frames.extend(archive_frames)

    # ── Phase 3: Current season from live site ──────────────────────────
    print(f"Scraping {CURRENT_YEAR} (live)…")
    try:
        df = scrape_year(scraper, LIVE_URL, CURRENT_YEAR)
        if not df.empty:
            all_frames.append(df)
            print(f"  → {len(df)} teams")
    except Exception as exc:
        print(f"  ERROR: {exc}")

    if not all_frames:
        raise RuntimeError("No data loaded — check CSV directory and network.")

    kenpom = pd.concat(all_frames, ignore_index=True)

    # ── Normalise names & join TeamID ───────────────────────────────────
    kenpom = normalise_team_names(kenpom)
    kenpom = join_team_ids(kenpom)

    matched = kenpom["TeamID"].notna().sum()
    total = len(kenpom)
    print(f"\nTeamID matched: {matched}/{total} ({matched / total:.1%})")

    unmatched = kenpom.loc[kenpom["TeamID"].isna(), "Team"].unique()
    if len(unmatched) > 0:
        print(f"Unmatched teams ({len(unmatched)}):")
        for t in sorted(unmatched):
            print(f"  {t}")

    kenpom = kenpom.dropna(subset=["TeamID"])
    kenpom["TeamID"] = kenpom["TeamID"].astype(int)
    kenpom["Season"] = kenpom["Season"].astype(int)

    out_path = "data/kenpom_pretourney.csv"
    kenpom.to_csv(out_path, index=False)
    print(f"\nSaved {len(kenpom)} rows across {kenpom['Season'].nunique()} seasons → {out_path}")
    print(f"Seasons: {sorted(kenpom['Season'].unique())}")
