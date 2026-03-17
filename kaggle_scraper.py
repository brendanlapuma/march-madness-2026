import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
import requests
from time import sleep


def parse_kenpom_table(html: str) -> pd.DataFrame:
    """
    Parse the KenPom-style ratings table HTML into a DataFrame.

    The table has a specific structure:
    - Repeated thead rows (ignored after first parse)
    - Each data row has td elements where ranked values appear as:
        <td class="td-left">VALUE</td><td class="td-right"><span class="seed">RANK</span></td>
    - Tournament teams have class="tourney" on the tr
    - Team name td contains an anchor + optional seed span
    """
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", {"id": "ratings-table"})

    # --- Column definitions ---
    # We'll derive columns by parsing the FIRST thead only.
    # The structure is two header rows stacked; we only need thead2.
    first_thead = table.find("thead")
    header_cells = first_thead.find("tr", class_="thead2").find_all("th")

    # Build base column names from title attributes or link text
    base_cols = []
    for th in header_cells:
        title = th.get("title", "")
        link = th.find("a")
        text = link.get_text(strip=True) if link else th.get_text(strip=True)
        base_cols.append((text, title))

    # The ranked stat columns each occupy TWO td cells: value + rank.
    # Columns with colspan="2" are paired value+rank columns.
    # Fixed single columns: Rk, Team, Conf, W-L, NetRtg (the main one)
    # Everything after that is value+rank pairs.
    # From the HTML: Rk, Team, Conf, W-L, NetRtg, ORtg(val,rank), DRtg(val,rank),
    #   AdjT(val,rank), Luck(val,rank), SOS_NetRtg(val,rank), SOS_ORtg(val,rank),
    #   SOS_DRtg(val,rank), NCSOS_NetRtg(val,rank)

    COLUMNS = [
        "Rk",
        "Team",
        "Seed",
        "Conf",
        "W",
        "L",
        "NetRtg",
        "ORtg",
        "ORtg_Rk",
        "DRtg",
        "DRtg_Rk",
        "AdjT",
        "AdjT_Rk",
        "Luck",
        "Luck_Rk",
        "SOS_NetRtg",
        "SOS_NetRtg_Rk",
        "SOS_ORtg",
        "SOS_ORtg_Rk",
        "SOS_DRtg",
        "SOS_DRtg_Rk",
        "NCSOS_NetRtg",
        "NCSOS_NetRtg_Rk",
        "Tourney",
    ]

    rows = []
    for tr in table.find("tbody").find_all("tr"):
        # Skip repeated thead rows that appear mid-table
        if tr.find("th"):
            continue

        tds = tr.find_all("td")
        if len(tds) < 5:
            continue

        row = {}

        # Tourney flag
        classes = tr.get("class", [])
        row["Tourney"] = 1 if "tourney" in classes else 0

        # Rk
        row["Rk"] = int(tds[0].get_text(strip=True))

        # Team + seed
        team_td = tds[1]
        seed_span = team_td.find("span", class_="seed")
        row["Seed"] = int(seed_span.get_text(strip=True)) if seed_span else None
        # Remove seed span text before grabbing team name
        if seed_span:
            seed_span.decompose()
        row["Team"] = team_td.get_text(strip=True)

        # Conf
        row["Conf"] = tds[2].get_text(strip=True)

        # W-L
        wl = tds[3].get_text(strip=True)
        parts = wl.split("-")
        row["W"] = int(parts[0])
        row["L"] = int(parts[1])

        # NetRtg (single td, no rank pair)
        row["NetRtg"] = float(tds[4].get_text(strip=True))

        # Remaining paired (value, rank) columns
        paired = [
            ("ORtg", 5),
            ("DRtg", 7),
            ("AdjT", 9),
            ("Luck", 11),
            ("SOS_NetRtg", 13),
            ("SOS_ORtg", 15),
            ("SOS_DRtg", 17),
            ("NCSOS_NetRtg", 19),
        ]

        for col_name, td_idx in paired:
            val_td = tds[td_idx]
            rank_td = tds[td_idx + 1]
            val_text = val_td.get_text(strip=True)
            rank_span = rank_td.find("span", class_="seed")
            row[col_name] = float(val_text)
            row[f"{col_name}_Rk"] = (
                int(rank_span.get_text(strip=True)) if rank_span else None
            )

        rows.append(row)

    df = pd.DataFrame(rows, columns=COLUMNS)

    # Type cleanup
    int_cols = [
        "Rk",
        "W",
        "L",
        "Seed",
        "ORtg_Rk",
        "DRtg_Rk",
        "AdjT_Rk",
        "Luck_Rk",
        "SOS_NetRtg_Rk",
        "SOS_ORtg_Rk",
        "SOS_DRtg_Rk",
        "NCSOS_NetRtg_Rk",
    ]
    for c in int_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    float_cols = [
        "NetRtg",
        "ORtg",
        "DRtg",
        "AdjT",
        "Luck",
        "SOS_NetRtg",
        "SOS_ORtg",
        "SOS_DRtg",
        "NCSOS_NetRtg",
    ]
    for c in float_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


if __name__ == "__main__":
    BASE_KENPOM_URL = "https://kenpom.com/index.php?y={year}"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.5112.79 Safari/537.36"
    }
    kenpom_df = pd.DataFrame()
    for year in range(2002, 2027):
        if year == 2020:
            continue
        response = requests.get(BASE_KENPOM_URL.format(year=year), headers=HEADERS)
        soup = BeautifulSoup(response.text, "html.parser")
        tables = soup.find_all("table", {"id": "ratings-table"})
        if not tables:
            print(f"No ratings table found for {year}")
            continue
        year_kenpom_df = parse_kenpom_table(str(tables[0]))
        year_kenpom_df["Season"] = year
        kenpom_df = pd.concat([kenpom_df, year_kenpom_df], ignore_index=True)
        print(f"Parsed KenPom data for {year}")
        sleep(np.random.random() * 2 + 1)

    kenpom_df["Team"] = kenpom_df["Team"].apply(
        lambda x: (
            re.sub(r"\d", "", x).strip().lower()
        )
    )
    team_name_map = {
        "arkansas little rock": "arkansas-little-rock",
        "louisiana lafayette": "louisiana-lafayette",
        "southwest missouri st.": "southwest missouri state",
        "illinois chicago": "illinois-chicago",
        "texas pan american": "texas-pan american",
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
    }
    kenpom_df["Team"] = kenpom_df["Team"].replace(team_name_map)

    spellings_df = pd.read_csv("data/MTeamSpellings.csv")
    kenpom_df = kenpom_df.merge(
        spellings_df,
        left_on="Team",
        right_on="TeamNameSpelling",
        how="left",
    ).drop(columns=["TeamNameSpelling"])
    kenpom_df.to_csv("data/kenpom.csv", index=False)
    print("KenPom data saved to kenpom.csv")
