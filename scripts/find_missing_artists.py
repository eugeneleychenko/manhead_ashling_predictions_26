#!/usr/bin/env python3
"""Find artists missing from artist_metadata.csv and output a CSV to fill in."""
import pandas as pd
import sys, os, re, csv

MASTER = sys.argv[1]  # Training_dataset_continuosly_updated.csv
META = sys.argv[2]    # artist_metadata.csv
OUT = sys.argv[3]     # output CSV

df = pd.read_csv(MASTER, low_memory=False, usecols=["artistName", "Genre", "Instagram"])
missing_names = sorted(df[df["Genre"].isna()]["artistName"].unique())

meta = pd.read_csv(META)
genres = sorted(meta["Genre"].dropna().unique())
print(f"Existing genres: {', '.join(genres)}")
print(f"Artists with metadata: {len(meta)}")
print(f"Artists missing metadata: {len(missing_names)}")

def norm(s):
    s = str(s).strip().lower()
    s = re.sub(r"\(mh\)|\(manhead\)", "", s)
    s = re.sub(r"[^a-z0-9 ]", "", s).strip()
    s = re.sub(r"\s+", " ", s)
    return s

meta_lookup = {}
for _, row in meta.iterrows():
    meta_lookup[norm(row["artistName"])] = row

rows = []
matched = 0
for name in missing_names:
    match = meta_lookup.get(norm(name))
    if match is not None:
        g = match["Genre"] if pd.notna(match["Genre"]) else ""
        ig_col = "Instagram_followers" if "Instagram_followers" in match.index else "Instagram"
        ig = int(match[ig_col]) if ig_col in match.index and pd.notna(match[ig_col]) else 0
        rows.append({"artistName": name, "Genre": g, "Instagram_followers": ig, "note": f"auto-matched from: {match['artistName']}"})
        matched += 1
    else:
        rows.append({"artistName": name, "Genre": "", "Instagram_followers": 0, "note": ""})

print(f"Auto-matched: {matched}")
print(f"Need manual fill: {len(missing_names) - matched}")

out_df = pd.DataFrame(rows)
out_df.to_csv(OUT, index=False)
print(f"Wrote: {OUT}")
