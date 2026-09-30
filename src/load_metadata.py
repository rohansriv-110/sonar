import psycopg
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TSV = ROOT / "mtg-jamendo-dataset/data/autotagging_moodtheme.tsv"
DSN = "postgresql://postgres:sonar@localhost:5432/sonar"

def rows():
    with open(TSV, encoding="utf-8") as f:
        f.readline()  # header
        for line in f:
            p = line.rstrip("\n").split("\t")
            tags = [t.split("---")[-1] for t in p[5:]]
            yield p[0], float(p[4]), tags

with psycopg.connect(DSN) as conn:
    with conn.cursor() as cur:
        with cur.copy(
            "COPY tracks (jamendo_id, duration_sec, tags) FROM STDIN"
        ) as copy:
            for r in rows():
                copy.write_row(r)
    conn.commit()
    print(conn.execute("SELECT count(*) FROM tracks").fetchone())
