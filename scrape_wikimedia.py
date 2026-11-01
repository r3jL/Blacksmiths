"""Download openly licensed training photos from Wikimedia Commons.

Only the Commons API is used (no general web scraping). Every kept file has a
known author and license, recorded in data/web_manifest.csv for disclosure.
Kept licenses: CC0, public domain, CC BY, CC BY-SA. JPEG photos only (drops
maps, diagrams, SVG/PNG). Files in AI-generated categories are skipped.

Usage: python scrape_wikimedia.py --per-class 150 [--contact you@example.com]
       python scrape_wikimedia.py --set test --per-class 40 --per-query 8   (held-out web test set)
"""
import argparse
import csv
import hashlib
import re
import time
from pathlib import Path

import requests

API = "https://commons.wikimedia.org/w/api.php"
THUMB_WIDTH = 500  # a standard Commons thumbnail width; CLIP only needs 224px

# Manifest columns, kept fixed so an interrupted first run still writes a usable header.
FIELDS = ["file", "class", "title", "author", "license", "source_url", "query"]

QUERIES = {
    "road_damage": ["pothole India", "pothole road", "pothole Delhi", "damaged road India",
                    "broken road", "road crack asphalt", "cracked pavement", "road damage"],
    "garbage": ["garbage dump India", "garbage Delhi", "overflowing garbage bin",
                "litter street India", "waste dumping roadside", "trash pile street",
                "garbage India street", "solid waste India"],
    "water_issue": ["waterlogging Delhi", "waterlogged road", "flooded street India",
                    "flooded road monsoon", "waterlogging India", "burst water pipe",
                    "water main break", "flooded street"],
    "streetlight": ["street light India", "street lights Delhi", "street lamp", "streetlight pole",
                    "broken street light", "street light at night", "lamp post street"],
    "pollution": ["air pollution Delhi", "burning garbage", "waste burning India",
                  "Yamuna pollution", "polluted river India", "stubble burning",
                  "smoke pollution factory", "smog Delhi"],
    "not_civic": ["Delhi street", "Connaught Place", "Delhi Metro station", "Indian thali",
                  "portrait India", "car parked street India", "shop front India",
                  "living room interior", "India Gate", "highway India", "building Delhi",
                  "classroom India", "selfie", "Delhi market", "auto rickshaw Delhi"],
}

# Held-out web test set: different queries (other Indian cities), and files or
# photographers already in the training manifest are excluded.
TEST_QUERIES = {
    "road_damage": ["potholes Mumbai", "pothole Bangalore", "road damaged monsoon", "pot holes",
                    "road cave in", "road in bad condition India", "pothole Kolkata",
                    "potholes", "potholed road", "pothole repair", "damaged road", "cracked road",
                    "broken road India", "road damage India", "asphalt damage"],
    "garbage": ["garbage Mumbai", "garbage Bangalore", "garbage heap", "littering India",
                "waste on street", "garbage Kolkata", "dumping ground"],
    "water_issue": ["flood Mumbai street", "waterlogging Mumbai", "flood Chennai street",
                    "waterlogging Bangalore", "water logging", "pipeline leak", "flooded underpass",
                    "flooded street India", "Kerala flood road", "Assam flood", "flooding road",
                    "water main burst", "street flooding", "flood Delhi"],
    "streetlight": ["street lamp India", "streetlight Mumbai", "lamp post road", "street light pole road",
                    "street lights night India", "solar street light"],
    "pollution": ["garbage burning", "smoke pollution India", "river pollution foam", "Bellandur lake",
                  "air pollution India", "chimney smoke", "open burning"],
    "not_civic": ["Mumbai street", "Bangalore street", "Indian wedding", "Indian street food",
                  "temple India", "office interior", "students India", "cricket India",
                  "Kolkata tram", "Jaipur", "laptop computer", "dog India"],
}

# Search also matches descriptions, so require a class keyword in the title or
# categories. not_civic must instead NOT look like any civic class.
KEYWORDS = {
    "road_damage": r"pothole|crack|damaged road|broken road|road damage|bad road|road condition|road repair",
    "garbage": r"garbage|trash|litter|waste|rubbish|dump|landfill|dustbin",
    "water_issue": r"flood|waterlog|water.?logg|inundat|burst|leak|water main|pipe",
    "streetlight": r"street.?light|street.?lamp|lamp.?post|street lighting|lamppost",
    "pollution": r"pollut|smog|smoke|burning|foam|haze|stubble|effluent",
}
CIVIC_ANY = re.compile("|".join(KEYWORDS.values()), re.I)


def on_topic(cls, title, categories):
    text = f"{title} {categories}"
    if cls == "not_civic":
        return not CIVIC_ANY.search(text)
    return re.search(KEYWORDS[cls], text, re.I) is not None


OK_LICENSE = re.compile(r"^(cc0|public domain|pd|cc[ -]by(-sa)?([ -][\d.]+)?)", re.I)
BAD_LICENSE = re.compile(r"\b(nc|nd)\b|non-?commercial|no ?deriv", re.I)
AI_CATEGORY = re.compile(r"ai-generated|generated by ai|midjourney|stable diffusion|dall-e", re.I)


def strip_html(s):
    return re.sub(r"<[^>]+>", "", s or "").strip()


def get(session, params, tries=5):
    for i in range(tries):
        r = session.get(API, params=params, timeout=30)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(5 * (i + 1))
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"API kept failing: {r.status_code}")


def search(session, query, limit):
    """Yield imageinfo dicts for File: pages matching the query."""
    params = {
        "action": "query", "format": "json", "generator": "search",
        "gsrsearch": f"{query} filetype:bitmap", "gsrnamespace": 6, "gsrlimit": 50,
        "prop": "imageinfo", "iiprop": "url|mime|extmetadata|size",
        "iiurlwidth": THUMB_WIDTH,
        "iiextmetadatafilter": "LicenseShortName|Artist|Categories|ImageDescription",
    }
    seen = 0
    while seen < limit:
        data = get(session, params)
        for page in data.get("query", {}).get("pages", {}).values():
            if "imageinfo" in page:
                seen += 1
                yield page["title"], page["imageinfo"][0]
        if "continue" not in data:
            return
        params.update(data["continue"])
        time.sleep(1)


def download(session, url, dest, tries=5):
    for i in range(tries):
        r = session.get(url, timeout=30)
        if r.status_code == 429:
            time.sleep(10 * (i + 1))
            continue
        r.raise_for_status()
        dest.write_bytes(r.content)
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=150)
    ap.add_argument("--per-query", type=int, default=0, help="cap per query (default: auto)")
    ap.add_argument("--set", choices=["train", "test"], default="train")
    ap.add_argument("--contact", default="student hackathon project, Forge26",
                    help="contact info for the User-Agent (Wikimedia API policy)")
    args = ap.parse_args()

    session = requests.Session()
    session.headers["User-Agent"] = f"Forge26CivicClassifier/0.1 ({args.contact}) python-requests"
    out = Path("data/web" if args.set == "train" else "data/webtest")
    manifest_path = Path("data/web_manifest.csv" if args.set == "train" else "data/webtest_manifest.csv")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: keep rows already downloaded on a previous run.
    rows, have_titles = [], set()
    if manifest_path.exists():
        with manifest_path.open() as f:
            rows = list(csv.DictReader(f))
        have_titles = {r["title"] for r in rows}
    banned_authors = set()
    if args.set == "test":
        train_manifest = Path("data/web_manifest.csv")
        if not train_manifest.exists():
            raise SystemExit("data/web_manifest.csv is missing; build the training set first.")
        with train_manifest.open() as f:
            train_rows = list(csv.DictReader(f))
        have_titles |= {r["title"] for r in train_rows}
        banned_authors = {r["author"].strip().lower() for r in train_rows if r["author"].strip()}

    for cls, queries in (QUERIES if args.set == "train" else TEST_QUERIES).items():
        (out / cls).mkdir(parents=True, exist_ok=True)
        count = sum(r["class"] == cls for r in rows)
        per_query = args.per_query or max(30, args.per_class // len(queries) + 10)
        for q in queries:
            if count >= args.per_class:
                break
            got_q = 0
            for title, info in search(session, q, limit=per_query * 4):
                if count >= args.per_class or got_q >= per_query:
                    break
                if title in have_titles or info.get("mime") != "image/jpeg":
                    continue
                if min(info.get("width", 0), info.get("height", 0)) < 200:
                    continue
                meta = info.get("extmetadata", {})
                lic = strip_html(meta.get("LicenseShortName", {}).get("value"))
                if not OK_LICENSE.match(lic) or BAD_LICENSE.search(lic):
                    continue
                categories = meta.get("Categories", {}).get("value", "")
                if AI_CATEGORY.search(categories) or not on_topic(cls, title, categories):
                    continue
                author = strip_html(meta.get("Artist", {}).get("value"))[:200]
                if author.strip().lower() in banned_authors:
                    continue
                url = info.get("thumburl") or info["url"]
                name = f"{cls}__web-{hashlib.md5(title.encode()).hexdigest()[:10]}.jpg"
                try:
                    if not download(session, url, out / cls / name):
                        continue
                except requests.RequestException as e:
                    print(f"  skip {title}: {e}")
                    continue
                have_titles.add(title)
                rows.append({
                    "file": f"{cls}/{name}", "class": cls, "title": title,
                    "author": author,
                    "license": lic, "source_url": info.get("descriptionurl", ""),
                    "query": q,
                })
                count += 1
                got_q += 1
                time.sleep(0.3)
            print(f"  {cls} <- '{q}': +{got_q}")
            with manifest_path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                w.writeheader()
                w.writerows(rows)
        print(f"== {cls}: {count} images")


if __name__ == "__main__":
    main()
