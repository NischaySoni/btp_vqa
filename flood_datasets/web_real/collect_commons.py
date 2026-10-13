"""Usage: python flood_datasets/web_real/collect_commons.py [--max_per_cat 40] [--download]
Collects candidate real (non-AI) flood photos from Wikimedia Commons for our own web dataset.
Step 1 (default): crawl seed categories + full-text searches, keep JPEGs >= 1000 px wide whose licence allows reuse
(public domain, CC0, CC BY, CC BY-SA), write candidates.csv with licence/author/source for attribution.
Step 2 (--download): fetch a <= 2048 px rendition of every candidate into images/ (id = Commons page id).
Requests are rate-limited (~1/s, exponential backoff) and carry a descriptive User-Agent per Wikimedia policy."""
import os, re, csv, time, json, argparse, urllib.parse, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
API = "https://commons.wikimedia.org/w/api.php"
UA = "btp-flood-vqa-research/0.1 (academic dataset collection; https://github.com/NischaySoni/btp_vqa)"
SEEDS = [  # (category, depth): aerial first, then events with mixed viewpoints
    ("Aerial photographs of floods", 2), ("Aerial photographs of floods in India", 1),
    ("Aerial photographs of floods in the United States", 2), ("2018 Kerala floods", 1),
    ("2022 Pakistan floods", 1), ("2023 Libya floods", 1), ("2015 South Indian floods", 1),
    ("Floods in Assam", 1), ("2021 European floods", 1), ("2024 Valencia floods", 1),
    ("Effects of Hurricane Harvey in Texas", 1), ("Flooding from Hurricane Katrina", 1),
]
SEARCHES = ["flood aerial view houses", "flooded village aerial", "drone view flood", "flooded streets aerial"]
OK_LICENCE = re.compile(r"^(public domain|pd|cc0|cc by(-sa)? [\d.]+|cc by(-sa)?)", re.I)
last = [0.0]

def get(params, tries=6):
    body = urllib.parse.urlencode(dict(params, format="json", maxlag=5)).encode()   # POST: 50 long titles overflow a URL
    for k in range(tries):
        time.sleep(max(0, 1.0 - (time.time() - last[0]))); last[0] = time.time()
        try:
            with urllib.request.urlopen(urllib.request.Request(API, data=body, headers={"User-Agent": UA}), timeout=30) as r:
                d = json.loads(r.read())
            if "error" in d and d["error"].get("code") == "maxlag": raise IOError("maxlag")
            return d
        except Exception as e:
            print(f"  retry {k + 1} ({type(e).__name__}: {e})"); time.sleep(2 ** k)
    raise IOError(f"giving up on {str(params)[:200]}")

def members(cat, kind):
    out, cont = [], {}
    while True:
        d = get(dict(action="query", list="categorymembers", cmtitle=f"Category:{cat}", cmtype=kind, cmlimit=500, **cont))
        out += [m["title"] for m in d["query"]["categorymembers"]]
        if "continue" not in d: return out
        cont = d["continue"]

def crawl(cat, depth, cap, seen):
    files = [t for t in members(cat, "file") if t.lower().endswith((".jpg", ".jpeg"))][:cap]
    got = {t: cat for t in files}
    if depth > 0:
        for sub in members(cat, "subcat"):
            sub = sub[len("Category:"):]
            if sub not in seen and not re.search(r"video|radar|satellite|map|diagram", sub, re.I):
                seen.add(sub); got.update({t: c for t, c in crawl(sub, depth - 1, cap, seen).items() if t not in got})
    return got

def search(q, n=100):
    d = get(dict(action="query", list="search", srsearch=f"{q} filetype:bitmap", srnamespace=6, srlimit=n))
    return [m["title"] for m in d["query"]["search"] if m["title"].lower().endswith((".jpg", ".jpeg"))]

def info(titles):
    rows = []
    for i in range(0, len(titles), 50):
        d = get(dict(action="query", titles="|".join(titles[i:i + 50]), prop="imageinfo",
                     iiprop="url|size|mime|extmetadata", iiurlwidth=2048))
        for p in d["query"]["pages"].values():
            ii = (p.get("imageinfo") or [None])[0]
            if not ii: continue
            md = {k: re.sub(r"<[^>]+>", "", str(v.get("value", ""))).strip() for k, v in ii.get("extmetadata", {}).items()}
            rows.append(dict(id=p["pageid"], title=p["title"], width=ii["width"], height=ii["height"], mime=ii["mime"],
                             licence=md.get("LicenseShortName", ""), artist=md.get("Artist", "")[:200],
                             date=md.get("DateTimeOriginal", "")[:40], description=md.get("ImageDescription", "")[:300],
                             page=ii.get("descriptionurl", ""), url=ii.get("thumburl") or ii["url"]))
    return rows

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--max_per_cat", type=int, default=40)
    ap.add_argument("--download", action="store_true"); a = ap.parse_args()
    out = f"{HERE}/candidates.csv"
    if not a.download:
        cache = f"{HERE}/crawl_titles.json"
        src, seen = (json.load(open(cache)), set()) if os.path.exists(cache) else ({}, set())
        for cat, depth in SEEDS if not src else []:
            print("category", cat)
            try: src.update({t: c for t, c in crawl(cat, depth, a.max_per_cat, seen).items() if t not in src})
            except IOError as e: print("  skipped:", e)
        for q in SEARCHES if not os.path.exists(cache) else []:
            print("search", q); src.update({t: f"search:{q}" for t in search(q) if t not in src})
        json.dump(src, open(cache, "w"))
        rows = [dict(r, found_via=src[r["title"]]) for r in info(sorted(src))]
        keep = [r for r in rows if r["mime"] == "image/jpeg" and r["width"] >= 1000 and OK_LICENCE.match(r["licence"])]
        print(f"{len(src)} titles -> {len(rows)} with info -> {len(keep)} usable (jpeg, >=1000 px, open licence)")
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(keep[0])); w.writeheader(); w.writerows(keep)
        return
    os.makedirs(f"{HERE}/images", exist_ok=True)
    rows = list(csv.DictReader(open(out)))
    for k, r in enumerate(rows):
        p = f"{HERE}/images/{r['id']}.jpg"
        if os.path.exists(p): continue
        for t in range(5):
            time.sleep(max(0, 1.0 - (time.time() - last[0]))); last[0] = time.time()
            try:
                with urllib.request.urlopen(urllib.request.Request(r["url"], headers={"User-Agent": UA}), timeout=60) as resp:
                    data = resp.read()
                open(p, "wb").write(data); break
            except Exception as e:
                print(f"  {r['id']} retry {t + 1}: {e}"); time.sleep(2 ** t)
        if (k + 1) % 50 == 0: print(f"downloaded {k + 1}/{len(rows)}")

if __name__ == "__main__":
    main()
