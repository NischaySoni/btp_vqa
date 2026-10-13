"""Usage: python flood_datasets/ai_generated/fetch_news.py
Step 1 of the AI-generated set: collects flood news articles from Wikinews (CC BY 2.5, so text can be quoted and
re-used) into news.jsonl: title, date, url, plain text. Later steps turn each article into scene specs (make_specs.py)
and images (generate.py). ReliefWeb would be larger but its API now needs an approved appname."""
import os, json, time, urllib.parse, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
API = "https://en.wikinews.org/w/api.php"
UA = "btp-flood-vqa-research/0.1 (academic dataset collection; https://github.com/NischaySoni/btp_vqa)"
CATS = ["Floods", "Flash floods"]
last = [0.0]

def get(**params):
    body = urllib.parse.urlencode(dict(params, format="json", maxlag=5)).encode()
    for k in range(6):
        time.sleep(max(0, 1.0 - (time.time() - last[0]))); last[0] = time.time()
        try:
            req = urllib.request.Request(API, data=body, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as r: return json.loads(r.read())
        except Exception as e:
            print(f"  retry {k + 1}: {e}"); time.sleep(2 ** k)
    raise IOError(f"giving up on {params}")

def pages(cat, depth=1, seen=None):
    seen = seen if seen is not None else set(); out, cont = [], {}
    while True:
        d = get(action="query", list="categorymembers", cmtitle=f"Category:{cat}", cmlimit=500, **cont)
        for m in d["query"]["categorymembers"]:
            if m["ns"] == 0: out.append(m["title"])
            elif m["ns"] == 14 and depth > 0 and m["title"] not in seen:
                seen.add(m["title"]); out += pages(m["title"][9:], depth - 1, seen)
        if "continue" not in d: return out
        cont = d["continue"]

def main():
    titles = sorted({t for c in CATS for t in pages(c)} - {""})
    titles = [t for t in titles if not t.startswith("Wikinews Shorts")]
    print(len(titles), "articles")
    rows = []
    for t in titles:                                   # whole-article extracts come back for one page per request
        d = get(action="query", prop="extracts|info", explaintext=1, inprop="url", titles=t)
        for p in d["query"]["pages"].values():
            text = p.get("extract", "")
            date = next((l for l in text.splitlines() if l.strip()), "")[:40]       # Wikinews articles open with the date
            if len(text) > 300: rows.append(dict(title=p["title"], date=date, url=p.get("fullurl", ""), text=text,
                                                 licence="CC BY 2.5 (Wikinews)"))
    with open(f"{HERE}/news.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} articles with text -> news.jsonl")

if __name__ == "__main__":
    main()
