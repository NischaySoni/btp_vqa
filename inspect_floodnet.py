"""Usage: python inspect_floodnet.py <vqa_folder> [<segmentation_folder>]
Prints folder layout, file counts, and a peek at question/answer files so we can
plan the pipeline. Does not modify anything."""
import sys, os, json, csv, collections

def walk(root):
    ext_count, ex = collections.Counter(), {}
    for d, _, files in os.walk(root):
        for f in files:
            e = os.path.splitext(f)[1].lower()
            ext_count[e] += 1
            ex.setdefault(e, os.path.join(d, f))
    return ext_count, ex

def peek(path, n=3):
    e = os.path.splitext(path)[1].lower()
    print(f"\n--- {path}")
    try:
        if e == ".json":
            data = json.load(open(path, encoding="utf-8"))
            if isinstance(data, dict):
                print("dict keys:", list(data.keys())[:10])
                first = next(iter(data.values()))
                print("sample value:", str(first)[:400])
            else:
                print("list length:", len(data))
                for r in data[:n]: print(str(r)[:400])
        elif e in (".csv", ".tsv", ".txt"):
            for i, line in enumerate(open(path, encoding="utf-8", errors="ignore")):
                if i >= n + 1: break
                print(line.rstrip()[:300])
    except Exception as ex:
        print("could not parse:", ex)

def main():
    for root in sys.argv[1:3]:
        print("=" * 60, "\nFOLDER:", root)
        counts, examples = walk(root)
        for e, c in counts.most_common():
            print(f"  {e or '(no ext)':10s} {c:6d}   e.g. {examples[e]}")
        for e in (".json", ".csv", ".tsv", ".txt"):
            if e in examples: peek(examples[e])
        imgs = [f for _, _, fs in os.walk(root) for f in fs
                if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        print("\nfirst image names:", sorted(imgs)[:5])

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main()