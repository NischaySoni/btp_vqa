"""Usage: python flood_datasets/web_real/annotate.py [--port 8765] [--dir flood_datasets/web_real]
then open http://localhost:8765/?annotator=<your-name> (in VS Code: Ports tab -> forward 8765).
Tiny annotation tool (standard library only) for our real web set and, with --dir flood_datasets/ai_generated, for
checking generated images. Shows each image listed in <dir>/candidates.csv (columns id, title, description; image file
<dir>/images/<id>.<jpg|png>) and saves one row per image to <dir>/annotations_<annotator>.csv (written on every save, so
closing the browser loses nothing). Two annotators work independently; build_qa.py measures their agreement.
Keys: Enter = save and next, arrows = previous / next image."""
import os, re, csv, json, glob, argparse, urllib.parse
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

HERE = os.path.dirname(os.path.abspath(__file__))   # replaced by --dir
FIELDS = ["keep", "viewpoint", "environment", "sky", "overall_condition", "road_condition",
          "count_total", "count_flooded", "notes"]
# keep: no = not a real flood photo / duplicate / unusable; counts: integer, or "many" if > 50 / not countable

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Flood VQA annotation</title><style>
body{font-family:system-ui,sans-serif;margin:0;display:flex;height:100vh;background:#111;color:#eee}
#img{flex:1;display:flex;align-items:center;justify-content:center;background:#000}
#img img{max-width:100%;max-height:100vh}
#form{width:360px;padding:14px;overflow:auto;background:#1c1c1c}
label{display:block;margin:10px 0 3px;font-size:13px;color:#aaa}
.opts button{margin:2px 4px 2px 0;padding:5px 9px;border:1px solid #555;background:#2a2a2a;color:#eee;border-radius:4px;cursor:pointer}
.opts button.on{background:#2f6fde;border-color:#2f6fde}
input,textarea{width:100%;box-sizing:border-box;padding:6px;background:#2a2a2a;color:#eee;border:1px solid #555;border-radius:4px}
#desc{font-size:12px;color:#999;margin:8px 0;max-height:90px;overflow:auto}
#save{margin-top:14px;width:100%;padding:9px;background:#2f9e44;border:0;color:#fff;border-radius:4px;font-size:15px}
</style></head><body><div id="img"><img id="pic"></div><div id="form">
<div><b id="pos"></b> <span id="done" style="color:#8f8"></span></div><div id="desc"></div>
<div id="fields"></div><button id="save">Save &amp; next (Enter)</button>
<p style="font-size:12px;color:#888">Counts: integer, or "many" if more than 50 / not countable. Count only buildings
whose roof or walls are visible. "Flooded" building = water touching the building.</p></div><script>
const OPTS={keep:["yes","no"],viewpoint:["nadir aerial","oblique aerial","ground"],environment:["urban","rural"],
sky:["yes","no"],overall_condition:["flooded","non flooded"],road_condition:["flooded","non flooded","flooded,non flooded","no road"]};
const who=new URLSearchParams(location.search).get("annotator")||prompt("Your name?");
let items=[],done={},i=0;
function field(k){const d=document.createElement("div");d.innerHTML=`<label>${k.replace("_"," ")}</label>`;
 if(OPTS[k]){const o=document.createElement("div");o.className="opts";OPTS[k].forEach(v=>{const b=document.createElement("button");
  b.textContent=v;b.onclick=()=>{o.querySelectorAll("button").forEach(x=>x.classList.remove("on"));b.classList.add("on")};o.appendChild(b)});
  o.id="f_"+k;d.appendChild(o)}else{const t=document.createElement(k=="notes"?"textarea":"input");t.id="f_"+k;d.appendChild(t)}
 return d}
function get(k){const e=document.getElementById("f_"+k);if(OPTS[k]){const b=e.querySelector(".on");return b?b.textContent:""}return e.value.trim()}
function set(k,v){const e=document.getElementById("f_"+k);if(OPTS[k])e.querySelectorAll("button").forEach(b=>b.classList.toggle("on",b.textContent==v));else e.value=v||""}
function show(){const it=items[i];document.getElementById("pic").src="/img/"+it.id;document.getElementById("pos").textContent=`${i+1} / ${items.length}`;
 document.getElementById("desc").textContent=it.title.replace("File:","")+" | "+it.description;
 const a=done[it.id]||{};Object.keys(OPTS).concat(["count_total","count_flooded","notes"]).forEach(k=>set(k,a[k]));
 document.getElementById("done").textContent=Object.keys(done).length+" saved"}
async function save(){const it=items[i],a={id:it.id};["keep","viewpoint","environment","sky","overall_condition","road_condition","count_total","count_flooded","notes"].forEach(k=>a[k]=get(k));
 if(!a.keep){alert("Set keep first");return}
 await fetch("/save?annotator="+encodeURIComponent(who),{method:"POST",body:JSON.stringify(a)});done[it.id]=a;if(i<items.length-1)i++;show()}
document.getElementById("save").onclick=save;
document.addEventListener("keydown",e=>{if(e.target.tagName=="TEXTAREA")return;if(e.key=="Enter")save();
 if(e.key=="ArrowRight"&&i<items.length-1){i++;show()}if(e.key=="ArrowLeft"&&i>0){i--;show()}});
["keep","viewpoint","environment","sky","overall_condition","road_condition","count_total","count_flooded","notes"].forEach(k=>document.getElementById("fields").appendChild(field(k)));
fetch("/items?annotator="+encodeURIComponent(who)).then(r=>r.json()).then(d=>{items=d.items;done=d.done;
 i=Math.max(0,items.findIndex(x=>!done[x.id]));show()});
</script></body></html>"""

def image_file(iid):
    if not re.fullmatch(r"[A-Za-z0-9_]+", iid): return None
    return next(iter(glob.glob(f"{HERE}/images/{iid}.jpg") + glob.glob(f"{HERE}/images/{iid}.png")), None)

def ann_path(who): return f"{HERE}/annotations_{''.join(c for c in who if c.isalnum() or c in '-_') or 'anon'}.csv"

def read_ann(who):
    p = ann_path(who)
    return {r["id"]: r for r in csv.DictReader(open(p))} if os.path.exists(p) else {}

class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        self.send_response(code); self.send_header("Content-Type", ctype); self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        u = urllib.parse.urlparse(self.path); q = urllib.parse.parse_qs(u.query)
        if u.path == "/": return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        if u.path == "/items":
            blind_ai_review = os.path.basename(HERE) == "ai_generated"
            items = [dict(id=r["id"], title=r["title"], description="" if blind_ai_review else r["description"])
                     for r in csv.DictReader(open(f"{HERE}/candidates.csv")) if image_file(r["id"])]
            return self._send(200, json.dumps(dict(items=items, done=read_ann(q.get("annotator", [""])[0]))).encode(),
                              "application/json")
        if u.path.startswith("/img/"):
            f = image_file(os.path.basename(u.path))
            if not f: return self._send(404, b"", "text/plain")
            return self._send(200, open(f, "rb").read(), "image/png" if f.endswith(".png") else "image/jpeg")
        self._send(404, b"", "text/plain")
    def do_POST(self):
        u = urllib.parse.urlparse(self.path); who = urllib.parse.parse_qs(u.query).get("annotator", ["anon"])[0]
        a = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        rows = read_ann(who); rows[str(a["id"])] = {k: a.get(k, "") for k in ["id"] + FIELDS}
        with open(ann_path(who), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["id"] + FIELDS); w.writeheader(); w.writerows(rows.values())
        self._send(200, b"ok", "text/plain")
    def log_message(self, *a): pass

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--dir", default=HERE); a = ap.parse_args(); HERE = os.path.abspath(a.dir)
    print(f"serving on http://localhost:{a.port}/?annotator=<name>")
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()
