"""Usage: python flood_datasets/ai_generated/make_specs.py [--scenes scenes_claude.json] [--per_article 3] [--seed 0] [--limit 0]
Step 2 of the AI-generated set: news.jsonl -> specs.jsonl (one row per image to generate).
A local LLM (Qwen2.5-7B-Instruct; GPU if available, else all CPU cores) reads each Wikinews article and extracts up to
3 visually concrete flood scenes (place, setting, building style, how the water looks, a supporting quote), or none if
the article describes no physical scene. With --scenes, scenes are read from a JSON file ({article index: [scene...]})
instead, e.g. scenes_claude.json, extracted by Claude under the same rules while the GPUs were unavailable. The attributes that carry VQA answers (viewpoint, number of buildings, how
many are flooded, road condition) are then sampled with a fixed seed so answers are balanced and known by design.
These intended answers are only provisional: generators do not follow counts reliably, so every generated image is
re-checked by humans with web_real/annotate.py before it enters qa.jsonl."""
import os, re, json, random, argparse
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "Qwen/Qwen2.5-7B-Instruct"
ASK = """You read a news article about a flood. List up to {n} distinct flood scenes that the article describes or
clearly implies, as a photographer standing there would see them. Use only facts from the article; do not invent
places. If the article describes no physical flood scene (e.g. only politics or finance), return [].
Every field must be a SHORT VISUAL noun phrase (at most 6 words) that an image could show; no events, no numbers of
people, no sentences. Answer with a JSON list only. Each item:
{{"place": "town or region", "country": "...", "setting": "urban" or "rural",
"building_type": "plural noun phrase, e.g. single-storey tin-roofed houses",
"water": "look of the water, e.g. muddy brown water", "extras": "visible objects, e.g. submerged cars, boats",
"quote": "short supporting quote from the article"}}

ARTICLE ({title}, {date}):
{text}"""

def llm():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    if torch.cuda.is_available(): dev, dt = "cuda", torch.bfloat16
    else:
        os.sched_setaffinity(0, range(os.cpu_count())); torch.set_num_threads(os.cpu_count()); dev, dt = "cpu", torch.float32   # no bf16 in this Xeon
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dt).to(dev).eval()
    print("LLM on", dev, flush=True)
    def ask(prompt):
        enc = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True).to(dev)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=700, do_sample=False)
        return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return ask

def parse(s):
    m = re.search(r"\[.*\]", s, re.S)
    try: return [x for x in json.loads(m.group(0)) if isinstance(x, dict) and x.get("place")] if m else []
    except json.JSONDecodeError: return []

def sample(rng):
    """Answer-carrying attributes, balanced by design. Counts stay small (1-8) so humans can verify them."""
    view = rng.choice(["top-down drone", "oblique aerial"])
    n = rng.randint(1, 8); fl = rng.choice([0, n, rng.randint(0, n)])
    return dict(viewpoint=view, n_buildings=n, n_flooded=fl, road=rng.choice(["flooded", "non flooded", "none"]),
                sky="yes" if view == "oblique aerial" and rng.random() < .5 else "no")

def clip_len(text, _tok=[]):
    """Token count under SDXL's CLIP tokenizer (77-token window); the tokenizer is in the HF cache after generate.py's
    model download."""
    if not _tok:
        from transformers import CLIPTokenizer
        _tok.append(CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="tokenizer"))
    return len(_tok[0](text).input_ids)

def prompt(sc, at):
    """Answer-carrying parts first (viewpoint, count, flooding, road, sky), then context; optional parts are dropped
    until the prompt fits SDXL's 77-token text window."""
    n, fl = at["n_buildings"], at["n_flooded"]
    bt = sc.get("building_type") or "houses"
    water = sc.get("water") or "muddy flood water"
    flood = ("none of them flooded" if fl == 0 else f"all {n} surrounded by {water}" if fl == n else
             f"{fl} of them surrounded by {water}, the other {n - fl} dry")
    road = {"flooded": "road covered by flood water", "non flooded": "dry road", "none": "no road"}[at["road"]]
    sky = "horizon and sky visible" if at["sky"] == "yes" else "no sky"
    setting = sc.get("setting", "rural")
    core = f"{at['viewpoint']} photo, exactly {n} {bt} ({'one building' if n == 1 else f'{n} buildings'}), {flood}, {road}, {sky}"
    flooded_scene = fl > 0 or at["road"] == "flooded"           # flood extras would contradict a dry scene
    extras = ", ".join(e.strip() for e in sc.get("extras", "").split(",")    # road-ish extras only if the road is flooded
                       if at["road"] == "flooded" or not re.search(r"road|street|highway|lane|bridge|traffic", e))
    for extra in (f", {setting} {sc.get('place', '')}, {sc.get('country', '')}, {extras}, realistic news photo"
                  if flooded_scene else None,
                  f", {setting} {sc.get('place', '')}, {sc.get('country', '')}, realistic news photo",
                  ", realistic news photo", ""):
        if extra is not None and clip_len(core + extra) <= 77: return core + extra
    return core

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--per_article", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--scenes", default=""); a = ap.parse_args()
    news = [json.loads(l) for l in open(f"{HERE}/news.jsonl")][: a.limit or None]
    given = json.load(open(a.scenes if os.path.isabs(a.scenes) else f"{HERE}/{a.scenes}")) if a.scenes else None
    ask, rng, rows = (None if given else llm()), random.Random(a.seed), []
    out = open(f"{HERE}/specs.jsonl", "w")
    for k, art in enumerate(news):
        scenes = (given.get(str(k), []) if given else
                  parse(ask(ASK.format(n=a.per_article, title=art["title"], date=art["date"], text=art["text"][:6000]))))
        for j, sc in enumerate(scenes[: a.per_article]):
            at = sample(rng)
            r = dict(id=f"g{k:03d}_{j}", article=art["title"], url=art["url"], scene=sc, attributes=at,
                     prompt=prompt(sc, at),
                     intended=dict(count_total=at["n_buildings"], count_flooded=at["n_flooded"],
                                   overall_condition="flooded" if at["n_flooded"] or at["road"] == "flooded" else "non flooded",
                                   road_condition=at["road"], sky_yn=at["sky"], environment=sc.get("setting", "")))
            out.write(json.dumps(r) + "\n"); out.flush(); rows.append(r)
        print(f"[{k + 1}/{len(news)}] {art['title'][:70]} -> {len(scenes)} scenes", flush=True)
    print(f"{len(rows)} specs -> specs.jsonl")

if __name__ == "__main__":
    main()
