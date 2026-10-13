"""Usage: /media/data_dump/Anwar/btp_gen_venv/bin/python flood_datasets/ai_generated/generate.py [--seeds 2] [--limit 0]
Step 3 of the AI-generated set: specs.jsonl -> images/<spec id>_s<seed>.png and candidates.csv (for annotate.py --dir
flood_datasets/ai_generated). Model: SDXL base 1.0 (CreativeML Open RAIL++-M, not gated) with the fp16-safe VAE, so it
also runs on 2080 Tis (fp16); bf16 on Ampere. The venv adds diffusers 0.36 / accelerate 1.12 on top of the system
torch (python -m venv --system-site-packages), leaving the shared Anaconda env untouched.
Every image must still be checked by a human (annotate.py), because the generator does not reliably follow counts."""
import os, csv, json, argparse
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL, VAE = "stabilityai/stable-diffusion-xl-base-1.0", "madebyollin/sdxl-vae-fp16-fix"
NEG = "cartoon, painting, illustration, drawing, 3d render, text, watermark, logo, blurry, distorted, duplicated houses"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=6.0); a = ap.parse_args()
    from diffusers import StableDiffusionXLPipeline, AutoencoderKL
    dt = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    vae = AutoencoderKL.from_pretrained(VAE, torch_dtype=dt)
    pipe = StableDiffusionXLPipeline.from_pretrained(MODEL, vae=vae, torch_dtype=dt, variant="fp16",
                                                     use_safetensors=True).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    specs = [json.loads(l) for l in open(f"{HERE}/specs.jsonl")][: a.limit or None]
    os.makedirs(f"{HERE}/images", exist_ok=True); rows = []
    for k, s in enumerate(specs):
        for seed in range(a.seeds):
            iid = f"{s['id']}_s{seed}"; p = f"{HERE}/images/{iid}.png"
            if not os.path.exists(p):
                g = torch.Generator("cuda").manual_seed(seed)
                pipe(s["prompt"], negative_prompt=NEG, width=1216, height=832, num_inference_steps=a.steps,
                     guidance_scale=a.guidance, generator=g).images[0].save(p)
            rows.append(dict(id=iid, title=f"{s['article']} [{s['scene'].get('place', '')}]",
                             description=f"PROMPT: {s['prompt']} | INTENDED: {json.dumps(s['intended'])}"))
        print(f"[{k + 1}/{len(specs)}] {s['id']}", flush=True)
    with open(f"{HERE}/candidates.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "title", "description"]); w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} images -> images/, candidates.csv")

if __name__ == "__main__":
    main()
