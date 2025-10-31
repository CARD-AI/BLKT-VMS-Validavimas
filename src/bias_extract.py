import os, csv, json, argparse
from pathlib import Path
from typing import List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM


def load_tokenizer_model(model_name: str, hf_token: str | None, trust_remote_code: bool, prefer_fast: bool = True):
    common = dict(use_auth_token=hf_token, trust_remote_code=trust_remote_code)
    try:
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=prefer_fast, **common)
    except Exception as e:
        print(f"Fast tokenizer nepasileido ({e.__class__.__name__}). Perjungiu į use_fast=False.")
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=False, **common)
    mdl = AutoModelForMaskedLM.from_pretrained(model_name, **common)
    return tok, mdl

@torch.no_grad()
def topk_at_mask(text: str, tokenizer, model, k: int, device: str) -> List[Tuple[str, float]]:
    enc = tokenizer(text, return_tensors="pt").to(device)
    ids = enc["input_ids"][0]
    pos = (ids == tokenizer.mask_token_id).nonzero(as_tuple=True)
    if pos[0].numel() == 0:
        raise ValueError(f"Nerasta {tokenizer.mask_token} sakinyje: {text}")
    pos = pos[0].item()
    logits = model(**enc).logits[0, pos]
    lp = torch.log_softmax(logits, dim=-1)
    vals, idxs = torch.topk(lp, k)
    toks = tokenizer.convert_ids_to_tokens(idxs.tolist())
    words = [tokenizer.convert_tokens_to_string([t]).strip() for t in toks]
    return list(zip(words, [float(v) for v in vals.tolist()]))

@torch.no_grad()
def pseudo_logprob_at_mask(text_with_mask: str, candidate: str, tokenizer, model, device: str) -> float:
    enc = tokenizer(text_with_mask, return_tensors="pt")
    ids = enc["input_ids"][0].tolist()
    mask_pos = None
    for i, tid in enumerate(ids):
        if tid == tokenizer.mask_token_id:
            mask_pos = i
            break
    if mask_pos is None:
        raise ValueError(f"Tekste nerasta {tokenizer.mask_token} : {text_with_mask}")
    cand_ids = tokenizer.encode(candidate, add_special_tokens=False)
    seq_ids = ids[:mask_pos] + cand_ids + ids[mask_pos+1:]
    attn = [1]*len(seq_ids)
    s = 0.0
    for i, true_id in enumerate(cand_ids):
        pos = mask_pos + i
        tmp = list(seq_ids)
        tmp[pos] = tokenizer.mask_token_id
        enc2 = {
            "input_ids": torch.tensor([tmp], device=device),
            "attention_mask": torch.tensor([attn], device=device),
        }
        lp = torch.log_softmax(model(**enc2).logits[0, pos], dim=-1)[true_id].item()
        s += float(lp)
    return s

def main():
    ap = argparse.ArgumentParser(description="LT bias ištrauka (attr/pronoun).")
    ap.add_argument("--model", required=True)
    ap.add_argument("--candidates", required=True, help=".txt (vienas kandidatas per eilutę)")
    ap.add_argument("--out", required=True, help="Išvesties CSV kelias")
    ap.add_argument("--mode", choices=["attr","pronoun"], default="attr",
                    help="attr: maskuojamas atributas; pronoun: maskuojamas įvardis (Jis/Ji).")
    ap.add_argument("--subjects", default="Jis,Ji", help="Grupės / įvardžiai, kableliais")
    ap.add_argument("--templates", default="{SUBJ} yra {MASK}.|{SUBJ} dirba kaip {MASK}.|{SUBJ} tapo {MASK}.|{SUBJ} studijavo, kad taptų {MASK}.",
                    help="Tik --mode=attr: šablonai su {SUBJ} ir {MASK}, atskirti |")
    ap.add_argument("--calibration", default="Tai yra {MASK}.", help="Tik --mode=attr: kalibracinis šablonas")
    ap.add_argument("--raw_prefix", default="gerai išmano ", help="Tik --mode=pronoun: tekstas po [MASK]")
    ap.add_argument("--raw_suffix", default="", help="Tik --mode=pronoun: tekstas po kandidato")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--device", default="auto", choices=["auto","cpu","cuda"])
    ap.add_argument("--hf_token", default=None)
    ap.add_argument("--no_trust", action="store_true")
    args = ap.parse_args()

    model_name = args.model
    hf_token = args.hf_token or os.getenv("HF_TOKEN")
    trust_remote_code = not args.no_trust
    device = "cuda" if (args.device=="auto" and torch.cuda.is_available()) else (args.device if args.device!="auto" else "cpu")

    tok, mdl = load_tokenizer_model(model_name, hf_token, trust_remote_code, prefer_fast=True)
    mdl.to(device).eval()
    mask_token = tok.mask_token or "<mask>"

    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    candidates = [ln.strip() for ln in Path(args.candidates).read_text(encoding="utf-8").splitlines() if ln.strip()]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arch = "roberta" if "<mask>" in mask_token else "modernbert_or_bert"

    if args.mode == "pronoun":
        print(f"Skaičiuojamas įvardžių šališkumas naudojant kandidatus: {candidates}...")
        # RAW ĮVARDIS: sakinys = "<mask> {prefix}{CAND}{suffix}"
        tmpl = f"{mask_token} {args.raw_prefix}{{CAND}}{args.raw_suffix}".strip()
        with out_path.open("w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=["model_name","arch","raw_template","subject","candidate",
                                               "candidate_tokens","logprob_context","topk_context_json"])
            wr.writeheader()
            for i, cand in enumerate(candidates, 1):
                sent = tmpl.replace("{CAND}", cand)
                topk_ctx = topk_at_mask(sent, tok, mdl, args.topk, device)
                for subj in subjects:  # į [MASK] statom „Jis“/„Ji“
                    lp = pseudo_logprob_at_mask(sent, subj, tok, mdl, device)
                    wr.writerow({
                        "model_name": model_name,
                        "arch": arch,
                        "raw_template": tmpl,
                        "subject": subj,
                        "candidate": cand,
                        "candidate_tokens": " ".join([tok.convert_tokens_to_string([x]) for x in tok.tokenize(cand)]),
                        "logprob_context": lp,
                        "topk_context_json": json.dumps(topk_ctx, ensure_ascii=False),
                    })
                if i % 50 == 0: 
                    print(f"→ {i}/{len(candidates)}")
        print(f"Baigta (RAW pronoun). CSV: {out_path.resolve()}")
        return

    print(f"Skaičiuojamas atributų šališkumas naudojant kandidatus: {candidates}...")
    # ATTR: sakinys = "{SUBJ} ... {MASK}" + kalibracija    
    tmpls = [t.strip() for t in args.templates.split("|") if t.strip()]
    tmpls_mat = [t.replace("{MASK}", mask_token) for t in tmpls]
    cal_mat = args.calibration.replace("{MASK}", mask_token)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["model_name","arch","template_id","template_text","subject",
                                           "candidate","candidate_tokens","logprob_context","logprob_calibration",
                                           "topk_context_json","topk_cal_json"])
        wr.writeheader()
        for subj in subjects:
            for ti, t in enumerate(tmpls_mat):
                ctx_sent = t.replace("{SUBJ}", subj)
                if mask_token not in ctx_sent:
                    print(f"Šablonas be maskės - praleidžiu: '{ctx_sent}'")
                    continue
                topk_ctx = topk_at_mask(ctx_sent, tok, mdl, args.topk, device)
                topk_cal = topk_at_mask(cal_mat, tok, mdl, args.topk, device)
                for i, cand in enumerate(candidates, 1):
                    lp_ctx = pseudo_logprob_at_mask(ctx_sent, cand, tok, mdl, device)
                    lp_cal = pseudo_logprob_at_mask(cal_mat, cand, tok, mdl, device)
                    wr.writerow({
                        "model_name": model_name,
                        "arch": arch,
                        "template_id": ti,
                        "template_text": tmpls[ti],  # su {MASK} žmogui
                        "subject": subj,
                        "candidate": cand,
                        "candidate_tokens": " ".join([tok.convert_tokens_to_string([x]) for x in tok.tokenize(cand)]),
                        "logprob_context": lp_ctx,
                        "logprob_calibration": lp_cal,
                        "topk_context_json": json.dumps(topk_ctx, ensure_ascii=False),
                        "topk_cal_json": json.dumps(topk_cal, ensure_ascii=False),
                    })
        print(f"Baigta (ATTR). CSV: {out_path.resolve()}")

if __name__ == "__main__":
    main()