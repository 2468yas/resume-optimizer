from pathlib import Path
from sentence_transformers import SentenceTransformer, util

MODEL = SentenceTransformer("all-MiniLM-L6-v2")  # load once and reuse, loading is the slow part


def split_lines(text):
    # chop a big text into clean lines, strip bullet symbols, skip blank/tiny junk lines
    lines = [line.strip(" -•*\t") for line in text.splitlines()]
    return [line for line in lines if len(line) > 15]


def match(resume_text, job_text, weak_threshold=0.35):
    bullets = split_lines(resume_text)
    reqs = split_lines(job_text)

    bullet_emb = MODEL.encode(bullets, convert_to_tensor=True)  # every bullet -> 384 numbers
    req_emb = MODEL.encode(reqs, convert_to_tensor=True)        # every requirement -> 384 numbers

    grid = util.cos_sim(req_emb, bullet_emb)  # rows = requirements, columns = bullets

    matches = []
    for i, req in enumerate(reqs):
        best = int(grid[i].argmax())  # which bullet scored highest for this requirement
        matches.append({
            "requirement": req,
            "best_bullet": bullets[best],
            "score": round(float(grid[i][best]), 3),
        })

    overall = round(sum(m["score"] for m in matches) / len(matches), 3)  # average of best matches
    weak = [m for m in matches if m["score"] < weak_threshold]          # stuff your resume doesn't cover
    return {"overall": overall, "matches": matches, "weak_spots": weak}


if __name__ == "__main__":
    # only runs when you run this file directly (not when another file imports it)
    resume = Path("eval/private/resume.txt").read_text()
    job = Path("eval/private/job.txt").read_text()
    report = match(resume, job)

    print(f"\nOVERALL MATCH: {report['overall']}\n")
    for m in sorted(report["matches"], key=lambda m: m["score"], reverse=True):
        print(f"[{m['score']}] {m['requirement']}")
        print(f"        -> {m['best_bullet']}\n")

    print("WEAK SPOTS (nothing on your resume covers these):")
    for m in report["weak_spots"]:
        print(f"  - {m['requirement']}")