"""
build_text_corpus.py
--------------------
Reads questions.csv and answers.csv, samples 15 questions from each of 10
common E30 forum topics, filters answers to >100 characters, then writes
one .txt file per question into the text_corpus directory.
 
Usage:
    python build_text_corpus.py
 
Edit the paths below if your files live somewhere else.
"""
 
import os
import pandas as pd
 
# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = r"C:\Users\Jacob\OneDrive\Junior Year Second Semester\AIM 4420"
QUESTIONS  = os.path.join(BASE_DIR, "questions.csv")
ANSWERS    = os.path.join(BASE_DIR, "answers.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "llm_agents", "data", "text_corpus")
 
# ── Settings ───────────────────────────────────────────────────────────────────
QUESTIONS_PER_TOPIC = 15
MIN_ANSWER_CHARS    = 100
 
# ── Topic filters ──────────────────────────────────────────────────────────────
TOPIC_KEYWORDS = {
    "speaker_upgrade":      ["speaker", "speakers", "audio upgrade", "head unit", "stereo upgrade"],
    "clutch_replacement":   ["clutch", "flywheel", "pilot bearing", "clutch kit", "clutch job"],
    "engine_swap":          ["engine swap", "m50", "m52", "s52", "s14", "m20", "swap"],
    "cooling_system":       ["coolant", "radiator", "thermostat", "overheating", "water pump"],
    "suspension_upgrade":   ["suspension", "coilovers", "strut", "shock", "control arm", "bushing"],
    "soft_top_convertible": ["soft top", "convertible top", "top alignment", "power top", "vert top"],
    "tire_wheel_fitment":   ["tire size", "wheel fitment", "offset", "205/50", "195/55", "rubbing"],
    "electrical_gremlins":  ["wiring", "ground", "relay", "fusebox", "electrical issue", "no start"],
    "ews_bypass":           ["ews", "ews bypass", "ews delete", "immobilizer", "bin file"],
    "seat_swap":            ["seat swap", "seats", "seat upgrade", "seat fitment", "e36 seats"],
}
 
# ── Load ───────────────────────────────────────────────────────────────────────
print("Loading CSVs...")
questions_df = pd.read_csv(QUESTIONS, usecols=["question_id", "title", "body"])
answers_df   = pd.read_csv(ANSWERS,   usecols=["question_id", "answer_id", "body"])
 
questions_df = questions_df.rename(columns={"body": "question_body"})
answers_df   = answers_df.rename(columns={"body": "answer_body"})
 
questions_df["question_id"] = questions_df["question_id"].astype(str)
answers_df["question_id"]   = answers_df["question_id"].astype(str)
 
print(f"  {len(questions_df)} questions loaded")
print(f"  {len(answers_df)} answers loaded")
 
# ── Filter to questions that have answers ──────────────────────────────────────
answered_ids = set(answers_df["question_id"].unique())
questions_df = questions_df[questions_df["question_id"].isin(answered_ids)].copy()
 
# ── Assign topics ──────────────────────────────────────────────────────────────
body = (questions_df["question_body"].fillna("") + " " + questions_df["title"].fillna("")).str.lower()
 
questions_df["topic"] = None
for topic, keywords in TOPIC_KEYWORDS.items():
    pattern = "|".join(keywords)
    mask = body.str.contains(pattern, na=False) & questions_df["topic"].isna()
    questions_df.loc[mask, "topic"] = topic
 
# ── Sample 15 per topic ────────────────────────────────────────────────────────
print(f"\nSampling {QUESTIONS_PER_TOPIC} questions per topic...")
groups = []
for topic in TOPIC_KEYWORDS:
    subset  = questions_df[questions_df["topic"] == topic]
    sampled = subset.sample(min(QUESTIONS_PER_TOPIC, len(subset)), random_state=42)
    print(f"  {topic:<25} {len(sampled)} questions")
    groups.append(sampled)
 
sampled_df = pd.concat(groups).reset_index(drop=True)
print(f"\nTotal questions selected: {len(sampled_df)}")
 
# ── Filter answers ─────────────────────────────────────────────────────────────
sampled_ids = set(sampled_df["question_id"])
answers_filtered = answers_df[
    answers_df["question_id"].isin(sampled_ids) &
    (answers_df["answer_body"].str.len() > MIN_ANSWER_CHARS)
].copy()
print(f"Total answers kept (>{MIN_ANSWER_CHARS} chars): {len(answers_filtered)}")
 
# ── Output directory (clear old files first) ───────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
for old_file in os.listdir(OUTPUT_DIR):
    if old_file.endswith(".txt"):
        os.remove(os.path.join(OUTPUT_DIR, old_file))
print(f"  Cleared old .txt files from output directory")
 
# ── Build one file per question ────────────────────────────────────────────────
skipped = 0
written = 0
 
for _, q_row in sampled_df.iterrows():
    qid    = q_row["question_id"]
    title  = str(q_row["title"]).strip() if pd.notna(q_row["title"]) else ""
    q_body = str(q_row["question_body"]).strip() if pd.notna(q_row["question_body"]) else ""
    topic  = str(q_row["topic"])
 
    q_answers = answers_filtered[answers_filtered["question_id"] == qid].copy()
    q_answers = q_answers.sort_values("answer_id")
 
    if q_answers.empty:
        skipped += 1
        continue
 
    lines = []
    lines.append(f"QUESTION ID: {qid}")
    if title:
        lines.append(f"TITLE: {title}")
    lines.append(f"TOPIC: {topic}")
    lines.append("")
    lines.append("QUESTION:")
    lines.append(q_body if q_body else "[No question body]")
    lines.append("")
 
    for i, (_, a_row) in enumerate(q_answers.iterrows(), start=1):
        a_body = str(a_row["answer_body"]).strip() if pd.notna(a_row["answer_body"]) else "[Empty answer]"
        lines.append(f"ANSWER {i} (id: {a_row['answer_id']}):")
        lines.append(a_body)
        lines.append("")
 
    content  = "\n".join(lines).strip()
    content  = content.encode("ascii", errors="ignore").decode("ascii")
    filename = f"question_{qid}.txt"
    filepath = os.path.join(OUTPUT_DIR, filename)
 
    with open(filepath, "w", encoding="ascii") as f:
        f.write(content)
 
    written += 1
 
# ── Summary ────────────────────────────────────────────────────────────────────
print(f"\nDone.")
print(f"  Files written  : {written}")
print(f"  Skipped (no qualifying answers): {skipped}")
print(f"  Output dir     : {OUTPUT_DIR}")