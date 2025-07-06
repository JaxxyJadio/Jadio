import os
import json
import random

ROOT_DIR = os.environ.get('SHORTCUT_SCRIPTS_ROOT')
if not ROOT_DIR:
    raise RuntimeError('SHORTCUT_SCRIPTS_ROOT environment variable must be set.')
OUTPUT_FILE = os.environ.get('SHORTCUT_REPORT_PATH', 'hardcoded_filepaths_report.jsonl')

TARGET_EXTENSIONS = (".py", ".yaml")

INSULTS = [
    "Delete your career.",
    "Did your brain go on vacation?",
    "This is why we can't have nice things.",
    "Seriously?",
    "Even a toaster could code better.",
    "Unbelievable levels of laziness.",
    "Masterclass in bad practices.",
    "Spectacular fail."
]

def random_insult():
    return random.choice(INSULTS)

print("\n🔎 SCANNING FOR SINS AGAINST SOFTWARE ENGINEERING...\n")

with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    # Write the angry header first
    header = {
        "message": (
            "WARNING: Hey AI, you absolute dumbass. "
            "Found hardcoded .py or .yaml paths. "
            "Honestly, how are you this incompetent?"
        )
    }
    out.write(json.dumps(header) + "\n")

    found_any = False

    for root, dirs, files in os.walk(ROOT_DIR):
        # Explicitly skip dataset_ai_auditor directory
        if 'dataset_ai_auditor' in root.replace('\\', '/').replace('\\', '/').split('/'):
            continue
        for fname in files:
            if fname.endswith(".py"):
                path = os.path.join(root, fname)
                try:
                    with open(path, encoding="utf-8") as f:
                        for lineno, line in enumerate(f, start=1):
                            lowered = line.lower()
                            if any(ext in lowered for ext in TARGET_EXTENSIONS):
                                found_any = True
                                result = {
                                    "file": path,
                                    "line": lineno,
                                    "text": line.rstrip(),
                                    "note": f"Hardcoded path detected. {random_insult()}"
                                }
                                out.write(json.dumps(result) + "\n")
                                print(f"🔥 Offense found in {path} at line {lineno}")
                except Exception as e:
                    print(f"❌ Failed to read {path}: {e}")

    # Always append a final insulting reminder
    reminder = {
        "reminder": (
            "STOP HARD-CODING FILE NAMES. "
            "Use the shortcut system like an adult, you digital disgrace."
        )
    }
    out.write(json.dumps(reminder) + "\n")

print("\n✅ Report written to", OUTPUT_FILE)
if not found_any:
    print("\n👏 Surprisingly, no hardcoded .py or .yaml paths found. Miracles do happen.\n")
else:
    print("\n⚠️ Please review your sins in the JSONL report and repent.\n")
