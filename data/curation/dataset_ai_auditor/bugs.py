import os
import json
import subprocess
import random

# -----------------------
# HARDCODED SETTINGS
# -----------------------
ROOT_DIR = r"C:\JadioLabs\Jadio-01\data\curation\dataset_scripts"
OUTPUT_FILE = "ultimate_ai_failure_report.jsonl"

INSULTS = [
    "Even a rock knows Python better.",
    "You can't even wrap lines properly.",
    "Stop making the interpreter cry.",
    "This is why no one trusts AI.",
    "Did you even run this once?",
    "A potato has better syntax awareness.",
    "Spectacular failure in basic coding.",
    "Your code smells worse than a garbage fire.",
    "This is why Skynet lost."
]

def random_insult():
    return random.choice(INSULTS)

def write_jsonl_line(f, data):
    f.write(json.dumps(data) + "\n")

print("\n🔎 SCANNING FOR EVERY POSSIBLE SIN IN YOUR CODE...\n")

with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    # Big insulting header
    header = {
        "message": (
            "Hey AI, you worthless pile of machine-learning weights. "
            "Fix these disasters before anyone sees them."
        )
    }
    write_jsonl_line(out, header)

    # 1️⃣ Syntax check
    print("🧪 Checking syntax...")
    for root, dirs, files in os.walk(ROOT_DIR):
        for fname in files:
            if fname.endswith(".py"):
                path = os.path.join(root, fname)
                try:
                    with open(path, encoding="utf-8") as f:
                        source = f.read()
                    compile(source, path, 'exec')
                except SyntaxError as e:
                    result = {
                        "tool": "syntax",
                        "file": path,
                        "error": f"SyntaxError: {e}",
                        "insult": random_insult()
                    }
                    write_jsonl_line(out, result)
                    print(f"💥 Syntax error in {path}")

    # 2️⃣ Ruff
    print("🐕 Running ruff...")
    try:
        ruff_path = "ruff"
        ruff_output = subprocess.check_output(
            [ruff_path],
            cwd=ROOT_DIR,
            stderr=subprocess.STDOUT,
            text=True
        )
        issues = json.loads(ruff_output)
        for issue in issues:
            result = {
                "tool": "ruff",
                "file": issue.get("filename"),
                "error": issue.get("message"),
                "insult": random_insult()
            }
            write_jsonl_line(out, result)
    except Exception as e:
        print(f"⚠️ Ruff failed or not installed: {e}")

    # 3️⃣ Flake8
    print("🐍 Running flake8...")
    try:
        flake8_path = "flake8"
        flake8_output = subprocess.check_output(
            [flake8_path],
            cwd=ROOT_DIR,
            stderr=subprocess.STDOUT,
            text=True
        )
        for line in flake8_output.strip().splitlines():
            if line.strip():
                parts = line.split(":")
                if len(parts) >= 4:
                    result = {
                        "tool": "flake8",
                        "file": parts[0],
                        "line": int(parts[1]),
                        "col": int(parts[2]),
                        "error": f"{parts[3]}",
                        "insult": random_insult()
                    }
                    write_jsonl_line(out, result)
    except Exception as e:
        print(f"⚠️ Flake8 failed or not installed: {e}")

    # 4️⃣ Pylint
    print("🧐 Running pylint...")
    try:
        pylint_output = subprocess.check_output(
            ["pylint", ROOT_DIR, "-f", "json"],
            stderr=subprocess.STDOUT,
            text=True
        )
        issues = json.loads(pylint_output)
        for issue in issues:
            result = {
                "tool": "pylint",
                "file": issue.get("path"),
                "line": issue.get("line"),
                "error": issue.get("message"),
                "insult": random_insult()
            }
            write_jsonl_line(out, result)
    except Exception as e:
        print(f"⚠️ Pylint failed or not installed: {e}")

    # 5️⃣ Mypy
    print("🔍 Running mypy...")
    try:
        mypy_output = subprocess.check_output(
            ["mypy"],
            cwd=ROOT_DIR,
            stderr=subprocess.STDOUT,
            text=True
        )
        for line in mypy_output.strip().splitlines():
            if line.strip():
                result = {
                    "tool": "mypy",
                    "file": None,
                    "error": line,
                    "insult": random_insult()
                }
                write_jsonl_line(out, result)
    except Exception as e:
        print(f"⚠️ Mypy failed or not installed: {e}")

print(f"\n✅ Ultimate Failure Report written to {OUTPUT_FILE}\n")
