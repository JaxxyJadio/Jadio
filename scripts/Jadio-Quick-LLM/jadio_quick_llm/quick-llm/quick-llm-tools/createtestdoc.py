# tools/createtestdoc.py
# Simple tool: creates a README_TEST.md in the project root with 'hello world'
import os

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = os.path.join(root, "README_TEST.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("hello world\n")
    print(f"README_TEST.md created at {out_path}")

if __name__ == "__main__":
    main()
