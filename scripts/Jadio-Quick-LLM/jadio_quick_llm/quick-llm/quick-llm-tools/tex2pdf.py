#!/usr/bin/env python3
"""
tex2pdf.py - Convert a LaTeX .tex file to PDF using pdflatex

Usage:
    python tex2pdf.py <input.tex>

- Requires pdflatex to be installed and in PATH.
- Outputs PDF in the same directory as the input .tex file.
- Prints clear success/failure messages.
"""
import sys
import os
import subprocess
import shlex

def find_pdflatex_on_windows():
    """Try to find pdflatex.exe in common TeX distributions on Windows."""
    import winreg
    candidates = []
    # Common TeX Live path
    candidates.append(r"C:\texlive\2023\bin\win32")
    candidates.append(r"C:\texlive\2024\bin\win32")
    # Common MiKTeX path
    candidates.append(r"C:\Program Files\MiKTeX 2.9\miktex\bin\x64")
    candidates.append(r"C:\Program Files\MiKTeX\miktex\bin\x64")
    candidates.append(r"C:\Program Files (x86)\MiKTeX 2.9\miktex\bin")
    # Check registry for MiKTeX
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\MiKTeX.org\MiKTeX\2.9") as key:
            path, _ = winreg.QueryValueEx(key, "InstallRoot")
            candidates.append(os.path.join(path, "miktex", "bin", "x64"))
    except Exception:
        pass
    for path in candidates:
        exe = os.path.join(path, "pdflatex.exe")
        if os.path.isfile(exe):
            return path
    return None

def main():
    if len(sys.argv) != 2:
        print("Usage: python tex2pdf.py <input.tex>", file=sys.stderr)
        sys.exit(1)
    tex_path = sys.argv[1]
    if not tex_path.lower().endswith('.tex'):
        print("Error: Input file must have a .tex extension.", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(tex_path):
        print(f"Error: File not found: {tex_path}", file=sys.stderr)
        sys.exit(1)
    # --- Ensure pdflatex is in PATH on Windows ---
    if os.name == 'nt':
        pdflatex_dir = find_pdflatex_on_windows()
        if pdflatex_dir and pdflatex_dir not in os.environ["PATH"]:
            os.environ["PATH"] = pdflatex_dir + os.pathsep + os.environ["PATH"]
    workdir = os.path.dirname(os.path.abspath(tex_path))
    tex_file = os.path.basename(tex_path)
    # Run pdflatex twice for references
    for i in range(2):
        try:
            result = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", tex_file],
                cwd=workdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=60
            )
        except FileNotFoundError:
            print("Error: pdflatex not found. Please install TeX Live or MiKTeX and ensure pdflatex is in your PATH.", file=sys.stderr)
            if os.name == 'nt':
                print("Tried to auto-detect TeX Live/MiKTeX in common locations.", file=sys.stderr)
            sys.exit(1)
        except subprocess.TimeoutExpired:
            print("Error: pdflatex timed out.", file=sys.stderr)
            sys.exit(1)
        if result.returncode != 0:
            print(f"pdflatex failed (pass {i+1}):\n{result.stdout.decode(errors='ignore')}", file=sys.stderr)
            sys.exit(1)
    pdf_path = os.path.splitext(tex_path)[0] + ".pdf"
    if os.path.isfile(pdf_path):
        print(f"Success: PDF generated at {pdf_path}")
        sys.exit(0)
    else:
        print("Error: PDF not generated.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
