"""
run_pipreqs.py
Copia tutti i file .py del progetto in una cartella temporanea,
ci esegue pipreqs sopra e scrive requirements.txt nella root del progetto.

Uso:
    python run_pipreqs.py                  # dalla root del progetto
    python run_pipreqs.py /path/al/progetto
    python run_pipreqs.py --mode compat    # passa flag extra a pipreqs
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Cartelle da ignorare sempre
IGNORE_DIRS = {
    ".git", ".hg", ".svn",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    ".venv", "venv", "env", ".env",
    "node_modules",
    "dist", "build", "*.egg-info",
    ".tox", ".nox", "marioenv"
}


def collect_py_files(root: Path) -> list[Path]:
    """Restituisce tutti i .py sotto root, saltando le dir da ignorare."""
    files = []
    for path in root.rglob("*.py"):
        # Salta se uno dei genitori è una dir ignorata
        parts = set(path.relative_to(root).parts[:-1])
        if parts & IGNORE_DIRS:
            continue
        files.append(path)
    return files


def mirror_to_temp(py_files: list[Path], root: Path, tmp: Path) -> None:
    """
    Ricrea la struttura delle directory nel tmp e copia solo i .py.
    pipreqs usa i nomi dei moduli importati — la struttura non è
    strettamente necessaria, ma aiuta se ci sono import relativi.
    """
    for src in py_files:
        rel = src.relative_to(root)
        dst = tmp / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def find_pipreqs() -> str:
    """Trova l'eseguibile pipreqs accanto al Python corrente, o nel PATH."""
    # Prima cerca nella stessa cartella del Python in uso (venv)
    candidate = Path(sys.executable).parent / "pipreqs"
    if candidate.exists():
        return str(candidate)
    # Fallback: cerca nel PATH di sistema
    found = shutil.which("pipreqs")
    if found:
        return found
    print("Errore: 'pipreqs' non trovato. Installalo con: pip install pipreqs", file=sys.stderr)
    sys.exit(1)


def run_pipreqs(tmp: Path, out: Path, extra_args: list[str]) -> int:
    """Lancia pipreqs e scrive requirements.txt in `out`."""
    pipreqs_bin = find_pipreqs()
    cmd = [
        pipreqs_bin,
        str(tmp),
        "--savepath", str(out),
        "--force",          # sovrascrive se esiste già
        *extra_args,
    ]
    print(f"\n$ {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Esegui pipreqs solo sui file .py")
    parser.add_argument(
        "project_dir",
        nargs="?",
        default=".",
        help="Root del progetto (default: directory corrente)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Percorso di output per requirements.txt (default: <project_dir>/requirements.txt)",
    )
    parser.add_argument(
        "--mode",
        choices=["no-pin", "compat", "gt", "non-pin"],
        default=None,
        help="Modalità versioning di pipreqs (opzionale)",
    )
    parser.add_argument(
        "--encoding",
        default=None,
        help="Encoding dei file sorgente (es. utf-8, latin-1)",
    )
    args = parser.parse_args()

    root = Path(args.project_dir).resolve()
    if not root.is_dir():
        print(f"Errore: '{root}' non è una directory valida.", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.out).resolve() if args.out else root / "requirements.txt"

    # Argomenti extra da passare a pipreqs
    extra: list[str] = []
    if args.mode:
        extra += ["--mode", args.mode]
    if args.encoding:
        extra += ["--encoding", args.encoding]

    # --- Raccolta file .py ---
    print(f"Progetto: {root}")
    py_files = collect_py_files(root)
    if not py_files:
        print("Nessun file .py trovato. Uscita.")
        sys.exit(0)
    print(f"Trovati {len(py_files)} file .py (cartelle ignorate: {', '.join(sorted(IGNORE_DIRS))})")

    # --- Copia in temp e lancia pipreqs ---
    with tempfile.TemporaryDirectory(prefix="pipreqs_tmp_") as tmp_str:
        tmp = Path(tmp_str)
        mirror_to_temp(py_files, root, tmp)
        print(f"File copiati in: {tmp}")

        rc = run_pipreqs(tmp, out_path, extra)

    if rc == 0:
        print(f"\n✅  requirements.txt scritto in: {out_path}")
    else:
        print(f"\n❌  pipreqs è uscito con codice {rc}.", file=sys.stderr)
        sys.exit(rc)


if __name__ == "__main__":
    main()