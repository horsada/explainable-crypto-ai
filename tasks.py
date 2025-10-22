import argparse
from pathlib import Path
import shutil

def clean():
    print("🧹 Starting clean...")

    # 1) Remove __pycache__ folders
    print("→ Removing __pycache__ directories...")
    count = 0
    for p in Path(".").rglob("__pycache__"):
        shutil.rmtree(p, ignore_errors=True)
        print(f"  - removed {p}")
        count += 1
    print(f"  ✅ Removed {count} __pycache__ directories.")

    # 2) Remove .pyc / .pyo files
    print("→ Removing .pyc / .pyo files...")
    count = 0
    for ext in (".pyc", ".pyo"):
        for f in Path(".").rglob(f"*{ext}"):
            f.unlink(missing_ok=True)
            print(f"  - removed {f}")
            count += 1
    print(f"  ✅ Removed {count} compiled Python files.")

    # 3) Remove build and dist folders
    print("→ Removing build/ and dist/ folders...")
    for p in ["build", "dist"]:
        path = Path(p)
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
            print(f"  - removed {path}")
        else:
            print(f"  - {path} not found, skipping.")
    print("  ✅ build/dist cleaned.")

    # 4) Remove egg-info directories
    print("→ Removing *.egg-info directories...")
    count = 0
    for p in Path(".").rglob("*.egg-info"):
        shutil.rmtree(p, ignore_errors=True)
        print(f"  - removed {p}")
        count += 1
    print(f"  ✅ Removed {count} egg-info directories.")

    print("✨ Clean completed successfully.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["clean"])
    args = ap.parse_args()
    globals()[args.cmd]()
