import subprocess, sys

def sh(args:list[str]):
    print("→"," ".join(args))
    r = subprocess.run(args, text=True)
    if r.returncode: sys.exit(r.returncode)