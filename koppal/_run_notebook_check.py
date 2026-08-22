# Runs every code cell of the notebook in order (headless) to verify it executes cleanly
# and produces the saved model. Skips Jupyter %magics and suppresses plot windows.
import json
import matplotlib
matplotlib.use("Agg")            # no GUI windows
import matplotlib.pyplot as plt
plt.show = lambda *a, **k: None  # don't block on show()

nb = json.load(open("koppal_intent_classifier.ipynb", encoding="utf-8"))
ns = {}
for n, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    src = "\n".join(l for l in src.splitlines() if not l.strip().startswith("%"))
    try:
        exec(compile(src, f"<cell {n}>", "exec"), ns)
    except Exception as e:
        print(f"\n!!! ERROR in cell {n}:\n{src}\n--> {type(e).__name__}: {e}")
        raise
print("\nALL CELLS RAN OK")
