import nbformat as nbf
from pathlib import Path

scripts = [
    "localBiMoranI.py",
    "localBiT.py",
    "localLeeL.py",
    "localMultiGearyC.py",
]

for name in scripts:
    path = Path(name)
    code = path.read_text()
    nb = nbf.v4.new_notebook()
    # Add a title markdown cell
    nb.cells.append(nbf.v4.new_markdown_cell(f"# `{name}`"))
    # Add the code
    nb.cells.append(nbf.v4.new_code_cell(code))
    # Write out the notebook
    out = path.with_suffix(".ipynb")
    with open(out, "w") as f:
        nbf.write(nb, f)
    print(f"Created {out}")
