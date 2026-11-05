import os
import re
import nbformat as nbf

src = open("train.py", encoding="utf-8").read()
src = re.sub(r'^""".*?"""\n', "", src, count=1, flags=re.S)
chunks = re.split(r"^# %%(.*)$", src, flags=re.M)
cells = []
for header, body in zip(chunks[1::2], chunks[2::2]):
    body = body.strip("\n")
    if header.strip().startswith("[markdown]"):
        text = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#")
                         for l in body.splitlines())
        cells.append(nbf.v4.new_markdown_cell(text))
    else:
        if header.strip():
            cells.append(nbf.v4.new_markdown_cell(f"## {header.strip()}"))
        cells.append(nbf.v4.new_code_cell(body))
nb = nbf.v4.new_notebook()
nb["cells"] = cells
os.makedirs("notebook", exist_ok=True)
nbf.write(nb, "notebook/analisis.ipynb")
print("notebook/analisis.ipynb dibuat")