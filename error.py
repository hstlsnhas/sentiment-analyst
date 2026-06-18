import nbformat

with open("Analisis_Sentimen.ipynb", "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

# hapus widget metadata
if "widgets" in nb["metadata"]:
    del nb["metadata"]["widgets"]

with open("fixed.ipynb", "w", encoding="utf-8") as f:
    nbformat.write(nb, f)