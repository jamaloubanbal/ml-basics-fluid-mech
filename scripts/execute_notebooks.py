"""Execute notebooks programmatically with nbclient and save executed copies.

Usage: run with the project venv python. The script will execute the notebooks listed
and write files named executed_<original>.ipynb in the same folder.
"""
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
from pathlib import Path

notebooks = [
    Path('examples/notebooks/regression.ipynb'),
    Path('examples/notebooks/classification.ipynb'),
    Path('examples/notebooks/clustering.ipynb'),
]

for nbpath in notebooks:
    print('Executing', nbpath)
    nb = nbformat.read(nbpath, as_version=4)
    client = NotebookClient(nb, timeout=120, kernel_name='python3')
    try:
        client.execute()
        outpath = nbpath.parent / ('executed_' + nbpath.name)
        nbformat.write(nb, outpath)
        print('Wrote', outpath)
    except CellExecutionError as e:
        print('Execution failed for', nbpath)
        print(e)
    except Exception as e:
        print('Unexpected error for', nbpath)
        print(e)
