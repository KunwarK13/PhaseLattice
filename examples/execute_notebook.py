"""Execute the walkthrough with the current interpreter and portable metadata."""

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import nbformat
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient


def main():
    root = Path(__file__).resolve().parents[1]
    path = root / "examples/reconstruction.ipynb"
    notebook = nbformat.read(path, as_version=4)
    with TemporaryDirectory(prefix="phaselattice-kernel-") as directory:
        spec = Path(directory) / "phaselattice"
        spec.mkdir()
        (spec / "kernel.json").write_text(
            json.dumps(
                {
                    "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                    "display_name": "Python 3",
                    "language": "python",
                }
            )
        )
        manager = KernelManager(
            kernel_name="phaselattice",
            kernel_spec_manager=KernelSpecManager(kernel_dirs=[directory]),
            transport="ipc",
        )
        client = NotebookClient(
            notebook, km=manager, timeout=180, resources={"metadata": {"path": str(root)}}
        )
        client.execute(cleanup_kc=True)
    for cell in notebook.cells:
        cell.metadata.pop("execution", None)
    notebook.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nbformat.write(notebook, path)
    count = sum(cell.cell_type == "code" for cell in notebook.cells)
    print(f"Executed {count} cells without errors: examples/reconstruction.ipynb")


if __name__ == "__main__":
    main()
