from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute a committed MonReader notebook.")
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--in-place", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.in_place and args.output is not None:
        raise SystemExit("Choose either --in-place or --output, not both.")

    notebook_path = args.notebook.resolve()
    repo_root = Path(__file__).resolve().parents[1]
    notebook = nbformat.read(notebook_path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(repo_root)}},
    )
    executed = client.execute()

    if args.in_place:
        output_path = notebook_path
    elif args.output is not None:
        output_path = args.output.resolve()
    else:
        output_path = notebook_path.with_name(f"{notebook_path.stem}.executed.ipynb")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(executed, output_path)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
