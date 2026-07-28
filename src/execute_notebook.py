from __future__ import annotations

import argparse
import contextlib
import io
import json
import traceback
from pathlib import Path


def execute(path: Path) -> dict[str, int]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace: dict[str, object] = {"__name__": "__notebook__"}
    execution_count = 0
    errors = 0
    for position, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        execution_count += 1
        cell["execution_count"] = execution_count
        cell["outputs"] = []
        stdout = io.StringIO()
        stderr = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exec(
                    compile(str(cell.get("source", "")), f"{path.name}:cell-{position}", "exec"),
                    namespace,
                    namespace,
                )
        except Exception as error:
            errors += 1
            cell["outputs"].append(
                {
                    "output_type": "error",
                    "ename": type(error).__name__,
                    "evalue": str(error),
                    "traceback": traceback.format_exc().splitlines(),
                }
            )
        captured_stdout = stdout.getvalue()
        captured_stderr = stderr.getvalue()
        if captured_stdout:
            cell["outputs"].append(
                {"output_type": "stream", "name": "stdout", "text": captured_stdout}
            )
        if captured_stderr:
            cell["outputs"].append(
                {"output_type": "stream", "name": "stderr", "text": captured_stderr}
            )
        if errors:
            break
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    temporary.replace(path)
    payload = {
        "code_cells_executed": execution_count,
        "errors": errors,
        "total_cells": len(notebook["cells"]),
    }
    if errors:
        raise RuntimeError(json.dumps(payload))
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("notebook", type=Path)
    return result


if __name__ == "__main__":
    print(json.dumps(execute(parser().parse_args().notebook)))
