# Provenance: HKUDS/OpenHarness (https://github.com/HKUDS/OpenHarness)
# Original: src/openharness/tools/notebook_edit_tool.py
# License: Apache-2.0
# Modified: Rewritten as Prometheus BaseTool

"""Edit Jupyter notebook cells (.ipynb files)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolResult


class NotebookEditInput(BaseModel):
    """Arguments for editing a notebook cell."""

    path: str = Field(description="Path to the .ipynb file")
    cell_index: int = Field(ge=0, description="Zero-based cell index to edit")
    new_source: str = Field(description="New source content for the cell")
    cell_type: Literal["code", "markdown"] = Field(
        default="code", description="Cell type"
    )
    mode: Literal["replace", "append"] = Field(
        default="replace",
        description="'replace' overwrites the cell; 'append' adds after it",
    )
    create_if_missing: bool = Field(
        default=True,
        description="Create the notebook if it does not exist",
    )


class NotebookEditTool(BaseTool):
    """Edit or append cells in a Jupyter .ipynb notebook."""

    name = "notebook_edit"
    description = (
        "Edit a cell in a Jupyter notebook (.ipynb). Can replace or append cells, "
        "and creates the notebook if it doesn't exist."
    )
    input_model = NotebookEditInput

    async def execute(
        self, arguments: NotebookEditInput, context: ToolExecutionContext
    ) -> ToolResult:
        path = Path(arguments.path)
        if not path.is_absolute():
            path = context.cwd / path

        if path.exists():
            try:
                nb = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                return ToolResult(output=f"Cannot parse notebook: {exc}", is_error=True)
        elif arguments.create_if_missing:
            nb = {
                "nbformat": 4,
                "nbformat_minor": 5,
                "metadata": {
                    "kernelspec": {
                        "display_name": "Python 3",
                        "language": "python",
                        "name": "python3",
                    },
                    "language_info": {"name": "python", "version": "3.11.0"},
                },
                "cells": [],
            }
        else:
            return ToolResult(output=f"Notebook not found: {path}", is_error=True)

        cells = nb.setdefault("cells", [])
        new_cell = {
            "cell_type": arguments.cell_type,
            "metadata": {},
            "source": arguments.new_source.splitlines(keepends=True),
            "outputs": [] if arguments.cell_type == "code" else [],
        }
        if arguments.cell_type == "code":
            new_cell["execution_count"] = None

        # Extend cell list if needed
        while len(cells) <= arguments.cell_index:
            cells.append(
                {
                    "cell_type": "code",
                    "metadata": {},
                    "source": [],
                    "outputs": [],
                    "execution_count": None,
                }
            )

        if arguments.mode == "replace":
            cells[arguments.cell_index] = new_cell
            action = "Replaced"
        else:  # append
            cells.insert(arguments.cell_index + 1, new_cell)
            action = "Appended after"

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(nb, indent=2, ensure_ascii=False), encoding="utf-8")

        return ToolResult(
            output=f"{action} cell {arguments.cell_index} in {path} "
            f"({arguments.cell_type}, {len(arguments.new_source)} chars)"
        )
