# Provenance: HKUDS/OpenHarness (https://github.com/HKUDS/OpenHarness)
# Original: src/openharness/tools/notebook_edit_tool.py
# License: MIT
# Modified: Rewritten as Prometheus BaseTool

"""Edit Jupyter notebook cells (.ipynb files)."""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field

from prometheus.permissions.path_schema import PATH_FIELD_WRITE
from prometheus.security import safe_fs
from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolResult


class NotebookEditInput(BaseModel):
    """Arguments for editing a notebook cell."""

    path: str = Field(
        json_schema_extra=PATH_FIELD_WRITE,description="Path to the .ipynb file")
    cell_index: int = Field(ge=0, description="Zero-based cell index to edit")
    new_source: str = Field(description="New source content for the cell")
    cell_type: Literal["code", "markdown"] = Field(
        default="code",
        description=(
            "Whether the cell holds executable code or markdown prose. "
            "Defaults to code; only used when creating a cell."
        ),
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
    find_me = ("edit a Jupyter notebook cell", "append a cell to an ipynb")

    async def execute(
        self, arguments: NotebookEditInput, context: ToolExecutionContext
    ) -> ToolResult:
        # The path the gate checked, opened through no link (security/safe_fs.py, GHSA-wgcv-xw9m-9q9q).
        try:
            path = safe_fs.resolve_tool_path(arguments.path, context.cwd, context.metadata)
            present = safe_fs.exists(path)
            if present:
                nb = json.loads(safe_fs.read_text(path))
        except safe_fs.PathChanged as exc:
            return ToolResult(output=str(exc), is_error=True)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            return ToolResult(output=f"Cannot parse notebook: {exc}", is_error=True)
        if not present and arguments.create_if_missing:
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
        elif not present:
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

        try:
            safe_fs.write_text(path, json.dumps(nb, indent=2, ensure_ascii=False))
        except safe_fs.PathChanged as exc:
            return ToolResult(output=str(exc), is_error=True)

        return ToolResult(
            output=f"{action} cell {arguments.cell_index} in {path} "
            f"({arguments.cell_type}, {len(arguments.new_source)} chars)"
        )
