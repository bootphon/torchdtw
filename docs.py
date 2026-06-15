# /// script
# requires-python = ">=3.14"
# dependencies = [
#   "griffe>=2.0.2",
#   "griffe2md>=1.5.0",
# ]
# ///
"""Simple script to generate markdown documentation from docstrings with griffe."""

import re
from pathlib import Path

import griffe
import griffe2md


def generate_docs(search_path: str) -> str:
    """Generate docs using griffe and griffe2md.

    Returns:
        The generated documentation.

    """
    package = griffe.load(
        next(Path(search_path).iterdir()).name,
        search_paths=[search_path],
        docstring_parser="sphinx",
    )
    package.docstring = None
    config = {
        **griffe2md.default_config,
        "summary": False,
        "show_root_full_path": False,
        "show_root_members_full_path": False,
        "show_object_full_path": False,
    }
    md = griffe2md.render_object_docs(package, config)
    md = re.sub(r"^## .+", "## Usage\n\nThis package provides three functions:", md, count=1)
    return re.sub(r"\[([^\]]+)\]\(#[^)]+\)", r"\1", md)


class MarkersError(RuntimeError):
    """To raise if the griffe markers are not found."""

    def __init__(self) -> None:
        """Create the exception."""
        super().__init__("Markers not found in README. Add:\n<!-- griffe -->\n<!-- /griffe -->")


def inject_into_readme(search_path: str, readme: str) -> None:
    """Replace content between griffe markers in the README.

    Raises:
        MarkersError: If there is no <!-- griffe --> markers in README.

    """
    docs = generate_docs(search_path)
    content = Path(readme).read_text(encoding="utf-8")
    pattern = r"(<!-- griffe -->)(.*?)(<!-- /griffe -->)"
    replacement = rf"\1\n{docs}\n\3"
    new_content, count = re.subn(pattern, replacement, content, flags=re.DOTALL)
    if count == 0:
        raise MarkersError
    Path(readme).write_text(new_content, encoding="utf-8")


if __name__ == "__main__":
    inject_into_readme("./src", "./README.md")
