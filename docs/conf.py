"""Standalone Operange documentation."""
from importlib.metadata import PackageNotFoundError, version as package_version

project = "Operange"
author = "Operange contributors"
copyright = "2026, updatesupport and Operange contributors"
try:
    release = package_version("operange")
except PackageNotFoundError:
    release = "0.1.0"
version = ".".join(release.split(".")[:2])
extensions = ["myst_parser"]
source_suffix = {".md": "markdown"}
root_doc = "index"
exclude_patterns = ["_build", ".DS_Store"]
html_theme = "furo"
html_title = "Operange"
myst_heading_anchors = 3
