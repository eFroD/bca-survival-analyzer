# Configuration file for the Sphinx documentation builder.
import os
import sys
from datetime import datetime

# Add the project root directory to the path so Sphinx can find your modules
sys.path.insert(0, os.path.abspath('../..'))

# Project information
project = 'BCA Survival Analysis'
copyright = f'{datetime.now().year}, Eric Frodl'
author = 'Eric Frodl'

# The full version, including alpha/beta/rc tags
try:
    from bca_survival._version import version
    release = version
except ImportError:
    release = 'development'

# Extensions
extensions = [
    'sphinx.ext.autodoc',      # Auto-generate API documentation
    'sphinx.ext.viewcode',     # Add links to view source code
    'sphinx.ext.napoleon',     # Support for NumPy and Google style docstrings
    'sphinx.ext.intersphinx',  # Link to other projects' documentation
    'sphinx_autodoc_typehints', # Support for type hints in docstrings
    'myst_parser',             # Markdown support
]

# Add any paths that contain templates
templates_path = ['_templates']

# List of patterns to exclude from source
exclude_patterns = []

# The theme to use for HTML and HTML Help pages
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files
html_static_path = ['_static']

# MyST Markdown parser settings
myst_enable_extensions = [
    'colon_fence',      # Allow ::: code blocks
    'deflist',          # Definition lists
    'smartquotes',      # Smart quotes
    'replacements',     # Text replacements
    'tasklist',         # Task lists with [ ] and [x]
]

# Auto-generate API documentation
autodoc_member_order = 'bysource'
autoclass_content = 'both'  # Include both class and __init__ docstrings

html_context = {
    "display_github": True,
    "github_user": "eFroD",
    "github_repo": "bca-survival-analyzer",
    "github_version": "main",
    "conf_py_path": "/docs/source/",
}