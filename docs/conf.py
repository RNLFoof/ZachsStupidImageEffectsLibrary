# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "zsil"
copyright = '2023, Zoe Zablotsky'
author = 'Zoe Zablotsky'

import sys, os
sys.path.insert(0, os.path.abspath('..'))
print(__file__)

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    "sphinx_autodoc_typehints"
]

intersphinx_mapping = {
  'py': ('https://docs.python.org/3', None),
  'pil': ('https://pillow.readthedocs.io/en/stable/', None),
}

autodoc_member_order = 'bysource'
autodoc_typehints = 'none'

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
