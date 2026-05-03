# Configuration file for the Sphinx documentation builder.

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

project = 'kine'
copyright = '2026, Marianna Foschi, Antonio Fuentes, Brandon Zhao'
# author = 'Marianna Foschi, Antonio Fuentes, Brandon Zhao et al.'

release =  '0.1.0'
version =  '0.1.0'

language = 'en'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx_design',
    'nbsphinx'
]

# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

# autodoc_mock_imports = ['kine']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'breeze' # 'sphinx_rtd_theme', 'breeze'
html_title = 'kine'
# html_logo = 'path/to/myimage.png'
html_theme_options = {
    'repository_url': 'https://github.com/mariannafoschi/kine',
    'use_repository_button': True,
    'sidebar_position': 'right',
    'use_source_button': True,
}

# -- Options for EPUB output
epub_show_urls = 'footnote'
