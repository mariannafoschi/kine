# Configuration file for the Sphinx documentation builder.

project = 'kine'
copyright = '2026, Antonio Fuentes'
# author = 'Antonio Fuentes et al.'

release =  "0.0.1"
version =  "0.0.1"

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

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

# autodoc_mock_imports = ["kine"]

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_book_theme'
html_title = "kine"
# html_logo = "path/to/myimage.png"
html_theme_options = {
    # "github_url": "https://github.com/aefezeta/kine",
    "repository_url": "https://github.com/aefezeta/kine",
    "use_repository_button": True,
#    "use_source_button": True,
}

# -- Options for EPUB output
epub_show_urls = 'footnote'