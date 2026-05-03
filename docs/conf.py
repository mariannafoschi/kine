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
    'sphinx.ext.linkcode',
#    'sphinx.ext.napoleon',
    'sphinx_design',
    'nbsphinx',
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
    'use_source_button': True,
}

# -- Options for EPUB output
epub_show_urls = 'footnote'

# -- Data type specs
autodoc_type_aliases = {
    'ArrayLike': 'ArrayLike',
    'jax.typing.ArrayLike': 'ArrayLike',
    'jaxlib.xla_extension.ArrayImpl': 'Array',
}

import re

_ARRAYLIKE_RE = re.compile(
    r'Array \| ndarray \| bool \| number \| bool \| int \| float \| complex'
)

def _shorten_array_types(app, what, name, obj, options, signature, return_annotation):
    if signature:
        signature = _ARRAYLIKE_RE.sub('ArrayLike', signature)
    if return_annotation:
        return_annotation = _ARRAYLIKE_RE.sub('ArrayLike', return_annotation)
    return signature, return_annotation

def setup(app):
    app.connect('autodoc-process-signature', _shorten_array_types)

# -- Source button
def linkcode_resolve(domain, info):
    if domain != 'py' or not info['module']:
        return None

    import importlib, inspect, os

    try:
        mod = importlib.import_module(info['module'])
        obj_name = info['fullname'].split('.')[0]
        obj = getattr(mod, obj_name, None)
        if obj is None:
            return None
        source_file = inspect.getfile(obj)
        source_lines, start_line = inspect.getsourcelines(obj)
    except (TypeError, OSError, ImportError):
        return None

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    rel_path = os.path.relpath(source_file, repo_root)

    return (
        f"https://github.com/mariannafoschi/kine/blob/main/{rel_path}"
        f"#L{start_line}-L{start_line + len(source_lines) - 1}"
    )
