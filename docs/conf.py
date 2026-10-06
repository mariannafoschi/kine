# Configuration file for the Sphinx documentation builder.

import os
import re
import sys

# docs/conf.py -> repo root, i.e. the directory containing the `kine` package.
# os.path.abspath() is relative to sphinx-build's cwd, not to this file, so
# anchor on __file__ -- otherwise a local build only works thanks to the
# `pip install -e .` performed by .github/workflows/docs.yml.
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
)

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
    'sphinx.ext.napoleon',
    # napoleon renders a Google-style `Todo:` section as a `.. todo::`
    # directive; without this extension those are unknown-directive errors.
    'sphinx.ext.todo',
    "sphinx_copybutton",
    'sphinx_design',
    'nbsphinx',
]

# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

exclude_patterns = ['build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

# Keep the `Todo:` notes out of the published HTML; the directive still has to
# be known for the build to succeed.
todo_include_todos = False

# -- Intersphinx

intersphinx_mapping = {
    'python':     ('https://docs.python.org/3', None),
    'numpy':      ('https://numpy.org/doc/stable/', None),
    'scipy':      ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'jax':        ('https://docs.jax.dev/en/latest/', None),
    'flax':       ('https://flax.readthedocs.io/en/latest/', None),
    'optax':      ('https://optax.readthedocs.io/en/latest/', None),
    'astropy':    ('https://docs.astropy.org/en/stable/', None),
    # eht-imaging publishes an inventory, so `Bases: ehtim.obsdata.Obsdata`
    # on kine.obsdata.Obsdata becomes a live link.
    'ehtim':      ('https://achael.github.io/eht-imaging/', None),
    'sphinx':     ('https://www.sphinx-doc.org/en/master/', None),
}
# Require an explicit prefix for cross-project :doc:/:ref: targets, so that a
# local label can never be silently shadowed by an upstream one.
intersphinx_disabled_reftypes = ['std:doc', 'std:label']

# -- Napoleon (kine docstrings are Google style throughout)

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False  # `special-members` below decides
napoleon_use_param = True
napoleon_use_keyword = True
napoleon_use_rtype = True
napoleon_use_ivar = False
napoleon_attr_annotations = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_references = False
napoleon_preprocess_types = False

# -- Autosummary

# Every object is described exactly once, inline on its module page, by the
# autoclass/autofunction directives at the bottom of docs/api/*.rst. The
# autosummary tables carry no `:toctree:`, so they are pure on-page indexes
# and generate no stub pages (hence no duplicate object descriptions).
autosummary_generate = False
autosummary_imported_members = False

# -- Autodoc

autoclass_content = 'class'        # Args:/Attributes: live in the class docstring
autodoc_member_order = 'bysource'
autodoc_typehints = 'signature'    # required by _shorten_array_types() below
autodoc_preserve_defaults = True   # render `activ=nn.gelu`, not `<function gelu ...>`
# Without this, the dataclass-generated __init__ of every flax Module and of
# Trainer inherits object.__init__'s docstring, counts as documented, and is
# rendered -- flax-injected `parent`/`name` arguments and all -- next to the
# class signature that already shows the same fields. With it, only the
# hand-written constructors (Image, Video, HyperParams, Schedule) are kept.
autodoc_inherit_docstrings = False

# Options shared by every autoclass/autofunction directive, so that the API
# pages cannot drift apart again. Note that `undoc-members` is deliberately
# absent: autodoc keeps a special member only when it has a docstring, so
# `__init__` is rendered for Image/Video (documented) and skipped for the
# dataclass-generated constructors of the flax modules and of Trainer.
# `__call__` is left out on purpose: the forward pass of the flax modules is
# an implementation detail, not part of the documented API.
autodoc_default_options = {
    'members': True,
    'show-inheritance': True,
    'special-members': '__init__',
    # `activ` and `outactiv` are flax dataclass fields whose default value is
    # a jax activation function. autodoc would otherwise document each of
    # them as a method of the neural field, docstring and all, pasting jax's
    # reference page for gelu/softplus into kine.model. They are already
    # described in the `Args:` block of the class docstring.
    'exclude-members': 'activ,outactiv',
}

# -- Nitpick
# The build runs with -n, so every cross-reference has to resolve. These are
# the few that legitimately cannot: jax.typing.ArrayLike and optax.OptState
# are `py:data` upstream, while a signature annotation is emitted as a
# `py:class` reference, which the Python domain will not match.
nitpick_ignore = [
    # jax.typing.ArrayLike and optax.OptState are `py:data` in the upstream
    # inventories, but a signature annotation is emitted as a `py:class`
    # reference, which the Python domain will not match.
    ('py:class', 'jax.typing.ArrayLike'),
    ('py:class', 'optax.OptState'),
    ('py:class', 'optax.GradientTransformation'),
    ('py:class', 'typing_extensions.Self'),
    ('py:class', 'numpy.typing.NDArray'),
    # Private aliases leaking out of flax's and optax's own annotations.
    ('py:class', 'flax.linen.module.Module'),
    ('py:class', 'flax.core.scope.Scope'),
    ('py:class', 'ArrayTree'),
]
nitpick_ignore_regex = [
    (r'py:class', r'flax\..*\._.*'),
    (r'py:class', r'numpy\._typing\..*'),
]

# -- Options for HTML output

html_theme = 'breeze' # 'sphinx_rtd_theme', 'breeze'
html_title = 'kine'
# html_logo = 'path/to/myimage.png'
html_theme_options = {
    'header_tabs': True,
}
# The breeze theme takes the repository from html_context, not from
# html_theme_options: it drives the repo-stats button in the sidebar and the
# "Edit this page" link. (The per-object [source] links come from
# sphinx.ext.linkcode / linkcode_resolve() below, independently of this.)
html_context = {
    'github_user': 'mariannafoschi',
    'github_repo': 'kine',
    'github_version': 'main',
    'doc_path': 'docs',
}

# -- Options for EPUB output
epub_show_urls = 'footnote'

# -- Data type specs fix
# autodoc renders jax's ArrayLike union in full. Rewrite it to the public
# names, keeping the leading `~` so the Python domain still renders them
# short. The pattern stays loose because the private module holding jax's
# Array moves between releases (jaxlib.xla_extension -> jax.jaxlib._jax).
# The negative lookahead keeps this from re-matching the `~jax.typing.
# ArrayLike` that the union substitution below has just produced.
_ARRAY = r'~(?:jax|jaxlib)[\w.]*\.Array(?!\w)'
_ARRAYLIKE_RE = re.compile(
    _ARRAY + r' \| ~numpy\.ndarray \| ~numpy\.bool_? \| ~numpy\.number'
    r' \| bool \| int \| float \| complex'
)
_ARRAY_RE = re.compile(_ARRAY)


# Callable defaults (nn.gelu, nn.softplus, ...) repr as `<function gelu>` or,
# once jitted, `<PjitFunction of <function sigmoid>>`. Render the name the
# source actually uses instead.
_FUNC_DEFAULT_RE = re.compile(
    r'<(?:PjitFunction of )?<?function (\w+)(?: at 0x[0-9a-f]+)?>?>'
)


# flax appends `parent` and `name` to the dataclass constructor of every
# Module. Nobody passes them, and the `<flax.linen.module._Sentinel object>`
# default is not valid Python: Sphinx then cannot parse the arglist, and
# signatures holding brackets (e.g. `Callable[[...], Any]`) fall back to a
# single raw line instead of the one-parameter-per-line layout.
_FLAX_INJECTED_RE = re.compile(
    r'(?:,\s*)?parent: [^=]*= <flax\.linen\.module\._Sentinel object>'
    r',\s*name: str \| None = None(?=\)$)'
)


def _shorten_array_types(app, what, name, obj, options, signature,
                         return_annotation):
    def fix(spec):
        if not spec:
            return spec
        spec = _ARRAYLIKE_RE.sub('~jax.typing.ArrayLike', spec)
        spec = _ARRAY_RE.sub('~jax.Array', spec)
        spec = _FLAX_INJECTED_RE.sub('', spec)
        return _FUNC_DEFAULT_RE.sub(r'nn.\1', spec)

    return fix(signature), fix(return_annotation)


# The breeze theme only splits a signature over 60 characters one parameter
# per line, so a short class (e.g. PhaseGains) would stay on a single line.
# Force the multi-line layout on every class signature instead.
def _multiline_class_signatures(app, doctree):
    from sphinx import addnodes

    for desc in doctree.findall(addnodes.desc):
        if desc.get('domain') != 'py' or desc.get('objtype') != 'class':
            continue
        for sig in desc.findall(addnodes.desc_signature):
            for params in sig.findall(addnodes.desc_parameterlist):
                params['multi_line_parameter_list'] = True


def setup(app):
    app.connect('autodoc-process-signature', _shorten_array_types)
    app.connect('doctree-read', _multiline_class_signatures)


# -- Source button
def linkcode_resolve(domain, info):
    if domain != 'py' or not info['module']:
        return None

    import importlib, inspect, os

    try:
        obj = importlib.import_module(info['module'])
        # Walk the full dotted path so a method links to the method, not to
        # its enclosing class, then unwrap @jax.jit / @nn.compact / etc.
        for part in info['fullname'].split('.'):
            obj = getattr(obj, part)
        obj = inspect.unwrap(obj)
        source_file = inspect.getfile(obj)
        source_lines, start_line = inspect.getsourcelines(obj)
    except (TypeError, OSError, ImportError, AttributeError):
        return None

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    rel_path = os.path.relpath(source_file, repo_root)
    if rel_path.startswith(os.pardir):
        # Inherited from a dependency: no source link.
        return None

    return (
        f"https://github.com/mariannafoschi/kine/blob/main/{rel_path}"
        f"#L{start_line}-L{start_line + len(source_lines) - 1}"
    )
