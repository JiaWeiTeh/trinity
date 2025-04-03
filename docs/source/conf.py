# Configuration file for the Sphinx documentation builder.



# # -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys

# sys.path.insert(0, os.path.abspath("../.."))

# -- Project information

project = 'TRINITY'
copyright = '2022, Jia Wei Teh, et al.'
author = 'Jia Wei Teh'

release = '0.1'
version = '0.1.0'

    
# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx_rtd_theme',
    'sphinx_copybutton',
    'sphinx_toolbox.collapse',    
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}


intersphinx_disabled_domains = ['std']

# templates_path = ['_templates']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

# html_theme = 'alabaster'
html_theme = "sphinx_rtd_theme"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    "prev_next_buttons_location": "both",
    "collapse_navigation": False,
    "sticky_navigation": False,
    "includehidden": False,
    "titles_only": False,
    "display_version": False,
}

# Title formatting
html_title = "%s documentation" % (project)

# Date formatting
html_last_updated_fmt = "%a %d %b %Y"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ["_static"]

# OpenSearch. Requires documentation to be online.
html_use_opensearch = "https://trinitySF.readthedocs.io"

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
# html_sidebars = {}

html_split_index = True

html_baseurl = "https://trinitySF.readthedocs.io"



# -- Options for EPUB output
# epub_show_urls = 'footnote'











