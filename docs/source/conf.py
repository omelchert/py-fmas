# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

sys.path.append('/usr/local/Cellar/python@3.9/3.9.2_1/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/')
from sphinx_gallery.sorting import ExplicitOrder, FileNameSortKey
import sphinx_rtd_theme

sys.path.append('/Users/melchert/Programs/Python/git-optfrog/optfrog/')

# -- Project information -----------------------------------------------------

project = 'py-fmas'
copyright = '2021, O. Melchert'
author = 'O. Melchert'

# The full version, including alpha/beta/rc tags
release = '0.0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
'sphinx.ext.napoleon',
'sphinx.ext.autosummary',
'sphinx.ext.autodoc',
'sphinx_gallery.gen_gallery'
]

autosummary_generate = True

# OM - Sphinx-Gallery configuration dictionary
sphinx_gallery_conf = {
    #'thumbnail_size': (600,420),
    'examples_dirs': ['../../galleries','../../tutorials/'],
    'subsection_order': ExplicitOrder([
        '../../galleries/gallery_01',
        '../../galleries/gallery_02',
        '../../galleries/gallery_03',
        '../../tutorials/basics',
        '../../tutorials/specific',
        ]),
     'gallery_dirs': ["auto_examples","auto_tutorials"],
    'filename_pattern':'/g_'
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

#html_logo = 'logo/fmas_logo_v5b.svg'

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
#html_theme = 'classic'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
