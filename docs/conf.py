# Configuration file for Sphinx documentation builder

project = 'Laminocam'
copyright = '2024, Lawrence Berkeley National Laboratory'
author = 'Berkeley Lab'
version = '1.0'
release = '1.0'
language = 'en'

extensions = [
    'myst_parser',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
}

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

html_theme_options = {
    'logo_only': False,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': '#2980B9',
}
