# Skoltech Thesis Template in LaTeX

This LaTeX template conforms to the Skoltech Thesis specifications,
however the specifications can change.  We recommend that you verify the
layout of your title page with your thesis advisor and the Education department 
before printing your final copy.

# Structure

* Main file `main.tex` - contains the document class, use of packages, and inludes a list of other files.
* Style file `sk-thesis.cls` - contains layout specification, and defines macros for the title page.
* Folder `chapters` for the tex files of the chapters.
* Folder `graphics` for the graphic files.
* Folder `tables` for the text files of tables.

# Use
The template presumes use with PDFLaTeX.

This template can be used on [Overleaf](https://www.overleaf.com/).

The document class allows to activate draft or standard mode.

* `Draft Mode` renders a simple title page, excludes the dedication and acknowledgements page.
It adds a footer with the date of the last build.
Moreover, a list of open items (TODO list) is included at the beginning.

* In `Normal mode` the official cover page is rendered.

If needed one can activate the to render a double page PDF, where chapters always start on an odd page.

