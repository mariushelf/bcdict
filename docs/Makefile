# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?= -W -n
#SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

clean:
	rm -rf "$(BUILDDIR)"
	rm -rf "$(SOURCEDIR)/_apidoc"
	rm -rf "jupyter_execute"


livehtml:
	poetry run sphinx-autobuild -a -b html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(0) --open-browser --port 0 --watch ../bcdict --watch ../README.md

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -a -b $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) --keep-going $(O)
