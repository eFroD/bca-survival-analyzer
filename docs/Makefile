# Minimal makefile for Sphinx documentation

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Package to document
PACKAGE       = bca_survival

.PHONY: help clean apidoc html livehtml Makefile

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	rm -rf $(BUILDDIR)
	rm -rf $(SOURCEDIR)/api

apidoc:
	@echo "Generating API documentation..."
	@rm -rf $(SOURCEDIR)/api
	@sphinx-apidoc -f -o $(SOURCEDIR)/api ../$(PACKAGE) --separate

html: clean apidoc
	@echo "Building HTML documentation..."
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)
	@echo ""
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html"

livehtml: apidoc
	@sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: clean apidoc
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)