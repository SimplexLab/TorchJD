# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?= -W --keep-going -n
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile redirects redirects-check

redirects:
	@if [ -d "$(BUILDDIR)/dirhtml" ]; then \
		python _redirects.py "$(BUILDDIR)/dirhtml"; \
	elif [ -d "$(BUILDDIR)/html" ]; then \
		python _redirects.py "$(BUILDDIR)/html"; \
	else \
		echo "No build directory found; run make html or make dirhtml first."; \
	fi

redirects-check:
	@if [ -d "$(BUILDDIR)/dirhtml" ]; then \
		python _redirects.py --check "$(BUILDDIR)/dirhtml"; \
	elif [ -d "$(BUILDDIR)/html" ]; then \
		python _redirects.py --check "$(BUILDDIR)/html"; \
	else \
		echo "No build directory found; run make html or make dirhtml first."; \
		exit 1; \
	fi

html:
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@python _redirects.py "$(BUILDDIR)/html"

dirhtml:
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@python _redirects.py "$(BUILDDIR)/dirhtml"

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
