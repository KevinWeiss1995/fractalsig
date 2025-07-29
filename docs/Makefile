# Makefile for FractalSig LaTeX documentation

MAIN = fractalsig_theory_implementation
TEX = $(MAIN).tex
PDF = $(MAIN).pdf

.PHONY: all clean open help

# Default target
all: $(PDF)

# Compile PDF
$(PDF): $(TEX)
	@echo "🔧 Compiling LaTeX documentation..."
	@pdflatex -interaction=nonstopmode $(TEX) > /dev/null
	@pdflatex -interaction=nonstopmode $(TEX) > /dev/null
	@echo "✅ Documentation compiled: $(PDF)"

# Clean auxiliary files
clean:
	@echo "🧹 Cleaning auxiliary files..."
	@rm -f *.aux *.log *.out *.toc *.fls *.fdb_latexmk *.synctex.gz
	@echo "✅ Clean complete"

# Clean everything including PDF
distclean: clean
	@rm -f $(PDF)

# Open PDF (macOS/Linux)
open: $(PDF)
	@if command -v open >/dev/null 2>&1; then \
		open $(PDF); \
	elif command -v xdg-open >/dev/null 2>&1; then \
		xdg-open $(PDF); \
	else \
		echo "Cannot open PDF automatically. Please open $(PDF) manually."; \
	fi

# Help
help:
	@echo "FractalSig Documentation Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all        - Compile the PDF documentation (default)"
	@echo "  clean      - Remove auxiliary files"
	@echo "  distclean  - Remove all generated files including PDF"
	@echo "  open       - Open the compiled PDF"
	@echo "  help       - Show this help" 