.PHONY: paper figure-rgb-policy-by-metric figure-rgb-policy-baselines-compact figure-rgb-policy-baselines clean

PERSISTENT_TECTONIC := $(abspath $(CURDIR)/../../../tools/latex-runtime/bin/tectonic)
TECTONIC := $(if $(wildcard $(PERSISTENT_TECTONIC)),$(PERSISTENT_TECTONIC),tectonic)
PERSISTENT_PDFCROP := $(abspath $(CURDIR)/../../../tools/latex-runtime/bin/pdfcrop)
PDFCROP := $(if $(wildcard $(PERSISTENT_PDFCROP)),$(PERSISTENT_PDFCROP),pdfcrop)

paper:
	$(TECTONIC) -C --keep-logs main.tex

figure-rgb-policy-by-metric:
	cd figures && $(TECTONIC) -C --keep-logs rgb-policy-by-metric.tex
	cd figures && PATH=$(dir $(PERSISTENT_PDFCROP)):$$PATH $(PDFCROP) --ini --margins '0 0 0 0' \
		rgb-policy-by-metric.pdf rgb-policy-by-metric-crop.pdf
	mv figures/rgb-policy-by-metric-crop.pdf figures/rgb-policy-by-metric.pdf

figure-rgb-policy-baselines-compact:
	cd figures && $(TECTONIC) -C --keep-logs rgb-policy-baselines-compact.tex
	cd figures && PATH=$(dir $(PERSISTENT_PDFCROP)):$$PATH $(PDFCROP) --ini --margins '0 0 0 0' \
		rgb-policy-baselines-compact.pdf rgb-policy-baselines-compact-crop.pdf
	mv figures/rgb-policy-baselines-compact-crop.pdf figures/rgb-policy-baselines-compact.pdf

figure-rgb-policy-baselines:
	cd figures && $(TECTONIC) -C --keep-logs rgb-policy-baselines.tex
	cd figures && PATH=$(dir $(PERSISTENT_PDFCROP)):$$PATH $(PDFCROP) --ini --margins '0 0 0 0' \
		rgb-policy-baselines.pdf rgb-policy-baselines-crop.pdf
	mv figures/rgb-policy-baselines-crop.pdf figures/rgb-policy-baselines.pdf

clean:
	$(RM) main.aux main.log main.out main.synctex.gz main.pdf
