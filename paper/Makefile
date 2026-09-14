.PHONY: paper figure-benchmarks figure-generated-data-benchmark figure-policy-architecture-benchmark figure-rgb-policy-baselines-compact figure-rgb-policy-baselines clean

PERSISTENT_TECTONIC := $(abspath $(CURDIR)/../../../tools/latex-runtime/bin/tectonic)
TECTONIC := $(if $(wildcard $(PERSISTENT_TECTONIC)),$(PERSISTENT_TECTONIC),tectonic)
PERSISTENT_PDFCROP := $(abspath $(CURDIR)/../../../tools/latex-runtime/bin/pdfcrop)
PDFCROP := $(if $(wildcard $(PERSISTENT_PDFCROP)),$(PERSISTENT_PDFCROP),pdfcrop)

PYTHON ?= python3

figure-benchmarks: figure-generated-data-benchmark figure-policy-architecture-benchmark

figure-generated-data-benchmark:
	TECTONIC=$(TECTONIC) PDFCROP=$(PDFCROP) $(PYTHON) scripts/make_stacked_benchmark_bars.py \
		--config scripts/benchmark_configs/generated_data.json \
		--output paper/figures/generated-data-benchmark.pdf

figure-policy-architecture-benchmark:
	TECTONIC=$(TECTONIC) PDFCROP=$(PDFCROP) $(PYTHON) scripts/make_stacked_benchmark_bars.py \
		--config scripts/benchmark_configs/policy_architecture.json \
		--output paper/figures/policy-architecture-benchmark.pdf

paper:
	$(TECTONIC) -C --keep-logs main.tex

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
