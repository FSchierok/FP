all: auswertung  V70.pdf

TeXOptions = -lualatex \
			 -interaction=nonstopmode \
			 -halt-on-error \
			 -output-directory=build

V70.pdf: main.tex auswertung.py
	latexmk $(TeXOptions) main.tex
	mv build/main.pdf build/V70.pdf

auswertung: auswertung.py data/*.txt
	python auswertung.py

build:
	mkdir -p build

clean:
	rm -rf build
