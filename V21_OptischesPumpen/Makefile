all: auswertung  V21.pdf

TeXOptions = -lualatex \
			 -interaction=nonstopmode \
			 -halt-on-error \
			 -output-directory=build

V21.pdf: main.tex auswertung.py
	latexmk $(TeXOptions) main.tex
	mv build/main.pdf build/V21.pdf

document: main.tex
	latexmk $(TeXOptions) main.tex

auswertung: auswertung.py data/*.txt
	python auswertung.py

build:
	mkdir -p build

clean:
	rm -rf build
