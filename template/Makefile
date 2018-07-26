all: auswertung  VXX.pdf

TeXOptions = -lualatex \
			 -interaction=nonstopmode \
			 -halt-on-error \
			 -output-directory=build

document: main.tex auswertung.py
	latexmk $(TeXOptions) main.tex
	mv build/main.pdf build/main.pdf

auswertung: auswertung.py data/*.txt
	python auswertung.py

build:
	mkdir -p build

clean:
	rm -rf build
