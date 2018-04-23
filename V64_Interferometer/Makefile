all: auswertung  document

TeXOptions = -lualatex \
			 -interaction=nonstopmode \
			 -halt-on-error \
			 -output-directory=build

V64.pdf: main.tex auswertung.py
	latexmk $(TeXOptions) main.tex
	mv build/main.pdf uild V64.pdf

doc: main.tex
	latexmk $(TeXOptions) main.tex

auswertung: auswertung.py data/*.txt
	python auswertung.py

build:
	mkdir -p build

clean:
	rm -rf build
