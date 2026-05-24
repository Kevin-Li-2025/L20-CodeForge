ARXIV_SOURCE = l20-codeforge-arxiv-source.tar.gz

.PHONY: build clean arxiv-source

build:
	tectonic main.tex

clean:
	rm -f main.pdf main.aux main.bbl main.blg main.log main.out main.toc
	rm -f $(ARXIV_SOURCE)

arxiv-source: clean
	tar --format=ustar -czf $(ARXIV_SOURCE) main.tex references.bib
