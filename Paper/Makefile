R = opt
STY = *.cls
FIG = figures/

pdf: $(R).pdf

# $(R).bbl: $(R).bib
# 	pdflatex $(R)
# 	bibtex $(R)
# 	pdflatex $(R)

$(R).pdf: *.tex $(STY) $(FIG) #$(R).bbl
	#pdflatex $(R)
	latexmk -pdf $(R)
    # mkdir -p archive
    # cp $@ archive/`date "+%y.%m.%d-%H.%M.%S"`-`openssl md5 < $@`.pdf

clean:
	rm -f $(R).log $(R).aux $(R).bbl $(R).blg $(R).out
