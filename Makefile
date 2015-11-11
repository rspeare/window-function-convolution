


profiletime: 
	python -m cProfile -s time BispEstimator.py
profile:
	python -m cProfile -o profile.pstats BispEstimator.py
profileviz:
	~/Code/gprof2dot/gprof2dot.py -f pstats profile.pstats | dot -Tsvg -o profile.svg
	open profile.svg -a /Applications/Google\ Chrome.app/
clean:
	$(RM) *~ fortran_lib.so fortran_lib2.so