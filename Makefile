.PHONY: publish_test publish clean package

COMMAND=python
ifdef PYCMD
	COMMAND=$(PYCMD)
endif

clean:
	rm -rf dist/
	rm -rf build/
	rm -rf __pycache__/
	rm -rf clease/__pycache__/
	rm -rf clease/calculator/__pycache___/
	rm -rf clease/montecarlo/__pycache__/
	rm -rf clease/montecarlo/observers/__pycache__/
	rm -rf clease/montecarlo/constraints/__pycache__/
	rm -rf clease/gui/__pycache__/
	rm cxx/cython/clease_cxx.cpp

package:
	$(COMMAND) setup.py sdist

check:
	twine check dist/*

publish_test:
	make clean
	make package
	make check
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

publish:
	make clean
	make package
	make check
	twine upload dist/*
