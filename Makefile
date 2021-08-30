.PHONY: publish_test publish clean package

COMMAND=python
ifdef PYCMD
	COMMAND=$(PYCMD)
endif

clean:
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/
	rm -rf __pycache__/
	rm -rf */__pycache__/
	rm -f cxx/cython/clease_cxx.cpp

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
