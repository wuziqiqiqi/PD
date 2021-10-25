.PHONY: publish_test publish clean veryclean package

COMMAND=python
ifdef PYCMD
	COMMAND=$(PYCMD)
endif

BUILD_EXT=$(COMMAND) setup.py build_ext

# Build the external .so file
build_ext:
	$(BUILD_EXT)

inplace:
	$(BUILD_EXT) --inplace

# Clean stuff related to building the external C++ package
clean:
	rm -rf build/
	rm -f cxx/cython/clease_cxx.cpp

# Prepare for publishing, clean all pycache etc.
# Pyclean removes the __pycache__ and *.pyc files.
veryclean:
	make clean
	rm -rf dist/
	rm -rf *.egg-info/
	pyclean -v .

package:
	$(COMMAND) setup.py sdist

check:
	twine check dist/*

publish_test:
	make veryclean
	make package
	make check
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

publish:
	make veryclean
	make package
	make check
	twine upload dist/*
