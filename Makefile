
.PHONY: clean build install editable-install


clean:
	rm -rf build
	rm -rf dist
	rm -rf qmodel.egg-info
	rm -f qmodel/data/cpp/*cpython*.so

build: clean
	python -m build --wheel

install: clean
	pip install .

editable-install: clean
	pip install -e .