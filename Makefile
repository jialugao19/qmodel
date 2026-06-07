
.PHONY: clean build install editable-install release-wheel


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

release-wheel: clean
	python -m build --wheel
	python -m twine upload dist/*.whl
