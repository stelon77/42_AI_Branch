# pip install --upgrade build
python -m pip install --upgrade pip
pip install --upgrade setuptools
pip install --upgrade wheel
# python3 -m build
python3 setup.py sdist bdist_wheel

pip install ./dist/my_minipack-1.0.0.tar.gz
# pip install ./dist/my_minipack-1.0.0-py3-none-any.whl