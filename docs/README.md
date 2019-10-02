To rebuild the documentation, install the following Python packages:
```
pip install sphinx sphinx_rtd_theme nbsphinx fastcache ipykernel
```

Install also `pandoc`:
```
sudo apt-get install pandoc
```

and then run:

```
sphinx-build -M html . build
```
