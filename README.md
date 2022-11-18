# Image Processing Helpers

## Installation

The easiest way to use this code is to install it using `pip`.

1. Clone the repo into any directory you'd like
2. Run the following commands

    `cd imageprocessing`

    `pip install git+https://github.com/dk0d/imageprocessing@main`

3. Add `import imageprocessing` to your python script.

_Note_

> The testing directory is excluded from install as it is only used
> to test classes and additions during development

## Documentation

> Documentation is done using [sphinx](https://www.sphinx-doc.org/en/master/index.html). There is a sphinx tutorial [here](docs/brandons-sphinx-tutorial.pdf).

Full documentation can be found [here](docs/build/html/index.html).

Docs can be updated by:

```
cd docs

make build
```

> Note: sphinx must be installed to update the documentation with any changes to code docstrings.

Install sphinx with:

`conda install sphinx`

`conda install -c astropy sphinx-automodapi`

or

`pip install sphinx`

`pip install sphinx-automodapi`