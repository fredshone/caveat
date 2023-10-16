
# Installation

## Setting up a user environment

As a `caveat` user, it is easiest to install using the [mamba](https://mamba.readthedocs.io/en/latest/index.html) package manager, as follows:

1. Install mamba with the [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge) executable for your operating system.
2. Open the command line (or the "miniforge prompt" in Windows).

3. Create the caveat mamba environment: `mamba create -n caveat -c conda-forge -c city-modelling-lab caveat`
4. Activate the caveat mamba environment: `mamba activate caveat`


All together:

--8<-- "README.md:docs-install-user"
### Running the example notebooks
If you have followed the non-developer installation instructions above, you will need to install `jupyter` into your `caveat` environment to run the [example notebooks](https://github.com/fredshone/caveat/tree/main/examples):

``` shell
mamba install -n caveat jupyter
```

With Jupyter installed, it's easiest to then add the environment as a jupyter kernel:

``` shell
mamba activate caveat
ipython kernel install --user --name=caveat
jupyter notebook
```

### Choosing a different environment name
If you would like to use a different name to `caveat` for your mamba environment, the installation becomes (where `[my-env-name]` is your preferred name for the environment):

``` shell
mamba create -n [my-env-name] -c conda-forge --file requirements/base.txt
mamba activate [my-env-name]
ipython kernel install --user --name=[my-env-name]
```
## Setting up a development environment

The install instructions are slightly different to create a development environment compared to a user environment:

--8<-- "README.md:docs-install-dev"

For more detailed installation instructions specific to developing the caveat codebase, see our [development documentation][setting-up-a-development-environment].
