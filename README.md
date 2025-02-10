# Particle-Vtools

Tool set used to visulise & explore paticles in porous media.

<img width="973" alt="image" src="https://github.com/user-attachments/assets/71f2aa30-8388-4164-8d9d-30d209fc0e29" />


## Enviroment installation:

```shell
conda env create -f environment.yml
```

if you wish to update environment dependencies (not recommended) please do:

```
conda env update --name particle-vtools --file environment.yml --prune
```

if you wish to register this kernel to jupyterlab/
```shell
python -m ipykernel install --user --name=particle-vtools --display-name "particle-vtools"
```


## Package setup

```
pip install -e <path-to-folder-root>
```