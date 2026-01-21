## Setup environment
The following provides intructions for setting up a python environment for a linux OS. Some of this may transfer to other OSs, but no guarantees.

Installations are performed with pip. If you prefer Anaconda (recommended for windows OS), then consult the appropriate documentation.

### Clone the current repo
```
git clone https://github.com/cabrittin/cebraindev_abm.git
```

### Setup virual environment
```
> mkdir -p ~/.venv
> python3 -m venv ~/.venv/cebraindev_abm
```
Now link the repo's main library to the virtual environment
```
> cd ~/.venv/cebraindev_abm/lib/python3.12/site-packages
> ln -s /path/to/repo/cebraindev_abm/abm/
```
Adjust the above paths accordingly.

### Activate virtual environment
To activate environment
```
> source ~/.venv/cebraindev_abm/bin/activate
```

For convenience, it is recommended to add the following line to your .bash_aliases file
```
> alias workon_cebraindevabm="source ~/.venv/cebraindev_abm/bin/activate"
```
Then to activate the environment, you can just run 
```
> workon_cebraindevabm
```

### Install requirements
In the repo directiory and with your local environment activated, standard pip install from the requirements.txt file is
```
pip3 install -r requirements.txt
```

However, library versions quickly become out of date and pip will routinely fail. Therefore, I recommend installing with 
```
cat requirements.txt | xargs -n 1 pip3 install
```
To see any libraries that failed to install
```
diff requirements.txt <(pip freeze)
```
Most of the output will likely reflect only changes in version numbers. If cebraindev_abm scripts fail to run, then consult this output and the python error to determine any packages that need to be installed. 


