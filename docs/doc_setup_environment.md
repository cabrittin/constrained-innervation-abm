## Setup environment
The following provides intructions for setting up a python environment for a linux OS. Some of this may transfer to other OSs, but no guarantees.

Installations are performed with pip. If you prefer Anaconda (recommended for windows OS), then consult the appropriate documentation.

### Clone the current repo
```
git clone https://github.com/cabrittin/constrained-wiring-abm.git
```

### Setup virual environment
```
> mkdir -p ~/.venv
> python3 -m venv ~/.venv/constrained-wiring-abm
```
Now link the repo's main library to the virtual environment
```
> cd ~/.venv/cebraindev_abm/lib/python3.12/site-packages
> ln -s /path/to/repo/constrained-wiring-abm/abm/
```
Adjust the above paths accordingly.

### Activate virtual environment
To activate environment
```
> source ~/.venv/constrained-wiring-abm/bin/activate
```

For convenience, it is recommended to add the following line to your .bash\_aliases file
```
> alias workon_constrainedwiringabm="source ~/.venv/constrained-wiring-abm/bin/activate"
```
Then to activate the environment, you can just run 
```
> workon_constrainedwiringabm
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
Most of the output will likely reflect only changes in version numbers. If a constrained-wiring-abm scripts fails to run, then consult this output and the python error to determine any packages that need to be installed. 


