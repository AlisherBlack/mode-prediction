# Mode prediction

## Installing packages

Use Conda or Python virtual environment to install python with 3.11 version.

### Conda virtual environment
```bash
conda create --name <venv_name> 'python==3.11.3'
```

```bash
conda activate <venv_name>
```

### Python virtual environment

```bash
python3 -m pip install --user virtualenv
```

```bash
python3.11 -m venv <venv_name>
```
> If python3.11 is not recognized, but python3 points to Python 3.7:
> ```bash
> python3 -m venv <venv_name>
> ```

```bash
source ./<venv_name>/bin/activate  # for Unix/Linux/MacOS
```

or

```bash
.\<venv_name>\Scripts\activate   # for Windows
```



### Install packages

```bash
pip install -r requirements.txt
```


## Install dataset

The following script will create an `./input` directory in the root of the repository, where it will download the necessary data for training and visualisation. 

```bash
bash download_all_files.sh 
```

## Train model

The following script will train the model, create a `./experiment` directory in the root of the repository where the model weights will be saved.  The `loss.pdf` file with the training plot will be saved in the same directory.

```bash
python train_model.py
```

You can skip train step and download the test model to the `./experiments` directory using the following command:

```bash
mkdir ./experiments
curl -L $(yadisk-direct https://disk.yandex.ru/d/S5HvchhPkvUd6g) -o ./experiments/model.pth
```

## Visualize prediction

Open `visualize.ipynb` and run all cells. 


# Note!

For correct working, directory names must not be changed!




