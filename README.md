# Kaggle credit card fraud detector
<br>
Credit card fraud has become more common nowadays. Building machine learning models that learn from historical data and accurately detect new fraud instances is a prime example of how ML can be applied in the real world and solve business problems.
<br><br>
Read more about the dataset and Download it <a href="https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data" target="blank">here</a>. Before running the project source code, download the dataset, rename it to 'creditcard.csv' and place it in the project directory. The dataset was not included in this repository due to its size.
<br><br>

### Goal of this project:
Build a classifier that learns from heavily imbalanced labeled data to successfully and accurately detect fraudulent credit card transactions from a large pool transactions
<br>
Success metric : Achieve 90%+ F1 score
<br><br>

### Evaluation metrics:
Recall : true positives / (true positives + false negatives)
<br>
Precision : true positives / (true positives + false positives)
<br>
F1 : 2 * (precision*recall) / (precision+recall)
<b><br><br>

## Running project files
1- main.py :
<br>
Perform EDA on the dataset
<br>
```
python main.py
```
<br>
2- fraudDetector.py :
<br>
Class containing model structure
<br><br>
3- model.py :
<br>
prepare training and testing data, initialize and train model, evaluate and save final model parameters
<br>
```
python model.py
```
<b><br><br>


## Installations
In order to clone the project files and run them on your machine, install the following libraries
<br><br>
**1- python 3.10**
<br><br>
**2- Pandas**
<br>
  To install
<br>
```
# conda
conda install -c conda-forge pandas
# or PyPI
pip install pandas
```
<br>

**3- Numpy**
<br>
  To install
<br>
```
# conda
conda install -c anaconda numpy
# or PyPI
pip install numpy
```
<br>

**4- scikit-learn**
<br>
  To install
<br>
```
# conda
conda create -n sklearn-env -c conda-forge scikit-learn
# or PyPI
pip install -U scikit-learn
```
<br>

**5- torch**
<br>
  To install
<br>
```
# conda
conda install pytorch torchvision torchaudio cpuonly -c pytorch
# or PyPI
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```
<br>

**6- imblearn**
  To install
  <br>
<br>
```
# conda
conda install conda-forge::imbalanced-learn
# or PyPI
pip install imblearn
```
<br>

## Project files
This repository contains the following :
<br><br>
**main.py**
<br>
Exploratory data analysis
<br>
**fraudDetector.py**
<br>
Artificial neural network model. Binary classifier built with torch
<br>
**model.py**
<br>
Prepare training and evaluation data, train model, evaluate model and save model parameters
<br>
**model_metrics.pickle**
<br>
Model training and evaluation metrics (training loss, F1 score, recall and precision)
<br>
**model.pth**
<br>
Model and optimizer parameters


## Acknowledgements
<a href="https://pytorch.org/get-started/locally/">torch download</a>
<br>
