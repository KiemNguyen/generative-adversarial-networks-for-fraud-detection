# Generative Adversarial Networks for Fraud Detection
![Preview](https://github.com/KiemNguyen/generative-adversarial-networks-for-fraud-detection/blob/master/assets/gans.png)

Losses due to fraud are routinely estimated to be in the billions of dollars annually. This project attempts to solve class imbalance problem to build a better fraud detection model.

## Requirements
* Linux or Mac
* Python 3.6
* Pip

## Instalation
Get into the project folder

Setting up the virtual env
```bash
pip install virtualenv
virtualenv gan_frauds
source gan_frauds/bin/activate
```

Installing dependencies
```bash
pip install -r requirements.txt
```

Downloading dataset using Kaggle command line tool
```bash
pip install kaggle-cli
kg download -u <username> -p <password> -c creditcardfraud -f creditcard.csv
```

## Start

```bash
python training.py
```

You will train two networks to battle together. Eventually, the 'fake' dataset will look very similar to the 'real' dataset.
