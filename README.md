# Used Car Price Prediction (PyTorch)

Small PyTorch project to predict used car prices from a few simple features.
I started with a linear baseline and then compared it to a small non-linear neural network (MLP) to capture non-linear depreciation patterns (e.g., larger price drop in early years).

## Dataset

Source: Kaggle — Used Car Price Prediction Dataset (used_cars.csv)

Columns in the dataset include:

brand, model, model_year, milage, fuel_type, engine, transmission, ext_col, int_col, accident, clean_title, price

This project intentionally uses a minimal feature set to keep the model and explanations simple.

## Features used

From the original dataset, I use:

- age = max(model_year) - model_year

- mileage = milage cleaned as a number

- (transform) log1p(mileage): I use log1p to reduce skew in mileage values (very high mileage values can dominate training otherwise)

## Target:

- I train on log1p(price) and convert back to dollars when reporting MAE/RMSE. This stabilizes training because prices are skewed.

## Models
1) Linear baseline

A single linear layer:

- nn.Linear(2, 1)

This fits a plane in the feature space (age, mileage → price).

2) Non-linear model (MLP)

A small feed-forward network:

- nn.Linear(2, 32) → ReLU → nn.Linear(32, 16) → ReLU → nn.Linear(16, 1)

It’s non-linear because ReLU introduces a non-linear activation. That lets the model learn piecewise patterns instead of a single straight plane.

## Training setup

- Train/validation split: 85% / 15%

- Normalization: feature mean/std computed on train only and applied to both train + validation (prevents leakage)

- Loss function: MSELoss

- Optimizer: Adam (chosen as a reliable default that usually converges faster than plain SGD)

- Validation metrics (reported in dollars):

- MAE (Mean Absolute Error)

- RMSE (Root Mean Squared Error)

## How to run

1. Install dependencies:

`pip install -r requirements.txt`

2. Train and compare models:

`python train.py`

The script prints validation MAE/RMSE periodically and saves the best model checkpoint:

`car_price_model.pt`

## Outputs

- Console logs showing validation MAE/RMSE for:

  - linear model

  - non-linear MLP

- Saved checkpoint:

  - includes model weights + normalization stats (X_mean, X_std) so the model can be used later for inference

## Why this project

This repo is a small exercise in:

- cleaning real-world tabular data

- building a baseline model

- improving it with a simple non-linear network

- evaluating performance on a validation split

- saving a trained model for reuse

## Results

Final / best validation performance from one run:

### Model	Best Val MAE	
Linear model:	$15,824 (around epoch 180)

**Non-linear MLP:	$14,463 (epoch 200)**

### Model Best Val RMSE
Linear model:	$27,421 (epoch 200)

**Non-linear MLP: $24,453 (epoch 240)**
