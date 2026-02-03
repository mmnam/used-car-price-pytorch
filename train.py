import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import kagglehub
from kagglehub import KaggleDatasetAdapter
import os

# --------------------
# 1) Load dataset and clean variables
# --------------------
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "taeefnajib/used-car-price-prediction-dataset",
  "used_cars.csv"
)

# price transformation: "$12,345" -> 12345. This is our variable for prediciton y
price = (df["price"].astype(str)
         .str.replace("$", "", regex=False)
         .str.replace(",", "", regex=False)
         .astype(float))

df["price_num"] = price

# filter extreme outliers
df = df[df["price_num"].between(1000, 300000)]

# mileage transformation: "12,345 mi." -> 12345. This is one of our predictor variables X
milage = (df["milage"].astype(str)
          .str.replace(",", "", regex=False)
          .str.replace(" mi.", "", regex=False)
          .astype(float))

df["milage_num"] = milage

# This is another of our predictor variables X
age = df["model_year"].max() - df["model_year"]
df["age"] = age

# Stacking our 2 features in Matrix X. Converting milage to log because it's very skewed
# X = [age, log1p(mileage)]
X = np.column_stack([
    df["age"].to_numpy(dtype=np.float32),
    np.log1p(df["milage_num"].to_numpy(dtype=np.float32))
])
print(X.shape)

# y = log1p(price) (predict log-price, will convert back to dollars later)
y = np.log1p(df["price_num"].to_numpy(dtype=np.float32)).reshape(-1, 1)

# --------------------
# 2) Creating the Train and validation split
# --------------------
# random number generator, makes sure you get the same random split every time
rng = np.random.default_rng(42)
# creates a list with all the indices
idx = np.arange(len(X))
rng.shuffle(idx)

split = int(0.85 * len(X))
tr_idx, va_idx = idx[:split], idx[split:]

X_tr, y_tr = X[tr_idx], y[tr_idx]
X_va, y_va = X[va_idx], y[va_idx]

# We normalize X using TRAIN stats only to avoid leakage
# axis=0 means compute the mean per column, one for each feature
# keepdims=True to prevent the 2D matrix to become a 1D vector
X_mean = X_tr.mean(axis=0, keepdims=True)
# + 1e-8 to prevent divisiion by zero if there isn't much variance
X_std = X_tr.std(axis=0, keepdims=True) + 1e-8
X_tr = (X_tr - X_mean) / X_std
X_va = (X_va - X_mean) / X_std

# tensors
X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
X_va_t = torch.tensor(X_va, dtype=torch.float32)
y_va_t = torch.tensor(y_va, dtype=torch.float32)

# to feed the data to the model in mini batches
train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=256, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_va_t, y_va_t), batch_size=512, shuffle=False)

# --------------------
# 3) Models: linear vs non linear
# --------------------
linear_model = nn.Linear(2, 1)

# using ReLU, non linear activation
nonlinear_model = nn.Sequential(
    nn.Linear(2, 32),    
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
)

# --------------------
# 4) Defining train and evaluate helpers
# --------------------
loss_fn = nn.MSELoss()

# for converting log back to dollars when predicting
def dollars_from_log(log_y: torch.Tensor) -> torch.Tensor:
    return torch.expm1(log_y)

# same as: with torch.no_grad()
# mae: mean absolute error
# rmse: root mean squared error
@torch.no_grad()
def eval_mae_rmse(model):
    model.eval()
    preds = model(X_va_t)
    pred_dollars = dollars_from_log(preds)
    true_dollars = dollars_from_log(y_va_t)

    # item.() turns tensor into a float
    err = pred_dollars - true_dollars    
    mae = torch.mean(torch.abs(err)).item()
    rmse = torch.sqrt(torch.mean(err ** 2)).item()
    return mae, rmse

# Adam optimizer is best for initial experimentation
def train_model(model, epochs=200, lr=0.001):    
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

        if epoch % 20 == 0 or epoch == 1:
            mae, rmse = eval_mae_rmse(model)
            print(f"epoch {epoch:03d} | val MAE=${mae:,.0f} | val RMSE=${rmse:,.0f}")

# --------------------
# 5) Running both models
# --------------------
print("\n--- Linear baseline ---")
train_model(linear_model, epochs=200, lr=0.01)

print("\n--- Non-linear MLP ---")
train_model(nonlinear_model, epochs=250, lr=0.001)

# --------------------
# 6) Saving the better model
# --------------------
mae_lin, rmse_lin = eval_mae_rmse(linear_model)
mae_nn, rmse_nn = eval_mae_rmse(nonlinear_model)

best = nonlinear_model if rmse_nn < rmse_lin else linear_model
best_name = "mlp" if rmse_nn < rmse_lin else "linear"

os.makedirs("./model", exist_ok=True)
torch.save({
    "model_state": best.state_dict(),
    "model_type": best_name,
    "X_mean": torch.tensor(X_mean),
    "X_std": torch.tensor(X_std),
}, "./model/car_price_model.pt")

print(f"\nSaved best model: {best_name} -> car_price_model.pt")

# --------------------
# 7) Example prediction
# --------------------
@torch.no_grad()
def predict_price(model, age_years, milage_value):
    x = np.array([[age_years, np.log1p(milage_value)]], dtype=np.float32)
    x = (x - X_mean) / X_std
    x_t = torch.tensor(x, dtype=torch.float32)
    log_pred = model(x_t)
    return float(torch.expm1(log_pred).item())

best.eval()
print("\nExample predictions:")
print("Age=5, mileage=10,000 -> $", round(predict_price(best, 5, 10000)))
print("Age=2, mileage=10,000 -> $", round(predict_price(best, 2, 10000)))