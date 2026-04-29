# Polynomial Regression - Position Salary Prediction

This project demonstrates how to build and visualize a **Polynomial Regression** model in Python using a small position-level salary dataset. The goal is to compare a basic Linear Regression model with a Polynomial Regression model and show how polynomial features can better capture non-linear salary growth.

## Project Overview

The dataset contains job positions, position levels, and salaries. The model uses the numeric **Level** column as the independent variable and **Salary** as the target variable.

The project includes:

- Loading and preparing data with **pandas**
- Creating feature and target arrays with **NumPy**
- Training a **Linear Regression** model
- Transforming features using **PolynomialFeatures**
- Training a Polynomial Regression model using **LinearRegression**
- Visualizing Linear Regression and Polynomial Regression results with **Matplotlib**
- Predicting salary for a position level of `6.5`

## Technologies Used

- Python
- NumPy
- pandas
- Matplotlib
- scikit-learn
- LinearRegression
- PolynomialFeatures
- train_test_split *(commonly used in regression workflows; this specific small-dataset example trains on the full dataset)*
- VS Code

## Dataset

The project uses `Position_Salaries.csv`.

Expected columns:

| Column | Description |
|---|---|
| `Position` | Job position title |
| `Level` | Numeric level assigned to the position |
| `Salary` | Salary for the position |

Example rows:

| Position | Level | Salary |
|---|---:|---:|
| Business Analyst | 1 | 45000 |
| Junior Consultant | 2 | 50000 |
| Senior Consultant | 3 | 60000 |
| Manager | 4 | 80000 |
| Country Manager | 5 | 110000 |

## Project Structure

```text
.
├── polynomial_regression.ipynb
├── polynomial_regression.py
├── Position_Salaries.csv
└── README.md
```

## Installation

Clone this repository:

```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```

Install the required Python libraries:

```bash
pip install numpy pandas matplotlib scikit-learn
```

## How to Run

### Option 1: Run the Python script

```bash
python polynomial_regression.py
```

### Option 2: Run the Jupyter Notebook

```bash
jupyter notebook polynomial_regression.ipynb
```

Or open the notebook directly in VS Code.

## Important Note About File Paths

The uploaded Python script uses a local Windows file path:

```python
dataset = pd.read_csv(r"C:\Users\PC\Desktop\VScode\Position_Salaries.csv")
```

For GitHub portability, update it to a relative path:

```python
dataset = pd.read_csv("Position_Salaries.csv")
```

This allows the code to run as long as `Position_Salaries.csv` is in the same folder as the script or notebook.

## Model Workflow

1. Import the required libraries.
2. Load the salary dataset.
3. Select `Level` as the input feature.
4. Select `Salary` as the target variable.
5. Train a Linear Regression model on the full dataset.
6. Create polynomial features with degree `4`.
7. Train a second Linear Regression model on the transformed polynomial features.
8. Visualize both Linear and Polynomial Regression results.
9. Predict salary for level `6.5` using both models.

## Key Code Snippet

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

prediction = lin_reg_2.predict(poly_reg.transform([[6.5]]))
print(prediction)
```

## Visualizations

The project generates three plots:

1. **Linear Regression result**  
   Shows how a straight-line model fits the salary data.

2. **Polynomial Regression result**  
   Shows how a polynomial curve fits the salary data more closely.

3. **High-resolution Polynomial Regression curve**  
   Uses a smoother grid of values to create a cleaner polynomial curve.

## Learning Objectives

By completing this project, you will understand:

- How Linear Regression works
- Why Linear Regression may not fit non-linear data well
- How PolynomialFeatures transforms input data
- How Polynomial Regression can model curved relationships
- How to visualize regression models using Matplotlib
- How to make predictions with trained scikit-learn models

## Example Prediction

The project predicts salary for a position level of `6.5`:

```python
lin_reg.predict([[6.5]])
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
```

The polynomial model is expected to give a more realistic prediction for this dataset because the salary values increase non-linearly as position level increases.

## Future Improvements

Possible improvements include:

- Add model evaluation metrics such as MAE, MSE, RMSE, or R² score
- Use `train_test_split` for larger datasets
- Try different polynomial degrees and compare performance
- Save the trained model using `joblib` or `pickle`
- Add an interactive prediction input
- Convert the notebook into a complete machine learning report

