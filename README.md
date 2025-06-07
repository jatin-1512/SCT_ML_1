# House Price Prediction - Linear Regression

This project implements a linear regression model to predict house prices using the Kaggle House Prices: Advanced Regression Techniques dataset.

## Features Used
- **GrLivArea**: Above grade (ground) living area square feet
- **BedroomAbvGr**: Number of bedrooms above basement level
- **FullBath**: Full bathrooms above grade

## Files
- `linear_regression_house_prices.py`: Main script for training and evaluating the model
- `train.csv`: Training data from Kaggle
- `.gitignore`: Excludes virtual environment, Kaggle credentials, and unnecessary files

## Setup & Usage
1. **Clone the repository**
2. **Create and activate a virtual environment** (recommended):
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```
3. **Install dependencies**:
   ```bash
   pip install pandas numpy scikit-learn matplotlib
   ```
4. **Run the script**:
   ```bash
   python linear_regression_house_prices.py
   ```

## Output
- Prints model evaluation metrics (Mean Squared Error, R² Score)
- Displays a scatter plot of actual vs. predicted house prices

## Dataset
- [Kaggle Competition: House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

## Notes
- Do not commit your `kaggle.json` or any sensitive credentials.
- The virtual environment and large data files are excluded via `.gitignore`. 