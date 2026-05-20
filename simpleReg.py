import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Age (X)
X = np.array([
    46, 20, 52, 30, 57, 25, 28, 36, 22, 43, 57, 33,
    22, 63, 40, 48, 28, 49, 52, 58, 29, 34, 24, 50
]).reshape(-1, 1)

# Cholesterol (Y)
y = np.array([
    3.5, 1.9, 4.0, 2.6, 4.5, 3.0, 2.9, 3.8, 2.1, 3.8, 4.1, 3.0,
    2.5, 4.6, 3.2, 4.2, 2.3, 4.0, 4.3, 3.9, 3.3, 3.2, 2.5, 3.3
])

model = LinearRegression()
model.fit(X, y)

print(f"Intercept (β0): {model.intercept_}")
print(f"Slope (β1): {model.coef_[0]}")

y_pred = model.predict(X)

plt.figure(figsize=(8, 5))
#Scatterplot 
plt.scatter(X, y, color='#3a5f7d', label="Actual Data") 
#Regressionline
plt.plot(X, y_pred, color='#e67e22', linewidth=2, label="Regression Line") 

plt.xlabel("Age")
plt.ylabel("Cholesterol Level")
plt.title("Simple Linear Regression (Age vs Cholesterol)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

residuals = y - y_pred

plt.figure(figsize=(8, 4))
plt.scatter(y_pred, residuals, color='#3a5f7d', alpha=0.7)
plt.axhline(0, color='black', linestyle='--')

plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

print("\nEvaluation Metrics:")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")



