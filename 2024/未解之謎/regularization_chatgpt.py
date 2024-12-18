import numpy as np
# 构造更复杂的数据集：二次关系并加入较大的噪声
np.random.seed(0)  # 设置随机种子以获得可重复的结果
x = np.linspace(1, 10, 10)
y = 3 * x**2 + np.random.normal(0, 20, size=x.shape)  # 二次关系 + 噪声
x = x.reshape(-1, 1)  # 1,10變成了10,1

# 为了拟合高次模型，创建高次特征
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso

poly = PolynomialFeatures(degree=10)  # 使用10次多项式特征来故意导致过拟合
x_poly = poly.fit_transform(x)  # 相當於x裡提取出的特徵，可以理解成1到10十個房子，創造出10階共11個特徵

# 创建三种模型：无正则化的线性回归，L2正则化（Ridge）和L1正则化（Lasso）
linear_model_poly = LinearRegression()
ridge_model_poly = Ridge(alpha=1)  # L2正则化系数 λ = 1
lasso_model_poly = Lasso(alpha=1)  # L1正则化系数 λ = 1

# 拟合模型
linear_model_poly.fit(x_poly, y)
ridge_model_poly.fit(x_poly, y)
lasso_model_poly.fit(x_poly, y)

# 生成预测值
x_vals = np.linspace(1, 10, 100).reshape(-1, 1)
x_vals_poly = poly.transform(x_vals)
linear_preds_poly = linear_model_poly.predict(x_vals_poly)
ridge_preds_poly = ridge_model_poly.predict(x_vals_poly)
lasso_preds_poly = lasso_model_poly.predict(x_vals_poly)

# 绘制拟合结果
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='black', label='Data Points')
plt.plot(x_vals, linear_preds_poly, label='No Regularization', color='blue')
plt.plot(x_vals, ridge_preds_poly, label='L2 Regularization (Ridge)', color='green')
plt.plot(x_vals, lasso_preds_poly, label='L1 Regularization (Lasso)', color='red')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("High-Degree Polynomial Fit with and without Regularization")
plt.grid(True)
plt.show()

# 输出每个模型的参数（只展示前5个系数便于比较）
linear_model_poly.coef_[:5], ridge_model_poly.coef_[:5], lasso_model_poly.coef_[:5]
