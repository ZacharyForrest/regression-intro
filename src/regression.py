import matplotlib.pyplot as plt
from scipy.stats import *
import statsmodels.api as sm

# Simulate data
tv_size = uniform.rvs(size=100, loc=0, scale=0)

other_factors = norm.rvs(size=100, loc=200, scale=100)

# The linear model!
theta = 10
tv_cost = (theta * tv_size) + other_factors


X = sm.add_constant(tv_size)
model = sm.OLS(tv_cost,X)
results = model.fit()

plt.scatter(tv_size, tv_cost)
plt.plot(tv_size, results.params[0] + tv_size * results.params[1], color = 'green' )
plt.vlines(tv_size, results.params[0] + tv_size * results.params[1], tv_cost, color='red')
plt.xlabel('TV Size')
plt.ylabel('Cost')
plt.show()




