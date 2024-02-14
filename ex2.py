'''
Author: Orlando Closs
Class: Iode to calculate markowitz minimum variance, efficient frontier and tangent portfolio using 
    scipy optimisation function
Date: 15/09/2023
'''


import pandas as pd
import matplotlib.pyplot as plt
import sigfig
import seaborn as sns
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

#-------------------------minimum variance class-------------------------

class MinimumVariance():
    def __init__(self, cov, expected_returns, n):
        self.cov = cov
        self.expected_returns = expected_returns
        self.n = n
    
    # portfolio variance objective function
    def objective(self, weights):
        return weights.T @ self.cov @ weights

    #in order to minimise the portfolio variance we are going to use scipy minimise
    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    def optimise(self):
        #type and eq mean equality constraint where 'fun' defines the function
        constraints = [
            {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}] #weights equal 1
        #     {"type": "eq", "fun": lambda weights: np.dot(weights, self.expected_returns) - np.median(self.expected_returns)}
        # ] #no constraint for return for minimum variance
        
        # inital values needed for minimise function
        initial_weights = [(1./self.n) for i in range(self.n)]
        bounds = [(0.05, 1) for i in range(self.n)]

        # run optimisation
        solution = minimize(self.objective, initial_weights, bounds=bounds, constraints=constraints, method='SLSQP')
        #SLSQP is  Sequential Least Squares Quadratic Programming was recommended by stack overflow
        #optimisation algorithm which is why initial values are needed
        weights = solution.x
        variance = solution.fun
        volatility = np.sqrt(variance)
        exp_return = np.dot(weights, self.expected_returns)
        return weights, exp_return, volatility

#-------------------------efficient frontier and tangent portfolio-------------------------

class EfficientFrontier():
    def __init__(self, cov, expected_returns, weights, min_return):
        self.cov = cov
        self.expected_returns = expected_returns
        self.n = len(expected_returns)
        self.starting_weights=weights
        self.target=min_return
        self.r0=min_return/2

    def objective(self, weights):
        # Objective function to minimize (portfolio variance)
        return weights.T @ self.cov @ weights

    def optimize(self, target_return):
        constraints = [
            {"type": "eq", "fun": lambda weights: np.sum(weights) - 1},
            {"type": "eq", "fun": lambda weights: np.dot(weights, self.expected_returns) - target_return}
        ]
        
        bounds = [(0.05, 1) for _ in range(self.n)]

        # Use the minimum variance weights as the starting point for optimization
        solution = minimize(self.objective, self.starting_weights, bounds=bounds, constraints=constraints, method='SLSQP')
        
        #preventing frontier from increasing verticly 
        if solution.success:
            variance = solution.fun
            weights = solution.x
            return variance, weights
        else:
            return None, None
    
    def plot_tangent(self,optimal_p_return, optimal_p_risk):
        weights = [0, 1, 1.1]
        x_values = []
        y_values = []
        
        for weight in weights:
            y = (weight * optimal_p_return) + ((1 - weight) * self.r0)
            x = weight * optimal_p_risk
            x_values.append(x)
            y_values.append(y)
        
       # Highlight the tangent portfolio with a red dot at weight 1
        plt.scatter(optimal_p_risk, optimal_p_return, color='red', s=150, label='Tangent Portfolio')
            
        # Connect the points at the ends with a line to form the Capital Allocation Line
        plt.plot([x_values[0], x_values[2]], [y_values[0], y_values[2]], 'r--', label='Capital Allocation Line')

    def plot_frontier(self, num_points=100, color='blue'):
        min_return = self.target
        max_return = max(self.expected_returns)

        target_returns = np.linspace(min_return, max_return, num_points)
        portfolio_volatilities = []
        max_sharpe_ratio=float('-Inf')
        optimal_p_return=0
        optimal_p_risk=0
        optimal_weights=[]
        for target_return in target_returns:
            variance, weights = self.optimize(target_return)
            
            if variance is not None:
                portfolio_volatilities.append(np.sqrt(variance))
                sharpe_ratio=(target_return-self.r0)/(np.sqrt(variance))
                if sharpe_ratio > max_sharpe_ratio:
                    max_sharpe_ratio = sharpe_ratio
                    optimal_p_return = target_return
                    optimal_p_risk = (np.sqrt(variance))
                    optimal_weights = weights
            else:
                # Stop plotting when the optimization is not successful
                break
        
        self.plot_tangent(optimal_p_return, optimal_p_risk)

        plt.plot(portfolio_volatilities, target_returns[:len(portfolio_volatilities)], '-', color=color, label='{} asset efficient frontier'.format(self.n))
        plt.xlabel('Portfolio Volatility')
        plt.ylabel('Expected Return')
        plt.title('Efficient Frontier with Tangent Portfolio')
        plt.legend(loc='upper left')
        plt.grid(True)

        return optimal_weights, optimal_p_return, optimal_p_risk

