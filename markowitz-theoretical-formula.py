'''
Author: Orlando Closs
Description: Code to calculate markowitz minimum variance, efficient frontier and tangent portfolio using 
    theoretical formula
Date: 03/10/2023
'''



import pandas as pd
import matplotlib.pyplot as plt
import sigfig
import seaborn as sns
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

#-------------------------markowitz class-------------------------

class MarkowitzPortfolio():
    def __init__(self, expected_returns, cov):
        self.expected_returns = expected_returns
        self.n = len(expected_returns)
        self.cov = cov
        self.incov = np.linalg.inv(cov)  # Inverse of the covariance matrix
        self.a = None
        self.b = None
        self.c = None
        self.ones_vector = np.ones(self.n)
        self.compute_abc()

    def compute_abc(self):
        self.a = self.expected_returns.T @ self.incov @ self.ones_vector
        self.b = self.expected_returns.T @ self.incov @ self.expected_returns
        self.c = self.ones_vector.T @ self.incov @ self.ones_vector
    
    def minimum_variance(self):
        volatility = 1 / np.sqrt(self.c)
        return_mv = self.a / self.c 
        weights = (1/self.c) * (self.incov @ self.ones_vector)
        self.return_mv = return_mv
        self.risk_mv = volatility
        self.r0 = return_mv/2
        return volatility, return_mv, weights

    def optimal_risk_formula(self, r):
        variance = (self.c * (((r - (self.a/self.c))**2) / (self.b*self.c - (self.a**2)))) + (1 / self.c)
        return variance
    
    def efficient_frontier(self, num_points=100):
        min_return = self.return_mv
        max_return = max(self.expected_returns) 
        target_returns = np.linspace(min_return, max_return, num_points)
        portfolio_volatilities = []
        for target_return in target_returns:
            variance = self.optimal_risk_formula(target_return)
            if variance is not None:
                portfolio_volatilities.append(np.sqrt(variance))
            else:
                break
        return (portfolio_volatilities, target_returns)
    
    def tangent_portfolio(self):
        portfolio_volatilities, target_returns = self.efficient_frontier()
        max_sharpe_ratio=float('-Inf')
        optimal_p_return=0
        optimal_p_risk=0
        for index, target_return in enumerate(target_returns):
            volatility = portfolio_volatilities[index]
            sharpe_ratio=(target_return-self.r0)/(volatility)
            if sharpe_ratio > max_sharpe_ratio:
                max_sharpe_ratio = sharpe_ratio
                optimal_p_return = target_return
                optimal_p_risk = volatility
        
        return optimal_p_return, optimal_p_risk
    
    def get_weights_for_return(self,r):
        x0 = (self.b - self.a * r) / (self.b * self.c - self.a ** 2)
        xr = (self.c * r - self.a) / (self.b * self.c - self.a ** 2)
        weights = self.incov @ (x0 * self.ones_vector + xr * self.expected_returns)
        return weights
    
    def plot_efficient_frontier(self, color = 'blue'):
        portfolio_volatilities, target_returns = self.efficient_frontier()
        plt.plot(portfolio_volatilities, target_returns[:len(portfolio_volatilities)], '-', color=color, label='{} asset efficient frontier'.format(self.n))
        plt.xlabel('Portfolio Volatility')
        plt.ylabel('Expected Return')
        title = '{} asset portfolio efficient frontier'.format(self.n)
        plt.title(title)
        plt.legend(loc='upper left')
        plt.grid(True)

    def plot_tangent_portfolio(self):
        optimal_p_return, optimal_p_risk = self.tangent_portfolio()
        weights = [0, 1, 1.1]
        x_values = []
        y_values = []
        
        for weight in weights:
            y = (weight * optimal_p_return) + ((1 - weight) * self.r0)
            x = weight * optimal_p_risk
            x_values.append(x)
            y_values.append(y)

        title = '{} asset portfolio efficient frontier with capital allocation line'.format(self.n)
        plt.title(title)
        
       # Highlight the tangent portfolio with a red dot at weight 1
        plt.scatter(optimal_p_risk, optimal_p_return, color='red', s=150, label='Tangent Portfolio')
            
        # Connect the points at the ends with a line to form the Capital Allocation Line
        plt.plot([x_values[0], x_values[2]], [y_values[0], y_values[2]], 'r--', label='Capital Allocation Line')
    

#-------------------------preprocessing data-------------------------

data_list=['AAPL-weekly.csv', 'GOOG-weekly.csv', 'MSFT-weekly.csv', \
           'NVDA-weekly.csv',  'SONY-weekly.csv']

def compute_returns(file_path):
    data = pd.read_csv(file_path) #read csv file
    data['Date'] = pd.to_datetime(data['Date']) #change date format
    data = data.sort_values(by='Date').reset_index(drop=True) #sorts list by date 
    data['Returns'] = data['Close'].pct_change() *100 #makes new column in data and \
    #pct_change calulates percentage change
    return data

#make dictionary and add returns to prepare for dataframe
asset_returns={}
for csv in data_list:
    data=compute_returns(csv)
    stock_name=csv[0:4]
    asset_returns[stock_name] = data['Returns']

returns_dataframe = pd.DataFrame(asset_returns)

#get covariance matrices
cov_matrix_3 = returns_dataframe[['AAPL','GOOG', 'MSFT']].cov() 
#covariance matrix function 
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.cov.html
cov_matrix_4 = returns_dataframe[['AAPL', 'GOOG', 'MSFT', 'NVDA']].cov()
cov_matrix_5 = returns_dataframe[['AAPL', 'GOOG', 'MSFT', 'NVDA', 'SONY']].cov()

expected_returns_3 = returns_dataframe[['AAPL','GOOG', 'MSFT']].mean(axis=0).values #more direct way calculating mean values
expected_returns_4 = returns_dataframe[['AAPL', 'GOOG', 'MSFT', 'NVDA']].mean(axis=0).values
expected_returns_5 = returns_dataframe[['AAPL', 'GOOG', 'MSFT', 'NVDA', 'SONY']].mean(axis=0).values

cov_matrix_3 = cov_matrix_3.values #grabs values ready for calculation
cov_matrix_4 = cov_matrix_4.values
cov_matrix_5 = cov_matrix_5.values

#-------------------------perform actions-------------------------
    
three_asset = MarkowitzPortfolio(expected_returns_3, cov_matrix_3)
four_asset = MarkowitzPortfolio(expected_returns_4, cov_matrix_4)
five_asset = MarkowitzPortfolio(expected_returns_5, cov_matrix_5)

#three asset minimum variance and efficient frontier

volatility_mv_3, return_mv_3, weights_mv_3 = three_asset.minimum_variance()
print('\n-----MINIMUM VARIANCE 3 ASSET-----')
print('\nAAPL, GOOG, MSFT')
print('WEIGHTS: {}, {}, {}'.format(weights_mv_3[0], weights_mv_3[1], weights_mv_3[2]))
print('EXPECTED RETURN: {}'.format(return_mv_3))
print('VOLATILITY: {}'.format(volatility_mv_3))

three_asset.plot_efficient_frontier()
plt.savefig('efficient_frontier_3.png')

plt.clf()

#four asset minimum variance and efficient frontier

volatility_mv_4, return_mv_4, weights_mv_4 = four_asset.minimum_variance()
print('\n-----MINIMUM VARIANCE 4 ASSET-----')
print('\nAAPL, GOOG, MSFT, NVDA')
print('WEIGHTS: {}, {}, {}, {}'.format(weights_mv_4[0], weights_mv_4[1], weights_mv_4[2], weights_mv_4[3]))
print('EXPECTED RETURN: {}'.format(return_mv_4))
print('VOLATILITY: {}'.format(volatility_mv_4))

four_asset.plot_efficient_frontier(color='red')
plt.savefig('efficient_frontier_4.png')

plt.clf()

#five asset minimum variance and efficient frontier

volatility_mv_5, return_mv_5, weights_mv_5 = five_asset.minimum_variance()
print('\n-----MINIMUM VARIANCE 5 ASSET-----')
print('\nAAPL, GOOG, MSFT, NVDA, SONY')
print('WEIGHTS: {}, {}, {}, {}, {}'.format(weights_mv_5[0], weights_mv_5[1], weights_mv_5[2], weights_mv_5[3], weights_mv_5[4]))
print('EXPECTED RETURN: {}'.format(return_mv_5))
print('VOLATILITY: {}'.format(volatility_mv_5))

five_asset.plot_efficient_frontier(color='green')
plt.savefig('efficient_frontier_5.png')

#tangent portfolio

five_asset.plot_tangent_portfolio()

plt.savefig('efficient_frontier_5_tangent.png')

plt.clf()

optimal_p_return, optimal_p_risk = five_asset.tangent_portfolio()
optimal_weights = five_asset.get_weights_for_return(optimal_p_return)

print('\n-----OPTIMAL PORTFOLIO 5 ASSET-----')
print('AAPL, GOOG, MSFT, NVDA, SONY')
print('WEIGHTS: {}, {}, {}, {}, {}'.format(optimal_weights[0], optimal_weights[1], optimal_weights[2], optimal_weights[3], optimal_weights[4]))
print('EXPECTED RETURN: {}'.format(optimal_p_return))
print('VOLATILITY: {}'.format(optimal_p_risk))

# all asset frontier

three_asset.plot_efficient_frontier()
four_asset.plot_efficient_frontier(color='red')
five_asset.plot_efficient_frontier(color='green')
plt.savefig('efficient_frontier_all.png')
