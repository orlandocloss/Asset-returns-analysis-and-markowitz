'''
Author: Orlando Closs
Description: Code to calculate returns, mean and volatility (with table), plot time series 
             and plot empirical vs normal graph
Date: 14/09/2023
'''


import pandas as pd
import matplotlib.pyplot as plt
import sigfig
import seaborn as sns
import numpy as np
from scipy.stats import norm


data_list=['AAPL-daily.csv', 'AAPL-weekly.csv', 'GOOG-daily.csv', \
           'GOOG-weekly.csv', 'MSFT-daily.csv', 'MSFT-weekly.csv', \
            'NVDA-daily.csv', 'NVDA-weekly.csv', 'SONY-daily.csv', 'SONY-weekly.csv']

def compute_returns(file_path):
    data = pd.read_csv(file_path) #read csv file
    data['Date'] = pd.to_datetime(data['Date']) #change date format
    data = data.sort_values(by='Date').reset_index(drop=True) #sorts list by date 
    data['Returns'] = data['Close'].pct_change() *100 #makes new column in data and \
    #pct_change calulates percentage change
    return data

def plot_time_series(data, weekly_data, stock_name):
    #plots daily returns over time
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    title='{} Daily Returns'.format(stock_name)
    plt.plot(data['Date'], data['Returns'], color='blue')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Return (%)')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    title='{} Weekly Returns'.format(stock_name)
    plt.plot(weekly_data['Date'], weekly_data['Returns'], color='red')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Return (%)')
    plt.legend()
    
    plt.tight_layout()
    
    # save the plot
    filename = "{}_returns.png".format(stock_name)
    plt.savefig(filename)

def plot_empirical_vs_normal(data, weekly_data, stock_name, daily_mean, \
                             daily_volatility, weekly_mean, weekly_volatility):
    daily_returns = data['Returns'].dropna() #gets returns data drops missing values
    weekly_returns = weekly_data['Returns'].dropna()

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    sns.kdeplot(daily_returns, label="Empirical Density", shade=True) #makes empirical \
    #density graph https://seaborn.pydata.org/generated/seaborn.kdeplot.html
    x_daily = np.linspace(daily_returns.min(), daily_returns.max(),\
                           1000) #empty data for x axis for normal distribution
    plt.plot(x_daily, norm.pdf(x_daily, daily_mean, daily_volatility), \
             'r-', label="Normal Distribution") #plots normal distribution
    plt.title(f"{stock_name} Daily Returns: Empirical Density vs. Normal Distribution")
    plt.xlabel("Returns (%)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True) #adds gridlines - useful for this type of graph

    plt.subplot(1, 2, 2)
    sns.kdeplot(weekly_returns, label="Empirical Density", shade=True)
    x_weekly = np.linspace(weekly_returns.min(), weekly_returns.max(), 1000)
    plt.plot(x_weekly, norm.pdf(x_weekly, weekly_mean, weekly_volatility), 'r-', label="Normal Distribution")
    plt.title(f"{stock_name} Weekly Returns: Empirical Density vs. Normal Distribution")
    plt.xlabel("Returns (%)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)

    #plots these two graphs in one image
    plt.tight_layout()
    filename = f"{stock_name}_empirical_normal.png"
    plt.savefig(filename)
    
def mean_and_volatility(data):
    mean = data['Returns'].mean()
    volatility = data['Returns'].std()
    return mean, volatility


# empty table to store results
mean_volatility_table = pd.DataFrame(columns=['Stock Name', 'Type', 'Mean (%)', 'Volatility'])

for index,csv in enumerate(data_list):

    if (index%2==0): #every other file
        stock_name=csv[0:4] #first four letters
        daily_data=compute_returns(csv)
        weekly_data=compute_returns(data_list[index+1])
        plot_time_series(daily_data, weekly_data, stock_name)

        daily_mean, daily_volatility = mean_and_volatility(daily_data)
        weekly_mean, weekly_volatility = mean_and_volatility(weekly_data)

        plot_empirical_vs_normal(daily_data, weekly_data, stock_name, daily_mean,\
                                  daily_volatility, weekly_mean, weekly_volatility)

        daily_mean=sigfig.round(daily_mean,3) #round to 3 significant figures 
        daily_volatility=sigfig.round(daily_volatility,3)
        weekly_mean=sigfig.round(weekly_mean,3) 
        weekly_volatility=sigfig.round(weekly_volatility,3)
        
        # add daily results to table
        index2 = len(mean_volatility_table)
        mean_volatility_table.loc[index2] = [stock_name, 'Daily', daily_mean, daily_volatility]

        # add weekly results to table
        index2 = len(mean_volatility_table)
        mean_volatility_table.loc[index2] = [stock_name, 'Weekly', weekly_mean, weekly_volatility]
        
#makes table plot and saves table image
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('off')
ax.axis('tight')
ax.table(cellText=mean_volatility_table.values, colLabels=mean_volatility_table.columns,\
          cellLoc = 'center', loc='center')
plt.savefig('mean_volatility_table.png')
plt.close()
