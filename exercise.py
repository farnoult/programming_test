import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Logic


def smallest_difference(array):
    '''
    Function that takes an array and returns the smallest
    absolute difference between two elements of this array
    '''
    if len(array) != len(list(set(array))): #avoid computation if there are duplicates
        return 0
    else:
        arr = np.diff(np.sort(np.array(array))) #convert to ndarray to avoid problem with your test set (if it's not ndarray)
        arr = [abs(x) for x in arr]
        return np.min(arr)

def smallest_difference2(array):
    '''
    More readible solution
    '''
    if len(array) != len(list(set(array))): #avoid computation if there are duplicates
        return 0
    else:
        arr = sorted(array)
        smallest_dif = abs(arr[0]-arr[1])
        for i in range(len(arr)-1):
            if abs(arr[i]-arr[i+1]) < smallest_dif:
                smallest_dif = abs(arr[i]-arr[i+1])
        return smallest_dif


# Finance and DataFrame manipulation


def macd(prices, window_short=13, window_long=26,exp_ma=False):
    '''
    Function that takes a DataFrame named prices and returns it's MACD as a DataFrame with same shape
    MACD = short period MA - long period MA
    '''
    df = prices.copy(deep=True)
    
    if exp_ma==True:
        df['MACD' + str(window_short)+'_'+ str(window_long)] = prices['SX5T Index'].ewm(span = window_short).mean() - prices['SX5T Index'].ewm(span = window_long).mean()
    
    else:
        df['MACD' + str(window_short)+'_'+ str(window_long)] = prices['SX5T Index'].rolling(window = window_short).mean() - prices['SX5T Index'].rolling(window = window_long).mean()
        
    return df  


def sortino_ratio(prices, target = 0): #the result is not exactly as the expected one, I think it comes from a mistake when I am trying to annualize my returns
    '''
    Function that takes a DataFrame named prices and returns the Sortino ratio for each column
    Sortino ratio: S(r*) = E(r-r*)/sigma(-)
    It takes only into account the downside risk
    '''
    sortino = []
    col = prices.columns
    for i in range(1,len(col)): #return the Sortino ratio for each column
        returns = np.array((prices[col[i]] - prices[col[i]].shift())/ prices[col[i]].shift())[1:] #not take the first NaN value
        adj_returns = returns - target

        num = ((1+adj_returns).prod())**(252/len(returns))-1
        den = np.std(adj_returns[adj_returns<target],ddof=1)*np.sqrt(252)
        
        
        sortino.append(num/den)
        
    return sortino


def expected_shortfall(prices, level=0.95):
    '''
    Function that takes a DataFrame named prices and returns the expected shortfall at a given level
    Expected Shortfall represents the mean of returns that are below the VaR
    '''
    returns = np.array((prices['SX5T Index'] - prices['SX5T Index'].shift())/ prices['SX5T Index'].shift())[1:]
    return np.mean(returns[returns <= np.percentile(returns, (1-level)*100)])


# Plot 


#/!\/!\ path argument is supposed to be in the following form : r'C:\Users\felix\Documents /!\/!\

def visualize(prices, path): 
    '''
    Function that takes a DataFrame named prices and saves the plot to the given path
    '''
    x = prices[prices.columns[0]]
    y = prices[prices.columns[1]]

    fig = plt.figure(figsize=(12,6))
    ax = plt.plot(x,y)
    ax = plt.title('Eurostoxx 50', fontsize=20)
    ax = plt.xlabel('Dates', fontsize=16)
    ax = plt.xticks(x[::150], rotation=45) #only display certain dates to more readability
    ax = plt.ylabel('Prices', fontsize=16)
    
    plt.savefig(path + '\\SX5T_FelixARNOULT.png')
