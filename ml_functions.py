# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 19:25:23 2021

@author: Jeswin
"""
from collections import Counter
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import time

sns.set(style="ticks", rc={'figure.figsize':(7,6)})
sns.set_context(rc = {"font.size":15, "axes.labelsize":15}, font_scale=2)
sns.set_palette('colorblind');
from pandas.api.types import CategoricalDtype
# pandas defaults
np.set_printoptions(precision=4)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

import warnings
warnings.filterwarnings('ignore')

from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax, chi2_contingency, chi2, entropy

# ---------------------------------------------------------------------------------------------------------------------------------

def phi_coefficient(a,b):
    """
    Used to find the phi-coefficient between 2 nominal binary variables. Its analogus to correlation in continuous variables.
    
    Parameters:
    Binary nominal categorical/object pandas series: a,b
    
    Returns:
    float: returning value
    """
    temp = pd.crosstab(a,b)
    nr = (temp.iloc[1,1] * temp.iloc[0,0]) - (temp.iloc[0,1]*temp.iloc[1,0])
    rs = np.prod(temp.apply(sum, axis = 'index').to_list(), dtype=np.int64)
    cs = np.prod(temp.apply(sum, axis = 'columns').to_list(), dtype=np.int64)
    return(round(nr/np.sqrt(rs*cs),4))
	
# ---------------------------------------------------------------------------------------------------------------------------------

def cramers_v(a,b):
    """
    Used to find the phi-coefficient between 2 nominal binary variables. Its analogus to correlation in continuous variables.
    
    Parameters:
    Binary nominal categorical/object pandas series: a,b
    
    Returns:
    float: returning value
    """
    crosstab = pd.crosstab(a,b)
    chi2v = chi2_contingency(crosstab)[0]  # chi-squared value
    n = crosstab.sum().sum()
    phi2 = chi2v/n
    r, k = crosstab.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return(np.sqrt(phi2corr/min((kcorr-1),(rcorr-1))))

# ---------------------------------------------------------------------------------------------------------------------------------

        
def generate_heatmap(df):
    """
    Generate a heatmap with the upper triangular matrix masked
    Compute the correlation matrix
    
    Parameters:
    Pandas dataframe having numerical values
    
    Returns:
    float: heatmap with correlation between 2 the numerical values
    """
    corr = df.corr(method="spearman")
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=np.bool))
    plt.figure(figsize = (15,9));
    # Draw the heatmap with the mask 
    sns.heatmap(corr, mask=mask, cmap='coolwarm', fmt = '.2f', linewidths=.5, annot = True);
    plt.title("Correlation heatmap");
    return

# ---------------------------------------------------------------------------------------------------------------------------------
    
def conditional_entropy(x, y):
    """
    Calculates the conditional entropy of x given y: S(x|y)
    Wikipedia: https://en.wikipedia.org/wiki/Conditional_entropy
    :param x: list / NumPy ndarray / Pandas Series
        A sequence of measurements
    :param y: list / NumPy ndarray / Pandas Series
        A sequence of measurements
    :return: float
    """
    # entropy of x given y
    y_counter = Counter(y)
    xy_counter = Counter(list(zip(x,y)))
    total_occurrences = sum(y_counter.values())
    entropy = 0.0
    for xy in xy_counter.keys():
        p_xy = xy_counter[xy] / total_occurrences
        p_y = y_counter[xy[1]] / total_occurrences
        entropy += p_xy * math.log(p_y/p_xy)
    return entropy

# ---------------------------------------------------------------------------------------------------------------------------------

def theils_u(x, y):
    s_xy = conditional_entropy(x,y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n/total_occurrences, x_counter.values()))
    s_x = entropy(p_x)
    if s_x == 0:
        return(1)
    else:
        return((s_x - s_xy)/s_x)
        
# ---------------------------------------------------------------------------------------------------------------------------------

def boxplot_and_histogram_plot(ser):
    """
    Cut the figure window in 2 halves. For the numerical series creates a boxplot and histogram.
    
    Parameters:
    Pandas series having numerical values
    
    Returns:
    Boxplot and histogram
    """
   
    fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})

    # Add a graph in each part
    sns.boxplot(ser, ax = ax_box)
    sns.distplot(ser, ax = ax_hist)

    # Remove x axis name for the boxplot
    ax_box.set(xlabel='');
    sns.despine(ax=ax_hist);
    sns.despine(ax=ax_box, left=True);
    
# ---------------------------------------------------------------------------------------------------------------------------------

def detect_outliers(df, n, features):
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than n outliers according
    to the Tukey method.
    Courtesy: https://www.kaggle.com/abisheksudarshan/titanic-top-10-solution
    
    Parameters:
    df - pandas dataframe
    n - number of occurences of outliers in a single row
    features - column names
    
    
    Returns:
    list of indices having outliers
    
    """
    
    outlier_indices = []
    
    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col],75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        
        # outlier step
        outlier_step = 1.5 * IQR
        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
        
    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    return multiple_outliers 

# ---------------------------------------------------------------------------------------------------------------------------------






