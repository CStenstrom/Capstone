import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import cross_val_score, RepeatedKFold, KFold
from scipy.stats import chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn import metrics



def freq_table(column1, column2):
    '''
    Input: two columns
    Output: frequency counts between the two columns'''
    tab = pd.crosstab(index=column1,
                     columns=column2)
    
    print(tab)
    
    return


def chi_test(counts):
    '''
    calculates the pvalue, chi statistic, and whether the variables are independent or dependent
    measuring on a significance level of 5% 
    
    INPUT: the associated frequency count variables in an array of arrays
    OUTPUT: Pvalue, Chi Statistic, Variable dependency / independency
            '''
    stat, p, dof, expected = chi2_contingency(counts, correction=False)
    print('P-value is: ' + str(p))
    print('Chi Statistic is: ' + str(stat))
    if p <= 0.05:
        print('There is an association between variables')
    else:
        print('No association')
        
    return stat


def cramerStat(data, stat):
    n = np.sum(data)
    md = min(data.shape) - 1

    V = np.sqrt((stat/n) / md)

    print(V)
    
    if V <= 0.2:
        print('Weak')
    elif 0.2 < V <= 0.6:
        print('Moderate')
    elif V > 0.6:
        print('Strong')
    
    return


def calculate_vif(x):
    vif = pd.DataFrame()
    vif['variables'] = x.columns
    vif['VIF'] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
    
    return vif

def heatmap_data(data, method):
    """
    Takes data passed and determines coorelation, creates mask,
    and shows resulting plot
    """

    corr = data.corr(method=method)
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style('white'):
        f, ax = plt.subplots(figsize=(15,15))
        ax = sns.heatmap(corr, mask=mask, annot=True, lw=1, linecolor='white', cmap='viridis')
        plt.title('Correlation')
        plt.xticks(rotation=60)
        plt.yticks(rotation=60)
    plt.show()


def plot_confusion(y_test, y_pred):
    sns.set(color_codes=True)
    plt.figure(1, figsize=(9,6))
    
    sns.set(font_scale=1.4)
    data = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(data, columns=np.unique(y_test), index=np.unique(y_test))
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    
    sns.heatmap(df_cm, annot=True, cmap='Blues', fmt='d')
    
    return

def auc_chart(X_test, y_test, model):
    
    y_pred_proba = model.predict_proba(X_test)[::,1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba, pos_label=1)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label='data 1, auc=' + str(auc))
    plt.legend(loc=4)
    plt.show()
    
    return

import math

def phi_stat(d,c,b,a):
    top = (d*a)-(b*c)
    bottom = math.sqrt((d+b)*(d+c)*(a+b)*(a+c))
    p = top / bottom
    
    print(p)
    return


def get_scores(model, X, y):
    
    p = cross_val_score(model, X, y, cv=3, scoring='precision')
    print(p)
    print('Average Precision Score: ' + str(p.mean()))
    print()
    
    r = cross_val_score(model, X, y, cv=3, scoring='recall')
    print(r)
    print('Average Recall Score: ' + str(r.mean()))
    print()
    
    f = cross_val_score(model, X, y, cv=3, scoring='f1')
    print(f)
    print('Average F1 Score: ' + str(f.mean()))
    
    return

from mlxtend.evaluate import paired_ttest_5x2cv

def pairTest(A, B, X, y, score):
    t, p = paired_ttest_5x2cv(estimator1=A, estimator2=B, X=X, y=y, scoring=score, random_seed=1)

    print('P-value: %.3f, t-Statistic: %.3f' % (p, t))

    if p <= 0.05:
        print('Difference between mean performance is real')
    else:
        print('Algorithms probably have the same performance')