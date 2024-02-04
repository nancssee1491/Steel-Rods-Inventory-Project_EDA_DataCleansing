# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 20:42:37 2023

@author: Nancssee
"""


'''Exploratory Data Analysis (EDA)'''

# Import libraries:
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import pylab

# Read the dataset:
steel = pd.read_excel("C:/Users/hanna/OneDrive/Desktop/Nancssee/Courses/Diploma In Practical Data Analytics/Project 1-Inventory of Steel Rods/Project Templates-Project 1/Project_28_Dataset .xlsx", sheet_name="TMT_Data")
steel.info()
steel_statistics = steel.describe()

'''
Since business objectives are to minimize underutilization of offcuts steel rods & 
to maximize profitability, which means to solve the cutting stock problem, therefore
features selected here are to provide information about characteristics which affecting 
the cutting pattern, as well as information about customer orders. Hence, features 
selected are:
1) Dia (different thickness may need different cutting way)	
2) Dia group (different thickness may need different cutting way)
3) Grade (higher quality level of steel rods probably harder to cut)	
4) Quantity (order quantity represents quantity to cut)
5) Rate (selected as cost may affect order quantity)
'''   


'''1) Business decision moments of 'Dia':'''
# 1st:
steel.Dia.mode()

# Count & frequency (in %) for 'Dia':
Dia_count = steel.Dia.value_counts()
Dia_frequency = steel.Dia.value_counts(normalize=True) * 100


'''2) Business decision moments of 'Dia group':'''
# 1st:
steel['Dia group'].mode()

# Count & frequency (in %) for 'Dia':
Dia_group_count = steel['Dia group'].value_counts()
Dia_group_frequency = steel['Dia group'].value_counts(normalize=True) * 100


'''3) Business decision moments of 'Grade':'''
# 1st:
steel.Grade.mode()

# Count & frequency (in %) for 'Grade':
Grade_count = steel.Grade.value_counts()
Grade_frequency = steel.Grade.value_counts(normalize=True) * 100

'''Bar charts for 'Dia', 'Dia group', 'Grade' (non-numeric):'''
import sweetviz as sv
s = sv.analyze(steel)
s.show_html()

'''
Inferences for 'Dia', 'Dia group', 'Grade' features:
1) The most popular types of TMT rods which have received the maximum numbers of 
   orders are: 8mm in thickness, with assorted set of thickness in 12mm - 32mm, 
   and 500D in grade.
2) Except for thickness ('Dia' feature), the difference in popularity among the 
   most preferred and less preferred options in terms of quality level ('Grade')
   and assorted set of different thickness ('Dia group') are more than & about 
   50% respectively. 
''' 


'''4) Business decision moments of 'Quantity':'''
# 1st:
steel.Quantity.mean()
steel.Quantity.mode()
steel.Quantity.median()

# 2nd: 
steel.Quantity.var()
steel.Quantity.std()
Qty_range = max(steel.Quantity) - min(steel.Quantity)
Qty_range

# 3rd:
steel.Quantity.skew()

# 4th:
steel.Quantity.kurt()

'''Graphs of 'Quantity':'''
# 1) Boxplot:
sns.boxplot(x=steel.Quantity)
plt.xlabel('Quantity in metric tonne')
plt.title('Quantity')

# 2) Histogram:
plt.hist(steel.Quantity, edgecolor='black')
plt.xlabel('Quantity in metric tonne')
plt.ylabel('Count')
plt.title('Quantity')   

# 3) Density Plot: 
sns.kdeplot(steel.Quantity, fill = True)
plt.xlabel('Quantity in metric tonne')
plt.title('Quantity')   

# 4) Q-Q Plot: 
stats.probplot(steel.Quantity, dist = "norm", plot = pylab)
plt.ylabel('Quantity in metric tonne')
plt.title('Quantity')   
'''
Theoretical quantiles to test the normality is not showing up on the x-axis as 
expected, potentially due to missing values.
'''
steel.isna().sum()  # Missing values exist.


'''5) Business decision moments of 'Rate':'''
# 1st:
steel.Rate.mean()
steel.Rate.mode()
steel.Rate.median()

# 2nd: 
steel.Rate.var()
steel.Rate.std()
Rate_range = max(steel.Rate) - min(steel.Rate)
Rate_range

# 3rd:
steel.Rate.skew()

# 4th:
steel.Rate.kurt()

'''Graphs of 'Rate':'''
# 1) Boxplot:
sns.boxplot(x=steel.Rate)
plt.xlabel('Cost per metric tonne')
plt.title('Rate')

# 2) Histogram:
plt.hist(steel.Rate, edgecolor='black')
plt.xlabel('Cost per metric tonne')
plt.ylabel('Count')
plt.title('Rate')   

# 3) Density Plot: 
sns.kdeplot(steel.Rate, bw=0.5, fill = True)
plt.xlabel('Cost per metric tonne')
plt.title('Rate')   

# 4) Q-Q Plot: 
stats.probplot(steel.Rate, dist = "norm", plot = pylab)
plt.ylabel('Cost per metric tonne')
plt.title('Rate') 
'''
Theoretical quantiles for data normality test is not showing up on the x-axis as 
expected, potentially due to missing values. 
'''
steel.isna().sum()  # Missing values exist.

'''
Inferences for 'Quantity' & 'Rate' features:
1) Both features are having outliers, but there are more outliers in 'Quantity'.
2) Both are right skewed, which means high frequencies at lower order quantity 
   side with cheaper charging cost for TMT rods are more common. 
3) As a larger standard deviation indicates greater variability or dispersion of 
   data points from the mean, therefore the costs of TMT rods per metric tonne 
   are greatly deviated from its mean compared to the order quantity in metric 
   tonne, as standard deviation of 'Rate’ and 'Quantity' are 9641.89 & 6.67 
   respectively.
''' 

'''Relationship between 'Rate' & 'Quantity' with scatter plot:'''
plt.scatter(x = steel.Rate, y = steel.Quantity) 
plt.xlabel('Cost per metric tonne')
plt.ylabel('Quantity in metric tonne')
plt.title('Cost vs Order Quantity') 

'''
From the scatter plot, a big tight cluster & outliers both exist.
1) Cluster: Tight cluster suggests there is a strong some kind of positive 
            relationship between cost per metric tonne and quantity ordered in 
            metric tonne, which means cheaper cost is resulting in a more orders.

2) Outliers: Suggests an existence of outliers in dataset.
'''

'''
EDA Conclusion: As there are outliers, potential missing values, data distribution 
                skewness, non-numeric categorical and unnecessary features present 
                in the dataset, a subsequential data cleaning & organizing 
                process is needed.
'''                
                