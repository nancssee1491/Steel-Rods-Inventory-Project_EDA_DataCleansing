# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 20:40:11 2023

@author: Nancssee
"""


'''Data Cleaning/Preparation/Munging/Wrangling/Data Organizing'''

# Import libraries:
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import pylab

# Read the dataset:
steel = pd.read_excel("C:/Users/hanna/OneDrive/Desktop/Nancssee/Courses/Diploma In Practical Data Analytics/Project 1-Inventory of Steel Rods/Project Templates-Project 1/Project_28_Dataset .xlsx", sheet_name="TMT_Data")
steel.info()
steel_statistics = steel.describe()
steel.dtypes

# Check the dataset:
steel.isna().sum()
duplicate = steel.duplicated()  
sum(duplicate)
len(steel)


'''1) Typecasting: Is not needed in this case, as selected non-numeric features
                   are ordinal, hence label encoding which converts non-numeric 
                   to numeric data while preserving their orders will be perform
                   later.'''


'''2) Missing values imputation:'''
# Missing values check:
steel.isna().sum()
'''Missing values exist, imputation is needed.'''

'''
Inferences:
From the previous EDA process,    
1) 'Quantity' & 'Rate' features are strongly correlated, indicated by the 
   scatter plot of "Cost vs Order Quantity".
2) Both 'Quantity' & 'Rate' features have outliers & are positively skewed.
3) Hence, median imputation is used for both, as median is a more representative 
   measure of central tendency than the mean in this case due to the skewness.
'''

from sklearn.impute import SimpleImputer 

# Missing values median imputation for 'Quantity':
median_imputer = SimpleImputer(missing_values = np.nan, strategy = 'median')
steel["Quantity"] = pd.DataFrame(median_imputer.fit_transform(steel[["Quantity"]]))
steel["Quantity"].isna().sum()  

# Missing values median imputation for 'Rate':
median_imputer = SimpleImputer(missing_values = np.nan, strategy = 'median')
steel["Rate"] = pd.DataFrame(median_imputer.fit_transform(steel[["Rate"]]))
steel["Rate"].isna().sum()

# Re-check imputed dataset:
steel.isna().sum()
duplicate = steel.duplicated()  
sum(duplicate)
len(steel)

'''Conclusion: All missing values in the dataset have been imputed.'''


'''3) Zero variance and near zero variance features'''
# Identify numeric, non-numeric & date columns, then check zero variance for all:
numeric_columns = steel.select_dtypes(include=['number']).columns
categorical_columns = steel.select_dtypes(include=['object']).columns
date_columns = steel.select_dtypes(include=['datetime']).columns

numeric_zero_var_columns = steel[numeric_columns].columns[steel[numeric_columns].var() == 0]
numeric_zero_var_columns

categorical_zero_var_columns = steel[categorical_columns].columns[steel[categorical_columns].nunique() == 1]
categorical_zero_var_columns

date_zero_var_columns = steel[date_columns].columns[steel[date_columns].nunique() == 1]
date_zero_var_columns

'''Conclusion: No zero variance column in the dataset, hence all selected features 
               can be included in data preparation process.'''


'''4) Label encoding: As slected non-numeric features ('Dia', 'Dia group', 'Grade') 
                      in the dataset are ordinal, therefore label encoding is used 
                      for non-numeric to numeric features conversion.'''   

from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()
steel['Dia'] = labelencoder.fit_transform(steel['Dia'])
steel['Dia group'] = labelencoder.fit_transform(steel['Dia group'])
steel['Grade'] = labelencoder.fit_transform(steel['Grade'])
steel.dtypes

'''Conclusion: 'Dia', 'Dia group', 'Grade' non-numeric ordinal features in the 
               dataset have been encoded into numeric with order preservation.'''

# Check encoded dataset:
steel.isna().sum()
duplicate = steel.duplicated()  
sum(duplicate)
len(steel)


'''5) Discretization: Not needed in this case, as discrete data is not needed.''' 


'''6) Transformation:'''
# Check if numeric features of 'Quantity' & 'Rate' in the encoded dataset are  
# normally distributed with Q-Q plots:
    
stats.probplot(steel['Quantity'], dist = "norm", plot = pylab)
plt.ylabel('Quantity in metric tonne')
plt.title('Quantity') 

stats.probplot(steel['Rate'], dist = "norm", plot = pylab)
plt.ylabel('Cost per metric tonne')
plt.title('Rate')     
    
'''
Inferences:
1) 'Quantity' is deviated from the theoretical quantiles (the red diagonal line), 
   indicating the presence of outliers, as well as skewness and non-normal distribution.
2) 'Rate' follows the theoretical quantiles.   
3) Hence, transformation is only needed for the 'Quantity' numeric feature, to 
   make it more normally distributed.
4) For 'Quantity', Yeo-Johnson transformation is used as it has negative values.   
'''    
    
from feature_engine import transformation
tf = transformation.YeoJohnsonTransformer(variables = 'Quantity')
steel_trans = tf.fit_transform(steel)

# Re-check normality of 'Quantity' with Q-Q plots:
stats.probplot(steel_trans['Quantity'], dist = "norm", plot = pylab)
plt.ylabel('Quantity in metric tonne')
plt.title('Quantity_transformed') 

# Check transformed dataset:
steel_trans.isna().sum()
duplicate = steel_trans.duplicated()  
sum(duplicate)
len(steel_trans)

'''Conclusion: 'Quantity' is now transformed and follows the theoretical 
               quantiles closer than it previously did.'''


'''7) Feature scaling/feature shrinking:'''
# Plot boxplot to check outliers of transformed dataset:
# 1) 'Quantity_transformed' boxplot:    

sns.boxplot(x=steel_trans.Quantity)
plt.xlabel('Quantity in metric tonne')
plt.title('Quantity_transformed')

# 2) 'Rate' boxplot:    
sns.boxplot(x=steel_trans.Rate)
plt.xlabel('Cost per metric tonne')
plt.title('Rate')   

'''
Inferences:
1) In the transformed dataset, outliers present in both 'Quantity' and 'Rate'.
2) Hence, robust scaling is used for feature scaling in this case, as it scale 
   features using statistics that are robust to outliers.
'''   

from sklearn.preprocessing import RobustScaler

# Select & scale selected columns, then covert outcome into DataFrame, then join
# scaled columns with unscaled columns, then check summary statistics:
columns_to_scale = ['Quantity','Rate']
selected_columns = steel_trans[columns_to_scale]

robust = RobustScaler()
scaled_columns = robust.fit_transform(selected_columns)
scaled_columns_df = pd.DataFrame(scaled_columns, columns=columns_to_scale)

unscaled_columns = steel_trans.drop(columns=columns_to_scale)
steel_trans_robust = pd.concat([unscaled_columns, scaled_columns_df], axis=1)
steel_trans_robust_s = steel_trans_robust.describe()

# Check scaled dataset:
steel_trans_robust.isna().sum()
duplicate = steel_trans_robust.duplicated()  
sum(duplicate)
len(steel_trans_robust)

'''Conclusion: Transformed dataset is now robustly scaled.'''


'''8) Outlier analysis/treatment:'''
# Plot boxplot to check outliers of scaled dataset:
# 1) 'Quantity' boxplot:
sns.boxplot(x=steel_trans_robust.Quantity)
plt.xlabel('Quantity in metric tonne')
plt.title('Quantity_transformed_robusted')

# 2) 'Rate' boxplot:    
sns.boxplot(x=steel_trans_robust.Rate)
plt.xlabel('Cost per metric tonne')
plt.title('Rate_robusted')

'''
Inferences: 
1) From boxplots, outliers are still presenting in 'Quantity' & 'Rate' features.
2) Refer to the original dataset 'steel', the reason for negative order quantities 
   is unknown and those extreme positive values in order quantity & 
   cost per metric tonne could be valid.
3) Therefore, in order to mitigate the impact of outliers without a complete 
   removal of it, IQR-based winsorization is used.
'''

from feature_engine.outliers import Winsorizer

# Define & select features, then initiate winsorizer & perform winsorization, 
# then plot boxplots to re-check winsorization outcomes:
columns_to_winsorize = ['Quantity', 'Rate']

winsor_iqr = Winsorizer(capping_method='iqr', 
                        tail='both', fold=1.5, 
                        variables=columns_to_winsorize)

steel_trans_robust_win = winsor_iqr.fit_transform(steel_trans_robust)

# Boxplots to re-check outliers of winsorized dataset:
# 1) 'Quantity' boxplot:
sns.boxplot(x=steel_trans_robust_win.Quantity)
plt.xlabel('Quantity in metric tonne')
plt.title('Quantity_transformed_robusted_win')

# 2) 'Rate' boxplot:    
sns.boxplot(x=steel_trans_robust_win.Rate)
plt.xlabel('Cost per metric tonne')
plt.title('Rate_robusted_win')

# Check winsorized dataset:
steel_trans_robust_win.isna().sum()
duplicate = steel_trans_robust_win.duplicated()  
sum(duplicate)
len(steel_trans_robust_win) 

'''Conclusion: All outliers in the transformed & scaled dataset have been treated.'''


'''9) Select required features:'''
steel_select = steel_trans_robust_win[['Dia', 'Dia group', 'Grade', 'Quantity', 'Rate']]

# Check selected features:
steel_select.isna().sum()
duplicate = steel_select.duplicated()  
sum(duplicate)
len(steel_select) 


'''10) Handling duplicates in columns:'''
# Check correlation among selected features:
steel_select_corr = steel_select.corr()

'''Conclusion: As all |r| are between -0.85 and 0.85, means all columns are not 
               highly similar. Therefore, all selected features can be included 
               in subsequential data pre-processing steps.'''


'''11) Handling duplicates in rows:'''
# Check row duplications in selected features: 
duplicate = steel_select.duplicated()  
sum(duplicate)
'''Duplicates exist, removal is needed.'''

# Remove duplicates and re-check cleaned dataset:
steel_clean = steel_select.drop_duplicates()

duplicate_check = steel_clean.duplicated()  
sum(duplicate_check)
len(steel_clean)
steel_clean.isna().sum()


'''Conclusion: All duplicated rows were removed from transformed, scaled and 
               outliers treated selected features.'''
               
'''
Data Cleansing/Preparation/Munging/Wrangling/Data Organizing process is completed!
'''

'''EDA after Data Cleaning/Preparation/Munging/Wrangling/Data Organizing:'''
'''1) Business decision moments of 'Quantity':'''
# 1st:
steel_clean.Quantity.mean()
steel_clean.Quantity.mode()
steel_clean.Quantity.median()

# 2nd: 
steel_clean.Quantity.var()
steel_clean.Quantity.std()
Qty_final_range = max(steel_clean.Quantity) - min(steel_clean.Quantity)
Qty_final_range

# 3rd:
steel_clean.Quantity.skew()

# 4th:
steel_clean.Quantity.kurt()

'''Graphs of 'Quantity':'''
# 1) Boxplot:
sns.boxplot(x=steel_clean.Quantity)
plt.xlabel('Quantity in metric tonne')
plt.title('Quantity_clean')

# 2) Histogram:
plt.hist(steel_clean.Quantity, edgecolor='black')
plt.xlabel('Quantity in metric tonne')
plt.ylabel('Count')
plt.title('Quantity_clean')   

# 3) Density Plot: 
sns.kdeplot(steel_clean.Quantity, fill = True)
plt.xlabel('Quantity in metric tonne')
plt.title('Quantity_clean')   

# 4) Q-Q Plot: 
stats.probplot(steel_clean.Quantity, dist = "norm", plot = pylab)
plt.ylabel('Quantity in metric tonne')
plt.title('Quantity_clean')   

'''2) Business decision moments of 'Rate':'''
# 1st:
steel_clean.Rate.mean()
steel_clean.Rate.mode()
steel_clean.Rate.median()

# 2nd: 
steel_clean.Rate.var()
steel_clean.Rate.std()
Rate_final_range = max(steel_clean.Rate) - min(steel_clean.Rate)
Rate_final_range

# 3rd:
steel_clean.Rate.skew()

# 4th:
steel_clean.Rate.kurt()

'''Graphs of 'Rate':'''
# 1) Boxplot:
sns.boxplot(x=steel_clean.Rate)
plt.xlabel('Cost per metric tonne')
plt.title('Rate_clean')

# 2) Histogram:
plt.hist(steel_clean.Rate, edgecolor='black')
plt.xlabel('Cost per metric tonne')
plt.ylabel('Count')
plt.title('Rate_clean')   

# 3) Density Plot: 
sns.kdeplot(steel_clean.Rate, bw=0.5, fill = True)
plt.xlabel('Cost per metric tonne')
plt.title('Rate_clean')   

# 4) Q-Q Plot: 
stats.probplot(steel_clean.Rate, dist = "norm", plot = pylab)
plt.ylabel('Cost per metric tonne')
plt.title('Rate_clean')   

steel_clean.isna().sum()  # Zero missing values.

'''Relationship between processed 'Rate' & 'Quantity':'''
plt.scatter(x = steel_clean.Rate, y = steel_clean.Quantity) 
plt.xlabel('Cost per metric tonne')
plt.ylabel('Quantity in metric tonne')
plt.title('Cost vs Order Quantity_clean') 

# SweetViz auto EDA for other features in the processed dataset:
import sweetviz as sv
s_processed = sv.analyze(steel_clean)
s_processed.show_html()


'''
Inferences: 
1) From boxplots, after data pre-processing, there is no visible outliers in 'Rate', 
   but there are visible outliers in 'Quantity’ which is much fewer, which 
   potentially due to an excessive presence of outliers in the raw 'Quantity’ data.
  
2) For ‘Quantity’, based on histograms and density plots, after data pre-processing, 
   there is a significant improvement in data distribution skewness 
   (much fewer extreme data points), which is confirmed by  the much smaller 
   skewness reading (3rd moment). For ‘Rate’ feature, there is a slightly 
   improvement in the same.

3) For 'Quantity’, based on Q-Q plots, after data pre-processing there is a 
   significant improvement in data distribution normality 
   (data points follow theoretical quantiles closer in Q-Q plot), which is 
   confirmed by the closer to zero kurtosis reading (4th moment). For ‘Rate’ 
   feature, as indicated by the reduced kurtosis reading after data pre-processing, 
   there are lesser extreme values and flatter peak than the normal distribution.
  
4) In conclusion, after pre-processed, data are less skewed and are closer to 
   normal distribution. 
            
Conclusion: Data preparation process done, the final completed dataset 
            "steel_clean" is saved.
'''

# Save final completed processed DataFrame to a CSV file with UTF-8 encoding:
steel_clean.to_csv('TMT_Data_processed.csv', index=False, encoding='utf-8')

# Searching & accessing the saved file:
import os
current_directory = os.getcwd()
print("Current Directory:", current_directory)
    