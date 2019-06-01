#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# 班級:財金三乙
# 姓名:林晏靚
# 學號:A105040092
# 
# # A.regression
# # from sklearn.preprocessing import PolynomialFeatures
# # PolynomialFeatures(degree=3, include_bias=True)
# # 40462.09922679004
# 
# # B.Classification
# # bag_clf=BaggingClassifier(DecisionTreeClassifier(),n_estimators=500,max_samples=100,bootstrap=True,n_jobs=-1)
# # 0.8931686046511628

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


housing = pd.read_csv("housing.csv")


# In[3]:


housing


# In[4]:


housing.info()


# In[5]:


housing[housing.isnull().any(axis=1)]


# In[6]:


housing = housing.drop("total_bedrooms", axis=1)


# In[7]:


housing.head()


# In[8]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()


# In[9]:


housing_cat = housing["ocean_proximity"]


# In[10]:


housing_cat_encoded = encoder.fit_transform(housing_cat)


# In[11]:



housing_cat_encoded


# In[12]:


type(housing_cat_encoded)


# In[13]:


housing_cat_encoded.shape


# In[14]:


pd.DataFrame(housing_cat_encoded)


# In[15]:


housing_cat_encoded.reshape(-1, 1)


# In[16]:


housing_cat_encoded.reshape(-1, 1).shape


# In[17]:


pd.DataFrame(housing_cat_encoded.reshape(-1, 1))


# In[18]:


from sklearn.preprocessing import OneHotEncoder


# In[19]:


encoder = OneHotEncoder(categories='auto')
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))


# In[20]:



housing_cat_1hot


# In[21]:


housing_cat_1hot.toarray()


# In[22]:


pd.DataFrame(housing_cat_1hot.toarray())


# In[23]:


housing.head()


# In[24]:


pd.DataFrame(housing_cat_1hot.toarray()).iloc[:, 1:].head()


# In[25]:


housing_final = pd.concat([housing, pd.DataFrame(housing_cat_1hot.toarray()).iloc[:, 1:]], axis=1)


# In[26]:


housing_final.head()


# In[27]:


X = housing_final.drop("median_house_value", axis=1).drop("ocean_proximity", axis=1)


# In[28]:


X.head()


# In[29]:


y = housing_final[["median_house_value"]]
y.head()


# In[30]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[31]:



from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)


# In[41]:


lin_reg.predict(X_train)


# In[42]:



from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(X_train)
lin_mse = mean_squared_error(y_train, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)

from sklearn.metrics import mean_absolute_error
lin_mae = mean_absolute_error(y_train, housing_predictions)
print(lin_mae)


# In[43]:



from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(X_test)
lin_mse = mean_squared_error(y_test, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)

from sklearn.metrics import mean_absolute_error
lin_mae = mean_absolute_error(y_test, housing_predictions)
print(lin_mae)


# In[44]:


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)


# In[45]:


lin_reg.intercept_


# In[46]:


lin_reg.coef_


# In[47]:


lin_reg.predict(X_test)


# In[92]:



from sklearn.preprocessing import PolynomialFeatures


# In[93]:


poly_features = PolynomialFeatures(degree=3, include_bias=True)


# In[94]:


X_poly = poly_features.fit_transform(X_train)


# In[95]:


X_poly.shape


# In[96]:


lin_reg = LinearRegression()
lin_reg.fit(X_poly, y_train)


# In[97]:


housing_predictions = lin_reg.predict(X_poly)


# In[98]:


housing_predictions.shape


# In[99]:


y_train.shape


# In[100]:


from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(X_poly)
lin_mse = mean_squared_error(y_train, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)

from sklearn.metrics import mean_absolute_error
lin_mae = mean_absolute_error(y_train, housing_predictions)
print(lin_mae)


# In[101]:


X_test.shape


# In[102]:


X_poly_test = poly_features.fit_transform(X_test)


# In[103]:


X_poly_test.shape


# In[104]:


from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(X_poly_test)
lin_mse = mean_squared_error(y_test, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)

from sklearn.metrics import mean_absolute_error
lin_mae = mean_absolute_error(y_test, housing_predictions)
print(lin_mae)


# In[ ]:





# In[32]:


from sklearn.linear_model import ElasticNet


# In[91]:


elastic_net=ElasticNet(alpha=0.1,l1_ratio=0.5)


# In[34]:


elastic_net.fit(X_train,y_train)


# In[35]:


elastic_net.predict(X_test)


# In[36]:


from sklearn.metrics import mean_squared_error

housing_predictions = elastic_net.predict(X_test)
ela_mse = mean_squared_error(y_test, housing_predictions)
ela_rmse = np.sqrt(ela_mse)
print(ela_rmse)

from sklearn.metrics import mean_absolute_error
ela_mae = mean_absolute_error(y_test, housing_predictions)
print(ela_mae)


# In[75]:


from sklearn.tree import DecisionTreeRegressor


# In[76]:


tree_reg=DecisionTreeRegressor(max_depth=2)


# In[77]:


tree_reg.fit(X_train,y_train)


# In[81]:


from sklearn.linear_model import Lasso


# 

# In[86]:


lasso_reg=Lasso(alpha=1)


# In[87]:


lasso_reg.fit(X_train,y_train)


# In[88]:


lasso_reg.predict(X_test)


# In[89]:


from sklearn.metrics import mean_squared_error

housing_predictions = lasso_reg.predict(X_test)
las_mse = mean_squared_error(y_test, housing_predictions)
las_rmse = np.sqrt(las_mse)
print(las_rmse)

from sklearn.metrics import mean_absolute_error
las_mae = mean_absolute_error(y_test, housing_predictions)
print(las_mae)


# In[ ]:





# In[ ]:





# In[ ]:





# 分類

# In[58]:


import copy


# In[59]:


housing_classify = copy.deepcopy(housing_final)
housing_classify.head(10)


# In[60]:


housing_classify["median_house_value"].median()


# In[61]:


housing_classify['midian_classify'] = housing_classify["median_house_value"]> housing_classify["median_house_value"].median()
housing_classify['midian_classify'].head(10)


# In[62]:


housing_classify['midian_classify'] = housing_classify['midian_classify'].astype(int)
housing_classify['midian_classify'].head(10)


# In[63]:


X = housing_classify.drop("median_house_value", axis=1).drop("ocean_proximity", axis=1).drop("midian_classify", axis=1)


# In[64]:


X.head()


# In[65]:


y = housing_classify[["midian_classify"]]
y.head()


# In[66]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[67]:


from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)


# In[68]:


# train performance
y_pred = log_reg.predict(X_train)
n_correct = sum(y_pred == y_train.as_matrix().ravel())
print(n_correct / len(y_pred))
# test performance
y_pred = log_reg.predict(X_test)
n_correct = sum(y_pred == y_test.as_matrix().ravel())
print(n_correct / len(y_pred))


# In[69]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


# In[70]:


from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
polynomial_svm_clf = Pipeline([
 ("poly_features", PolynomialFeatures(degree=3)),
 ("scaler", StandardScaler()),
 ("svm_clf", LinearSVC(C=10, loss="hinge"))
 ])
polynomial_svm_clf.fit(X, y)


# In[71]:


# train performance
y_pred = polynomial_svm_clf.predict(X_train)
n_correct = sum(y_pred == y_train.as_matrix().ravel())
print(n_correct / len(y_pred))
# test performance
y_pred = polynomial_svm_clf.predict(X_test)
n_correct = sum(y_pred == y_test.as_matrix().ravel())
print(n_correct / len(y_pred))


# In[72]:


from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier


# In[73]:


bag_clf=BaggingClassifier(DecisionTreeClassifier(),n_estimators=500,max_samples=100,bootstrap=True,n_jobs=-1)
bag_clf.fit(X_train,y_train)
y_pred=bag_clf.predict(X_train)


# In[74]:


bag_clf=BaggingClassifier(DecisionTreeClassifier(),n_estimators=500,oob_score=True,bootstrap=True,n_jobs=-1)


# In[75]:


bag_clf.fit(X_train,y_train)


# In[76]:


bag_clf.oob_score_


# In[77]:


from sklearn.metrics import accuracy_score


# In[78]:


y_pred=bag_clf.predict(X_test)


# In[79]:


accuracy_score(y_test,y_pred)


# In[ ]:




