import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
path='/Users/Shared/
df=pd.read_csv(path+'Loan Book.csv')
df.describe()
sddata=df.loc[df['Trade Type']=='Standard']
sddata.describe()
#adding features
epd=pd.to_datetime(sddata['Expected Payment Date']).dt.date
ad=pd.to_datetime(sddata['Advance Date']).dt.date 
ed=(epd-ad)/np.timedelta64(1,'D')
sddata['Expected Duration']=ed
sddata.head()
#fit model to predit
from sklearn.model_selection import train_test_split
X=Datafa[['Expected Duration','nPST']].values
y=Datafa['In Arrears'].values
from sklearn.linear_model import LogisticRegression
classifier1=LogisticRegression(C=100, solver='liblinear',random_state=1)
classifier1.fit(X_train,y_train)
y_pred=classifier1.predict(X_test)
#Evaluate the model
from sklearn import metrics
cf=metrics.confusion_matrix(y_test, y_pred)
cf_f1=metrics.f1_score(y_test,y_pred,average=None)
cf_precision=metrics.precision_score(y_test,y_pred,average=None)
cf_recall=metrics.recall_score(y_test,y_pred,average=None)
cf,cf_f1,cf_precision,cf_recall
