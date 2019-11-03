import pandas as pd
import os
import pickle
os.system("D:/R/R-3.6.1/bin/Rscript.exe C:/Users/ADMIN/Desktop/Gender_Identification_by_voice/ExtractFeatures.R") 
#To make it work on another system the user must change the working directory to current directory in Extract_feature.r file
#Also the path to Rscript.exe has to be changed.

test_df = pd.read_csv('myvoice2.csv')
new_data=test_df[["meanfreq","sd","median","Q25","Q75","IQR","skew","kurt","sp.ent","sfm","mode","centroid","meanfun","minfun","maxfun","meandom","mindom","maxdom","dfrange","modindx"]]


X=new_data.iloc[:,:].values
model=pickle.load(open('voice_model.pickle','rb'))
if model.predict(X)[0]==0:
    print("Its male voice")
else:
    print("Its female voice")

