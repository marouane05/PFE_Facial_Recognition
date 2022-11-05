import pandas as pd
import uuid
import random

#id = uuid.uuid4()
data = {
    'id': [0,uuid.uuid4().int,uuid.uuid4().int,uuid.uuid4().int],
    'Name': ['uknown','Marouane Dryouch', 'Obama Barack', 'Karima Student'],
    'Infos': ['','Etudiant master big data s4','professeur en informatique','etudiant en SMA s3']
}
"""
data ={
    'id':[uuid.uuid4()],
    'Name':['ahmed ahel'],
    'Infos':['chercheur en mathématique']
}
"""
# Make data frame of above data
df = pd.DataFrame(data)

# append data frame to CSV file
#df.to_csv('./data/persons.csv', mode='a',header=False,index=False)
new_df = pd.read_csv('./data/persons.csv')

print(new_df.to_string())
df_res = new_df.loc[new_df['id']==int('8905')];
arr = df_res.to_numpy()[0]
#strr=arr['Name']
print(arr)
