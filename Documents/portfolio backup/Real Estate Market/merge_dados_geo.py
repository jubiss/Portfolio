import pandas as pd

df0 = pd.read_csv('Dados geográficos parciais_0-3252.csv')
df1 = pd.read_csv('Dados geográficos parciais 3250-7000.csv')
df2 = pd.read_csv('Dados geográficos parciais 7000-9700.csv')
df3 = pd.read_csv('Dados geográficos parciais 9700-16895.csv')

merged_df = pd.concat([df0,df1,df2,df3]).drop('Unnamed: 0',axis=1)

merge_drop_dupli = merged_df.drop_duplicates()
merge_drop_dupli.to_csv('Geo dados total.csv')