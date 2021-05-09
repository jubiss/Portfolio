import sys
#Dataset location and functions.
sys.path.append('/home/jubi/Documents/Porfolio/Real Estate Market')

import functions as func

new_df,sem_sp = func.dados_ini(removed=True)

new_df,outlier = func.data_clean(new_df,clean_mode=0)

X = new_df.drop(['id','address','tower_name','point_estimate','maximum_estimate','minimum_estimate','latitude','longitude'],axis=1)

y = new_df[['minimum_estimate','point_estimate','maximum_estimate']]

space_random = [(100,2000), #n_estimators
         (1,100), #max_depth
         (2,120), #min_sample_splits
         (1,10), #min_samples_leaf
         ]
#from sklearn.
space_xgb = [(1e-3, 1e-1, 'log-uniform'), # learning rate
          (100, 2000), # n_estimators
          (1, 100), # max_depth 
          (1, 6.), # min_child_weight 
          (0, 15), # gamma 
          (0.5, 1.), # subsample 
          (0.5, 1.)] # colsample_bytree 


results = func.model_validation(X,y,space_xgb,space_random,train_rf=True)
results.to_csv('Resultado validação modelo llimpo.csv')
    
results_sem_loc = func.model_validation(X,y,space_xgb,space_random,train_rf=True,com_localizacao=False)
results_sem_loc.to_csv('Resultado validação modelo limpo sem loc.csv')

df_with_out = func.dados_ini(removed=False)
df_with_out = func.data_clean(df_with_out,clean_mode=1)
X_out = df_with_out.drop(['id','address','tower_name','point_estimate','maximum_estimate','minimum_estimate','latitude','longitude'],axis=1)
y_out = df_with_out[['minimum_estimate','point_estimate','maximum_estimate']]

new_df_ap = func.dados_ini(removed=False)
new_df_ap = func.data_clean(new_df_ap,clean_mode=2)

X_ap = new_df_ap.drop(['id','address','tower_name','point_estimate','maximum_estimate','minimum_estimate','latitude','longitude'],axis=1)
y_ap = new_df_ap[['minimum_estimate','point_estimate','maximum_estimate']]

results_ap = func.model_validation(X_ap,y_ap,space_xgb,space_random,train_rf=False)

results_out = func.model_validation(X_out,y_out,space_xgb,space_random,train_rf=True)
results_out.to_csv('Resultado validação modelo com outlier.csv')

results_sem_loc = func.model_validation(X,y,space_xgb,space_random,train_rf=True,com_localizacao=False)
results_sem_loc.to_csv('Resultado validação modelo limpo sem loc_ap.csv')

results_size = func.size_efect(X,y,space_xgb)
results_size.to_csv('Resultado tamanho da amostra_ap.csv')

loc_parcial  = func.area_limitada(new_df,space_xgb,1)
loc_parcial.to_csv('bound loc parcial_ap.csv')
com_loc = result_area_limit = func.area_limitada(new_df,space_xgb)
com_loc.to_csv('bound loc total_ap.csv')
sem_loc = result_area_limit = func.area_limitada(new_df,space_xgb,2)
sem_loc.to_csv('bound sem localizacao_ap.csv')
