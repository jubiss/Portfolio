import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

def dados_ini(removed=False):
    df = pd.read_csv('valuation_data.csv')
    df.drop('Unnamed: 0',axis=1,inplace=True)
    geo_dados = pd.read_csv('Geo dados total.csv')
    geo_dados.drop('Unnamed: 0',axis=1,inplace=True)

    df = df[df['latitude'] != 'None']
    df[['latitude','longitude']] = np.round(df[['latitude','longitude']].astype(float),2)
    
    geo_dados[['latitude','longitude']] = np.round(geo_dados[['latitude','longitude']].astype(float),2)
    geo_dados.drop_duplicates(subset=['latitude','longitude'],inplace=True)
    new_df = pd.merge(df,geo_dados, how='left',left_on=['latitude','longitude'],right_on=['latitude','longitude'])
    
    new_df.rename(columns={'1':'bairro'},inplace=True)
    new_df['0'].fillna('error',inplace=True)
    new_df_fora_de_sp = new_df[~new_df['0'].str.contains('São Paulo')]
    new_df = new_df[new_df['0'].str.contains('São Paulo')]
    new_df.drop(['0','altitude','2'],axis=1,inplace=True)
    if removed==True:
        return new_df, new_df_fora_de_sp
    return new_df

def find_outliers_turkey(x):
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    iqr = q3-q1
    floor = q1 - 1.5*iqr
    celing = q3 +1.5*iqr
    outlier_indicies = list(x.index[(x<floor)|(x>celing)])

    return outlier_indicies#, outlier_values

def data_clean(new_df,clean_mode=0):
    """
    Clean modes
    0 Main version used, transform outliers Apartament in cobertura and remove cobertura outliers
    1 Version with price outliers
    2 version removing preco_m2 outliers
    """
    new_df = new_df.drop(new_df[((new_df['rooms'] == -1) & (new_df['useful_area']== -1)) & (new_df['garages']== -1)].index)
    new_df = new_df[new_df['building_type']!='Conjunto']
    new_df = new_df[new_df['point_estimate'] != -1]

    #Imoveis restantes sem quartos. Informação encontrada no google
    new_df.loc[21690,'rooms'] = 1
    new_df.loc[11193,'rooms'] = 2
    new_df.loc[3756,'rooms'] = 3
    new_df['garages'].replace(np.nan,0,inplace=True) ##Melhor performace
    
    if clean_mode==1:
        new_df = pd.get_dummies(new_df,columns=['building_type'])
        return new_df
    if clean_mode==2:
        new_df['preco_m2'] = new_df['point_estimate']/new_df['useful_area']
        outliers_ap_p = find_outliers_turkey(new_df[new_df['building_type']=='Apartamento']['preco_m2']) #Transform outliers de 
        new_df.loc[outliers_ap_p,'building_type'] = 'Cobertura'
#        outlier_m2 = new_df.loc[find_outliers_turkey(new_df[new_df['building_type']=='Cobertura']['preco_m2'])]
        new_df = new_df.drop(find_outliers_turkey(new_df[new_df['building_type']=='Cobertura']['preco_m2']))
#        new_df = new_df.drop(find_outliers_turkey(new_df['preco_m2']))
        new_df.drop('preco_m2',axis=1,inplace=True)
        new_df = pd.get_dummies(new_df,columns=['building_type'])
        return new_df# , outlier_m2

#REMOÇÃO DE OUTLIERS    
    max_com_outlier = max(new_df['point_estimate'])
    outliers_ap_p = find_outliers_turkey(new_df[new_df['building_type']=='Apartamento']['point_estimate']) #Transform outliers de 
    new_df.loc[outliers_ap_p,'building_type'] = 'Cobertura'
#Remoção outliers de preço
  #  new_df = new_df.drop(find_outliers_turkey(new_df[new_df['building_type']=='Apartamento']['point_estimate'])) #Não melhora a performace do modelo
    outlier = new_df.loc[find_outliers_turkey(new_df[new_df['building_type']=='Cobertura']['point_estimate'])]
    new_df = new_df.drop(find_outliers_turkey(new_df[new_df['building_type']=='Cobertura']['point_estimate']))
    #new_df = new_df.drop(find_outliers_turkey(new_df['point_estimate']))
    max_sem_outlier = max(new_df['point_estimate'])
    new_df = pd.get_dummies(new_df,columns=['building_type'])
    return new_df, outlier

def remove_diff_columns(train,test):
    train_column = train.columns.values.tolist()
    test_column = test.columns.values.tolist()
    
    not_in_train = [i for i in test_column if i not in train_column]
    not_in_test  = [i for i in train_column if i not in test_column]    

    train.drop(not_in_test,axis=1,inplace=True)
    test.drop(not_in_train,axis=1,inplace=True)
    
    return train,test

def bairro(X_train,X_test,y_train,train_rf=False):
    X_train['preco_m2'] = y_train['point_estimate']/X_train['useful_area']
    bairro = X_train.groupby('bairro').mean()
    bairro['bairro faixa'] = pd.qcut(bairro['preco_m2'].sort_values(),10,range(10))
    bairro_dic = bairro['bairro faixa'].to_dict()
    X_train['bairro_faixa'] = X_train['bairro'].map(bairro_dic)
    X_test['bairro_faixa'] = X_test['bairro'].map(bairro_dic)
    X_train['bairro_merged'] = X_train['bairro'].mask(X_train['bairro'].map(X_train['bairro'].value_counts(normalize=True)) < 0.01, 'Other')
    bairro_merged = pd.Series(X_train['bairro_merged'].values,index=X_train['bairro']).to_dict()
    X_test['bairro_merged'] = X_test['bairro'].map(bairro_merged)
    X_train.drop(['preco_m2','bairro'],axis=1,inplace=True)
    X_test.drop(['bairro'],axis=1,inplace=True)
    if train_rf == True:
        X_test['bairro_faixa'].fillna(5,inplace=True)
    X_train = pd.get_dummies(X_train,columns=['bairro_merged'])
    X_test = pd.get_dummies(X_test,columns=['bairro_merged'])
    X_train, X_test = remove_diff_columns(X_train,X_test)
    return X_train,X_test

def model_validation(X,y,space_xgb,space_random,train_rf=False,plot = True,com_localizacao=True):
    from skopt import dummy_minimize
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.model_selection import cross_val_score, KFold
    import xgboost as xgb
    from sklearn.ensemble import RandomForestRegressor
    n_calls_hyp = 5
    squared_outer_point_results = []

    percen_outer_point_results = []
    percen_outer_maximum_results = []
    percen_outer_minimum_results = []

    squared_percen_point_results = []
    squared_percen_maximum_results = []
    squared_percen_minimum_results = []

    absolut_point_results = []

    squared_outer_mean_results = []
    percen_outer_mean_results = []

    pd.DataFrame(columns=['RMSPE RF point','MAPE RF point',
                      'RMSE XGB point','MAE XGB point','RMSEPE XGB point','MAPE XGB point',
                      'RMSEPE XGB max','MAPE XGB max','RMSEPE XGB min','MAPE XGB min',
                      'RMSE XGB Mean','MAPE XGB Mean'
                      ])


    
    cv_outer = KFold(n_splits=4,shuffle=True)
    param_results = []
    if train_rf==True:
        rmspe_rf = []
        mape_rf = []
        squared_rf = []
        absolut_rf = []
        
    for train_ix, test_ix in cv_outer.split(X):
        X_train, X_test = X.iloc[train_ix, :], X.iloc[test_ix,:]
        y_train, y_test = y.iloc[train_ix] , y.iloc[test_ix]
        if com_localizacao ==True:
            X_train, X_test = bairro(X_train,X_test,y_train,train_rf=train_rf)
        else:
            X_train.drop('bairro',axis=1,inplace=True)
            X_test.drop('bairro',axis=1,inplace=True)
        cv_inner = KFold(n_splits=2, shuffle=True)        
        def treina_xgb(params):
            learning_rate = params[0]
            n_estimators = params[1]
            max_depth = params[2]
            min_child_weight = params[3]
            gamma = params[4]    
            subsample = params[5]
            colsample_bytree = params[6]
            model = xgb.XGBRegressor(learning_rate=learning_rate,n_estimators=n_estimators,max_depth=max_depth,
                                      min_child_weight=min_child_weight, gamma=gamma, subsample=subsample,colsample_bytree=colsample_bytree)
            return -np.mean(cross_val_score(model,X_train,y_train['point_estimate'],cv=cv_inner,scoring="neg_mean_squared_error"))#mean_squared_error(y_test, p)
        
        resultado_xgb = dummy_minimize(treina_xgb,space_xgb,n_calls=n_calls_hyp,verbose=1)
        param_xgb = resultado_xgb.x
    
    
        dic = {0:'minimum_estimate',1:'point_estimate',2:'maximum_estimate'}
        dic_pred = {0:'minimum_estimate_pred',1:'point_estimate_pred',2:'maximum_estimate_pred'}
        pred = pd.DataFrame(y_test)
        param_results.append(param_xgb)
        xgb_reg = xgb.XGBRegressor(learning_rate=param_xgb[0],n_estimators=param_xgb[1],max_depth=param_xgb[2],
                                      min_child_weight=param_xgb[3], gamma=param_xgb[4], 
                                      subsample=param_xgb[5],colsample_bytree=param_xgb[6])
        medium_pred = []
        for i in range(3):
            xgbclf = xgb_reg.fit(X_train,y_train[dic.get(i)])
            xgb_pred = xgbclf.predict(X_test)
            medium_pred.append(xgb_pred)
            pred[dic_pred.get(i)] = xgbclf.predict(X_test)
    
            if i == 0:
                percen_outer_minimum_results.append(np.mean(np.abs(y_test[dic.get(i)]-xgb_pred)/y_test[dic.get(i)]))
                squared_percen_minimum_results.append((np.mean(((y_test[dic.get(i)]- xgb_pred)/y_test[dic.get(i)])**2))**0.5)

            if i == 1:
                def random_forest_on(rmspe_rf,mape_rf):
                    def treina_random_forest(params):
                        max_depth = params[0]
                        min_samples_split = params[1]
                        min_samples_leaf = params[2]
                        model = RandomForestRegressor(n_estimators=1500,max_depth=max_depth,
                                                  min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
                        return -np.mean(cross_val_score(model,X_train,y_train['point_estimate'],cv=cv_inner,scoring="neg_mean_squared_error"))#mean_squared_error(y_test, p)
                
                    resultado_random_forest = dummy_minimize(treina_random_forest,space_xgb,n_calls=n_calls_hyp,verbose=1)
                    param_random = resultado_random_forest.x
                    random_reg= RandomForestRegressor(n_estimators=1500,max_depth=param_random[0],
                                                  min_samples_split=param_random[1], min_samples_leaf=param_random[2])
        
                    randreg = random_reg.fit(X_train,y_train['point_estimate'])
                    rand_pred = randreg.predict(X_test)
                    rmspe_rf.append((np.mean(((y_test[dic.get(i)]- rand_pred)/y_test[dic.get(i)])**2))**0.5)
                    mape_rf.append(np.mean(np.abs(y_test[dic.get(i)]-rand_pred)/y_test[dic.get(i)]))
                    squared_rf.append(mean_squared_error(y_test[dic.get(i)], rand_pred)**0.5)
                    absolut_rf.append(mean_absolute_error(y_test[dic.get(i)],rand_pred))
                    return(rmspe_rf,mape_rf)
                if train_rf==True: 
                    rmspe_rf,mape_rf = random_forest_on(rmspe_rf,mape_rf)

                squared_outer_point_results.append(mean_squared_error(y_test[dic.get(i)], xgb_pred)**0.5)
                squared_percen_point_results.append((np.mean(((y_test[dic.get(i)]- xgb_pred)/y_test[dic.get(i)])**2))**0.5)
                percen_outer_point_results.append(np.mean(np.abs(y_test[dic.get(i)]-xgb_pred)/y_test[dic.get(i)]))
                absolut_point_results.append(mean_absolute_error(y_test[dic.get(i)],xgb_pred))
                
                if plot== True:
                    ax = sns.histplot((pred['point_estimate']-pred['point_estimate_pred'])/pred['point_estimate'])
                    ax.set(xlabel='Absolute percentage error',ylabel='Frequência')
                    plt.savefig('hist erro percentual absoluto.png')
                    plot = False
            if i == 2:
                squared_percen_maximum_results.append((np.mean(((y_test[dic.get(i)]- xgb_pred)/y_test[dic.get(i)])**2))**0.5)
                percen_outer_maximum_results.append(np.mean(np.abs(y_test[dic.get(i)]-xgb_pred)/y_test[dic.get(i)]))
    
        squared_outer_mean_results.append(mean_squared_error(y_test['point_estimate'], (medium_pred[0]+medium_pred[1]+medium_pred[2])/3)**0.5)
        percen_outer_mean_results.append(np.mean(np.abs(y_test['point_estimate']-(medium_pred[0]+medium_pred[1]+medium_pred[2])/3)/y_test['point_estimate']))
    print('rolou')

    if train_rf == True:
        results_dic = {'RMSPE RF point':rmspe_rf,'MAPE RF point':mape_rf, 'MAE RF':absolut_rf,'RMSE RF':squared_rf,
                          'RMSE XGB point':squared_outer_point_results,'MAE XGB point':absolut_point_results
                          ,'RMSEPE XGB point':squared_percen_point_results,'MAPE XGB point':percen_outer_point_results,
                          'RMSEPE XGB max':squared_percen_maximum_results,'MAPE XGB max':percen_outer_maximum_results
                          ,'RMSEPE XGB min':squared_percen_minimum_results,'MAPE XGB min':percen_outer_minimum_results,
                          'RMSE XGB Mean':squared_outer_mean_results,'MAPE XGB Mean':percen_outer_mean_results,
                          'XGB_Params':param_results
                          }


    else:
        results_dic = {'RMSE XGB point':squared_outer_point_results,'MAE XGB point':absolut_point_results,
                        'RMSEPE XGB point':squared_percen_point_results,'MAPE XGB point':percen_outer_point_results,
                         'RMSEPE XGB max':squared_percen_maximum_results,'MAPE XGB max':percen_outer_maximum_results
                          ,'RMSEPE XGB min':squared_percen_minimum_results,'MAPE XGB min':percen_outer_minimum_results,
                          'RMSE XGB Mean':squared_outer_mean_results,'MAPE XGB Mean':percen_outer_mean_results,
                          'XGB_Params':param_results
                          }

    return pd.DataFrame.from_dict(results_dic)

def size_efect(X,y,space_xgb):
    from skopt import dummy_minimize
    from sklearn.model_selection import cross_val_score, KFold
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    squared_percen_point_results = []
    percen_outer_point_results = []
    size_squared = []
    size_percent = []
    squared_error = []
    abs_error = []
    size_abs = []
    size_squa = []
    n_calls_hyp = 5
    size = [0.025,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
    for i in size:
        for j in range(3):
            X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=i)
            X_train, X_test = bairro(X_train,X_test,y_train,train_rf=False)
            y_train, y_test = y_train['point_estimate'], y_test['point_estimate']

            cv_inner = KFold(n_splits=2, shuffle=True)        
            def treina_xgb(params):
                learning_rate = params[0]
                n_estimators = params[1]
                max_depth = params[2]
                min_child_weight = params[3]
                gamma = params[4]    
                subsample = params[5]
                colsample_bytree = params[6]
                model = xgb.XGBRegressor(learning_rate=learning_rate,n_estimators=n_estimators,max_depth=max_depth,
                                          min_child_weight=min_child_weight, gamma=gamma, subsample=subsample,colsample_bytree=colsample_bytree)
                return -np.mean(cross_val_score(model,X_train,y_train,cv=cv_inner,scoring="neg_mean_squared_error"))#mean_squared_error(y_test, p)
            
            resultado_xgb = dummy_minimize(treina_xgb,space_xgb,n_calls=n_calls_hyp,verbose=1)
            param_xgb = resultado_xgb.x
        
            xgb_reg = xgb.XGBRegressor(learning_rate=param_xgb[0],n_estimators=param_xgb[1],max_depth=param_xgb[2],
                                          min_child_weight=param_xgb[3], gamma=param_xgb[4], 
                                          subsample=param_xgb[5],colsample_bytree=param_xgb[6])
            medium_pred = []
            xgbclf = xgb_reg.fit(X_train,y_train)
            xgb_pred = xgbclf.predict(X_test)
            medium_pred.append(xgb_pred)
            squared_percen_point_results.append((np.mean(((y_test- xgb_pred)/y_test)**2))**0.5)
            percen_outer_point_results.append(np.mean(np.abs(y_test-xgb_pred)/y_test))
            squared_error.append(mean_squared_error(y_test, xgb_pred)**0.5)
            abs_error.append(mean_absolute_error(y_test, xgb_pred))
    
 
        size_squared.append(np.mean(squared_percen_point_results))
        size_percent.append(np.mean(percen_outer_point_results))
        size_abs.append(np.mean(abs_error))
        size_squa.append(np.mean(squared_error))

    results_dic = {'RMSPE XGB point':size_squared,'MAPE XGB point':size_percent,
                   'ABS XGB':size_abs,'RMSQ':size_squa,'Train size':size}
    return pd.DataFrame(results_dic)

def area_limitada(new_df,space_xgb,get_bairro=0):
    """
    Bairro indica a forma de lidar com a variável de localização
    bairro = 0, usa a variável de localização sem alteração
    bairro = 1, usa a variável de localização removendo parte dos valóres de localização
    bairro = 2, não usa a variável de localização.
    """
    from skopt import dummy_minimize
    from sklearn.model_selection import cross_val_score, KFold
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import xgboost as xgb
    n_calls_hyp = 5
    squared_percen_point_results = []
    percen_outer_point_results = []
    squared_error = []
    abs_error = []
    X = new_df
    
    train_limit = X[((X['latitude']>-23.5884) & (X['latitude']<-23.5495) & (X['longitude']>-46.6817) & 
           (X['longitude']<-46.6379))]
    test_limit = X[~((X['latitude']>-23.5884) & (X['latitude']<-23.5495) & (X['longitude']>-46.6817) & 
           (X['longitude']<-46.6379))]
    X_train =train_limit.drop(['id','address','tower_name','point_estimate','maximum_estimate','minimum_estimate','latitude','longitude'],axis=1)
    X_test =test_limit.drop(['id','address','tower_name','point_estimate','maximum_estimate','minimum_estimate','latitude','longitude'],axis=1)
    
    y_train = train_limit[['point_estimate','maximum_estimate']]
    y_test = test_limit[['point_estimate','maximum_estimate']]
    if get_bairro == 0:
        X_train, X_test = bairro(X_train,X_test,y_train,train_rf=False)
    if get_bairro == 1:
        X_train.loc[X_train['bairro'].sample(int(np.round(len(X_train)*0.1))).index,'bairro'] = np.NaN
        X_train, X_test = bairro(X_train,X_test,y_train,train_rf=False)
    if get_bairro == 2:
        X_train.drop('bairro',axis=1,inplace=True)
        X_test.drop('bairro',axis=1,inplace=True)
    y_train, y_test = y_train['point_estimate'], y_test['point_estimate']

    cv_inner = KFold(n_splits=2, shuffle=True)        
    
    
    def treina_xgb_2(params):
        learning_rate = params[0]
        n_estimators = params[1]
        max_depth = params[2]
        min_child_weight = params[3]
        gamma = params[4]    
        subsample = params[5]
        colsample_bytree = params[6]
        model = xgb.XGBRegressor(learning_rate=learning_rate,n_estimators=n_estimators,max_depth=max_depth,
                                  min_child_weight=min_child_weight, gamma=gamma, subsample=subsample,colsample_bytree=colsample_bytree)
        return -np.mean(cross_val_score(model,X_train,y_train,cv=cv_inner,scoring="neg_mean_squared_error"))#mean_squared_error(y_test, p)
    resultado_xgb = dummy_minimize(treina_xgb_2,space_xgb,n_calls=n_calls_hyp,verbose=1)
    param_xgb = resultado_xgb.x
        

    xgb_reg = xgb.XGBRegressor(learning_rate=param_xgb[0],n_estimators=param_xgb[1],max_depth=param_xgb[2],
                                  min_child_weight=param_xgb[3], gamma=param_xgb[4], 
                                  subsample=param_xgb[5],colsample_bytree=param_xgb[6])
    
    xgbclf = xgb_reg.fit(X_train,y_train)
    xgb_pred = xgbclf.predict(X_test)
    
    squared_percen_point_results.append((np.mean(((y_test- xgb_pred)/y_test)**2))**0.5)
    percen_outer_point_results.append(np.mean(np.abs(y_test-xgb_pred)/y_test))
    squared_error.append(mean_squared_error(y_test, xgb_pred)**0.5)
    abs_error.append(mean_absolute_error(y_test, xgb_pred))
    results_dic = {'RMSPE':squared_percen_point_results,'MAPE':percen_outer_point_results,
                   'MAE':abs_error,'RMSQ':squared_error}
    return pd.DataFrame(results_dic)
    
#Feature Importance
def gain_fscore(X,y,space_xgb,space_random,obtain_param=False):
    import xgboost as xgb
    if obtain_param==True:
        results = model_validation(X,y,space_xgb,space_random,train_rf=False)
        param_xgb = results[results['MAPE XGB point']==max(results['MAPE XGB point'])]['XGB_Params'].values[0]
    param_xgb = [0.0055808474002494845,1864,6,5.132344780533354,7,0.6568520426055856,0.9194430021839984]
    xgb_reg = xgb.XGBRegressor(learning_rate=param_xgb[0],n_estimators=param_xgb[1],max_depth=param_xgb[2],
                                  min_child_weight=param_xgb[3], gamma=param_xgb[4], 
                                  subsample=param_xgb[5],colsample_bytree=param_xgb[6])
    X['preco_m2'] = y['point_estimate']/X['useful_area']
    bairro = X.groupby('bairro').mean()
    bairro['bairro faixa'] = pd.qcut(bairro['preco_m2'].sort_values(),10,range(10))
    bairro_dic = bairro['bairro faixa'].to_dict()
    X['bairro'] = X['bairro'].mask(X['bairro'].map(X['bairro'].value_counts(normalize=True)) < 0.01, 'Other')
    X['bairro_faixa'] = X['bairro'].map(bairro_dic)
    X.drop(['preco_m2'],axis=1,inplace=True)
    X = pd.get_dummies(X,columns=['bairro'])
    xgbclf = xgb_reg.fit(X,y['point_estimate'])
    
    fscore =  pd.Series(xgbclf.get_booster().get_fscore())
    fscore_top5 = pd.Series(fscore/fscore.sum()).sort_values(ascending=False).head(5).to_dict()
    gain = pd.Series(xgbclf.get_booster().get_score(importance_type='gain'))
    gain_top5 = pd.Series(gain/gain.sum()).sort_values(ascending=False).head(5).to_dict()
    return gain_top5,fscore_top5