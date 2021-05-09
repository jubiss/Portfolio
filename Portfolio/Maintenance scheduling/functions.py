"""LIMPEZA DE DADOS"""
def data_clean(df,test,na_threshold=101):
    """Limpeza de dados 
    Realiza o processo de limpeza de dados do dataset de treino e de teste.
    
    Parametero
    ----------
    df : DataFrame
        Datframe com os dados de treino
    test : DataFrame
        Datframe com os dados de teste
    na_threshold : float, opcional
        Remove colunas de ambos os dataframes, que possuam % de nan values menor que na_threshold.
        na_threshold=101, nenhuma coluna é removida.

    Retorna
    -------
    Dataframe, Dataframe
        Retorna dataframe de treino e de teste limpos
    """
    import numpy as np
    
    #Transforma missing values 'na' em np.nan
    df = df.replace('na',np.nan) 
    test = test.replace('na',np.nan) 
    #Transforma o tipo de todos os valores das colunas para float
    col = df.drop('class',axis=1).columns
    df[col] = df[col].astype(float)
    test[col] = test[col].astype(float)
    #Label encoder
    df['class'] = df['class'].apply(encoder)
    test['class'] = test['class'].apply(encoder)
    #Remove colunas com %NaN maior que na_threshold
    df,test = MissingPercent(df, test, na_threshold=na_threshold,missing=False) #Resolver problema
    #Remove colunas que aparecam em apenas um dos dataframes.
    df,test = DiferenteCol(df, test) #Resolver problema
    
    return df,test
 
def encoder(classification):
    """Limpeza de dados 
    Transforma a variável categórica em numérica (Encoder)
    Positivo =1, Negativo = 0

    Parametero
    ----------
    classification : Str
        Label

    Retorna
    -------
    1 para casos positivos
    0 para casos negativos
    """

    if classification == 'pos':
        return 1
    return 0

def DiferenteCol(df,test,retorna_diff = False):
    """Limpeza de dados 
    Checa a existência de colunas diferentes em train e test.

    Parametero
    ----------
    df : DataFrame
        Datframe com os dados de treino
    test : DataFrame
        Datframe com os dados de teste
    retorna_diff : bool, opcional
        True: Retorna lista de colunas diferentes entre os datasets.
        False: Retorna ambos os datasets sem colunas diferentes

    Retorna
    -------
    if retorna_diff == False
    Dataframe, Dataframe
        Retorna dataframe de treino e de teste sem colunas diferentes
    if retorna_diff == True
    List, List
        Retorna listas de colunas diferentes entre ambos os datasets.
    """

    #Checa se existem colunas diferentes no data set de treino e de teste
    train_column = df.columns.values.tolist()
    test_column = test.columns.values.tolist()
    not_in_train = [i for i in test_column if i not in train_column]
    not_in_test  = [i for i in train_column if i not in test_column]
    if retorna_diff == False:
        if (len(not_in_train)==0) and (len(not_in_test)==0):
            print('Não existem colunas diferentes entre o data set de treino e de teste.')
            return df,test
        else:
            df = df.drop(not_in_test,axis=1)
            test = test.drop(not_in_train,axis=1)
            if len(not_in_test) !=0:
                print('Colunas que estão apenas no treino'+ str(not_in_test))
            if len(not_in_train) !=0:
                print('Colunas que estão apenas no teste'+ str(not_in_train))
            return df,test
    return not_in_train,not_in_test

def MissingPercent(df,test,na_threshold,missing=True):
    """Limpeza de dados 
    Checa a porcentagem de missing values de cada coluna. Se a percentagem de uma coluna for maior
    do que na_threshold, a coluna é removida. 

    Parametero
    ----------
    df : DataFrame
        Datframe com os dados de treino
    test : DataFrame
        Datframe com os dados de teste
    missing : bool, opcional
        True: Retorna DataFrame com porcentagem missing values.
        False: Retorna df e test, sem colunas com percentagem de missing values>na_threshold

    Retorna
    -------
    if missing == False
    Dataframe, Dataframe
        Retorna DataFrame com porcentagem missing values.
    if missing == True
    DataFrame, DataFrame
        Retorna df e test, sem colunas com percentagem de missing values>na_threshold
    """

    import pandas as pd

    train_missing = pd.DataFrame({'column':df.columns,'percent missing': df.isnull().sum() * 100 / len(df)})
    test_missing = pd.DataFrame({'column':test.columns,'percent missing': test.isnull().sum() * 100 / len(df)})
    new_train = df.drop(train_missing[train_missing['percent missing']>na_threshold].index,axis=1)
    new_test = test.drop(train_missing[train_missing['percent missing']>na_threshold].index,axis=1)
    if missing ==False:
        return new_train, new_test
    return train_missing ,test_missing

"""LIMPEZA DE DADOS"""

"""MODELO"""
def find_optimal_threshold(train,rept=30,plot=True):
    """
    Retorna valor otimo de threshold obtido a partir do dataset de treino. Utiliza amostras diferentes para 
    encontrar esse valor otimo.

    Parametero
    ----------
    train : DataFrame
        Datframe com os dados de treino
    rept : Integer
        Numero de amostras utilizadas para encontrar o threshold otimo. Quanto maior o numero de amostras melhor
        o resultado.
    plot : bool, opcional
        True: Plota curva custo x threshold otida a partir das amostras.

    Retorna
    -------
    Threshold otimo, obtido a partir do dataset de treino.
    """

    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import xgboost as xgb
    import pandas as pd
    from sklearn.metrics import confusion_matrix
    import numpy as np

    num_round = 10
    threshold = np.linspace(0,0.45,400)
    for j in range(rept):
        print(j)
        custo_thres = []
        X_train, X_test, y_train, y_test = train_test_split(train.drop('class',axis=1), train['class'], 
                                                            test_size=0.4)
        scale_pos_weight = sum(y_train==0)/sum(y_train==1)
        param = {'silent':1,
    'min_child_weight':1, ## unbalanced dataset
    'objective':'binary:logistic',
    'eval_metric':'auc','scale_pos_weight':scale_pos_weight}                              
        dtrain = xgb.DMatrix(X_train,label = y_train)
        dtest  = xgb.DMatrix(X_test, label = y_test)
        xgb_2 = xgb.train(param,dtrain,num_round)
        for i in threshold:
            prediction_2 = (xgb_2.predict(dtest)+i).round().astype(int)
            fp = confusion_matrix(y_test,prediction_2)[0,1] #500
            fn = confusion_matrix(y_test, prediction_2)[1,0] #100
            custo_thres.append(fp*10+fn*500)
        if j == 0:
            all_cost_thres = [custo_thres]
        else:
            all_cost_thres.append(custo_thres)
    all_cost_df = pd.DataFrame(all_cost_thres,columns=threshold)
    mean_values = all_cost_df.mean()
    opt_theshold = mean_values[mean_values==min(mean_values)].index[0]
    if plot ==True:
        mean_values.plot.line()
        plt.xlabel('Threshold')
        plt.ylabel('$')
        plt.title('Threshold x Custo (Treino)')
        plt.show()
    return opt_theshold


def modelo_xgboost(train,test,threshold=0,return_model = False):
    """
    Implementacao do modelo para realizar predicoes sobre o custo. O threshold pode ser float ou numpy array.

    Parametero
    ----------
    train : DataFrame
        Datframe com os dados de treino
    test : DataFrame
        Datframe com os dados de teste
    threshold : float ou numpy array, opcional
        Float: Retorna um valor de custo utilizando threshold.
        Array: Retorna um vetor de custos de tamanho igual ao do threshold

    Retorna
    -------
    if return_model == False
    float or array
        Retorna o custo gerado pelo modelo.
    if return_model == True
    float or array, xgboost
        Retorna o custo gerado pelo modelo; Retorna o modelo.
    """

    import xgboost as xgb
    from sklearn.metrics import confusion_matrix
    import numpy as np
        
    num_round=10
    custo_vec = []
    X_train,y_train=train.drop('class',axis=1), train['class']
    X_test, y_test = test.drop('class',axis=1), test['class']
    scale_pos_weight = sum(y_train==0)/sum(y_train==1)
    dtrain = xgb.DMatrix(X_train,label = y_train)
    dtest  = xgb.DMatrix(X_test, label = y_test)
    param = {'silent':1,
         'min_child_weight':1, ## unbalanced dataset
         'objective':'binary:logistic',
         'eval_metric':'auc', 
         'scale_pos_weight':scale_pos_weight}
    xgb_2 = xgb.train(param,dtrain,num_round)

    if type(threshold) == np.ndarray :
        for i in threshold:
            prediction_2 = (xgb_2.predict(dtest)+i).round().astype(int)
            fp = confusion_matrix(y_test,prediction_2)[0,1] #500
            fn = confusion_matrix(y_test, prediction_2)[1,0] #100
            custo_vec.append(fp*10+fn*500) #custo_modelo_final 
        if return_model == True:
            return custo_vec, xgb_2
        return custo_vec
            
    prediction_2 = (xgb_2.predict(dtest)+threshold).round().astype(int)
    fp = confusion_matrix(y_test,prediction_2)[0,1] #500
    fn = confusion_matrix(y_test, prediction_2)[1,0] #100
    if return_model == True:    
        return  fp*10+fn*500, xgb_2 #custo_modelo_final 
    return fp*10+fn*500
"""MODELO"""


"""FUNCOES UTILIZADAS PARA PLOTS"""
def cost(X,Y):
    """ Calcula o custo para um numero X de falsos negativos e Y de falsos positivos.

    Parametero
    ----------
    X : int
        Numero de falsos negativos
    Y : int
        Numero de falsos positivos

    Retorna
    -------
    int
        Custo para X falsos negativos e Y falsos positivos
    """

    return (X*500+Y*10)

def predict_random(neg_p): 
    """
    Prediz aleatoriamoente uma observacao. 

    Parametero
    ----------
    neg_p : float
        Probabilidade de o caminhão não precisar de manutenção

    Retorna
    -------
    str
        Retorna neg se for previsto como negativo ou pos se for previsto como positivo
    """
    import numpy as np
    if round(np.random.random(),4) < neg_p:
        return 'neg'
    return 'pos'

def cost_random_model(true,predicted):
    """Retorna o custo se label e predição forem diferentes. (Random model)
    Parametero
    ----------
    true : str
        Label verdadeiro
    predicted: str
        Label previsto

    Retorna
    -------
    int
    500 se for falso negativo, 10 se for falso positivo
    """
    if true == 'pos':
        return 500
    return 10

def random_model():
    """
    Obtem o custo para uma escolha aleatoria de positivo, negativo.

    Retorna
    -------
    Custo medio para o caso de escolher aleatoriamente se é positivo ou negativo
    """

    import pandas as pd
    import numpy as np
    test = pd.read_csv('base_vigente_2020.csv')
    train = pd.read_csv('base_vigente_anos_anteriores.csv')
    
    #Random Model
    neg = train['class'].value_counts()[0]
    neg_p = neg/len(train['class'])
    custo = []
    for i in range(300):
        
        test['predict'] = predict_random(neg_p)
        
        diference = test[test['class']!=test['predict']][['class','predict']]
        
        diference['cost'] = np.vectorize(cost_random_model)(diference['class'],diference['predict'])
        
        custo.append(diference['cost'].sum())
        
    custo = np.array(custo)
    return np.mean(custo)
"""FUNCOES UTILIZADAS PARA PLOT"""
