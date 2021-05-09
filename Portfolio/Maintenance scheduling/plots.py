import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import functions as func
import numpy as np

plt.style.use("seaborn")
def Slide_5():    
    test = pd.read_csv('base_vigente_2020.csv')
    class_label = ['Negativo','Positivo']
    test['class'].value_counts().plot.pie(autopct='%1.2f%%', shadow=True, 
                                                fontsize=12, startangle=70,labels=class_label)
    plt.show()
    
    class_count = [test['class'].value_counts()[0],test['class'].value_counts()[1]]
    class_label = ['Negativo','Positivo']
    f , ax = plt.subplots(figsize=(8,6))
    sns.barplot(x=class_label,y=class_count)
    plt.ylabel('Quantidade')
    plt.show()

def Slide_6():
    import numpy as np
    costs = np.array([func.random_model(),func.cost(375,0),func.cost(0,15625),37000])
    label = ['Aleatória','Levar nenhum','Levar todos','Custo 2020']
    import seaborn as sns
    plt.title('Custos, estratégias simples e custo atual')
    plt.ylabel('$')
    sns.barplot(x=label,y=costs)
    plt.show()

def Slide_9():
    x = np.linspace(-5,5,50)
    y = 1/(1+np.exp(-x))
    x_1 = [0.5]*len(x)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(x,x_1)
    plt.plot(x,y)
    plt.show()

def Slide_12():
    costs = np.array([37000,17540,12130])
    label = ['Custo 2020','ML sem threshold','ML com threshold']
    import seaborn as sns
    plt.title('Custo atual e com Machine Learning')
    plt.ylabel('$')
    sns.barplot(x=label,y=costs)
    plt.show()

def Slide_13():
    econ_costs =(1- np.array([17540,12130])/37000)*(100)
    labe_econl = ['ML sem threshold','ML com threshold']
    plt.title('Redução de custos em relação ao custo de 2020')
    plt.ylabel('%')
    x1, x2 , y1, y2 = plt.axis()
    plt.axis((x1,x2,0,100))
    sns.barplot(x=labe_econl,y=econ_costs)
    plt.show()


# Slide 14, 15
def feature_importance_plot(xgb_2,df,plot_all_importace=False):
    """Realiza o plot de feature importance do modelo. O XGBoost permite a utilizacao de 5 metricas diferentes
    de feature importance.
    Weight: Representa o numero de vezes que uma feature apareceu em uma arvore.
    Gain: Considerada a mais importante das metricas, mostra a contribuicao de cada feature ao modelo.Total Gain/Cover
    Quanto maior o gain mais importante e a feature para gerar predicoes.
    Coverage: Numero de observacoes relacionadas a feature. Total Covier/ Weight
    Se plot_all_importance == False, plota as 10 features com maiores gain, e a distribuicao dessas 10 features,
    para o caso positivo e negativo.
    Parametero
    ----------
    xgb_2 : XGBoost
        Modelo utilizado para predicoes
    df : DataFrame
        Datframe com os dados de treino
    plot_all_importance: Bool
        True : Plota weight, gain, cover, total cover, total gatin para as 10 features com maior e menor gain.
        
    Retorna
    -------
    
    """

    
    importance = ['weight','gain','cover','total_gain','total_cover']
    feature_importance = [(xgb_2.get_score(importance_type=(i))) for i in importance]
    keys = list(feature_importance[0].keys())
    values = [list(feature_importance[i].values()) for i in range(5)] 
    importance_data = pd.DataFrame(data=values,columns=keys,index=importance).transpose() 
    importance_data['features'] = importance_data.index
    
    sns.barplot(y='features',x='gain',data=importance_data.sort_values('gain',ascending=False).head(10))
    plt.title('10 Features com maior Gain')
    plt.xscale('log')
    plt.xlabel('Log Gain')
    plt.show()
    if plot_all_importace==True:
        for i in importance:
            sns.barplot(y='features',x=i,data=importance_data.sort_values('gain',ascending=True).head(10))
            plt.title('10 features com menor Gain')
            plt.xlabel(i)
            plt.show()
            ###Features sem importancia e sua taxa de missing values
            
            sns.barplot(y='features',x=i,data=importance_data.sort_values('gain',ascending=False).head(10))
            plt.title('10 Features com maior Gain')
            plt.xlabel(i)
            plt.show()
        
    top_10_features = list(importance_data.sort_values('gain',ascending=False).head(5)['features'])
    for i in top_10_features:
        sns.displot(df[df['class']==1][i],kind='kde',bw_adjust=.25)
        plt.xlabel('Medida')
        plt.ylabel('Densidade')
        plt.title('Distribuição '+i+', positivo')
        plt.show()
        
        sns.displot(df[df['class']==0][i],kind='kde',bw_adjust=.25)
        plt.xlabel('Medida')
        plt.ylabel('Densidade')
        plt.title('Distribuição '+i+', negativo')
        plt.show()
        

#Slide 16
def feature_without_importance(xgb_2,df,test,plot_features=False):
    """Realiza o plot de features não utilizadas pelo modelo. 
    Se plot_features == False, plota grafico com todas as features nao utilizadas e percentagem de nan values 
    associados.
    Parametero
    ----------
    xgb_2 : XGBoost
        Modelo utilizado para predicoes
    df : DataFrame
        Datframe com os dados de treino
    test: Dataframe
        Dataframe com os dados de teste
    plot_features: Bool
        Plota distribuicao de features que nao foram utilizadas
    Retorna
    -------
    
    """
    import functions as func
    importance = ['weight','gain','cover','total_gain','total_cover']
    feature_importance = [(xgb_2.get_score(importance_type=(i))) for i in importance]
    keys = list(feature_importance[0].keys())
    values = [list(feature_importance[i].values()) for i in range(5)]
    
    train_missing,test_missing = func.MissingPercent(df,test,na_threshold=101)
    importance_data_without_transpose = pd.DataFrame(data=values,columns=keys,index=importance)
    not_in_df, features_not_used = func.DiferenteCol(df,importance_data_without_transpose,retorna_diff=True)
    features_without_importace_missing_percentage =train_missing.transpose()[features_not_used].transpose().drop('class').sort_values('percent missing',ascending=False)
    miss_feat = list(features_without_importace_missing_percentage['column'])
    miss_value = list(features_without_importace_missing_percentage['percent missing'])
    fig , ax = plt.subplots(figsize=(34,6))
    plt.xlabel('Features')
    plt.ylabel('Porcentagem de valores NaN')
    plt.title('Features com zero de importância (Excluidas pelo modelo).')
    sns.barplot(x=miss_feat,y=miss_value,ax=ax)
    plt.savefig('Missing Features')
    if plot_features==True:
        features_importance_no_importance = ['as_000','at_000','au_000','bv_000','cq_000']
        for i in features_importance_no_importance:
            sns.displot(df[df['class']==1][i],kind='kde',bw_adjust=.25)
            plt.xlabel('Medida')
            plt.ylabel('Densidade')
            plt.title('Distribuição '+i+', positivo')
            plt.show()
            
            sns.displot(df[df['class']==0][i],kind='kde',bw_adjust=.25)
            plt.xlabel('Medida')
            plt.ylabel('Densidade')
            plt.title('Distribuição '+i+', negativo')
            plt.show()

#Heatmaps não utilizados

def heatmaps():
    X,Y= np.linspace(0,375) , np.linspace(0,15625)
    
    X,Y = np.meshgrid(X, Y)
    import matplotlib.pyplot as plt
    plt.style.use("seaborn")
    #from matplotlib.pyplot import colorbar, pcolor, show
    Z = func.cost(X, Y)
    f , ax = plt.subplots(figsize=(8,6))
    plot = ax.pcolor(Y, X, Z,cmap='bwr')
    
    plt.xlabel('Falso Positivo')
    plt.ylabel('Falso negativo')
    plt.title('Heatmap de custo, relação falso negativo x falso positivo.')
    f.colorbar(plot,ax=ax)
    plt.plot([15625, 0], [0,312.5],color='brown')
    
    plt.plot([12270/10, 0], [0, 12270/500],color='black')
    plt.plot([37000/10,0],[0,37000/500],color='teal')
    plt.plot([20500/10,0],[0,20500/500],color='indigo')
    plt.show()
    
    X,Y= np.linspace(0,38000/500) , np.linspace(0,38000/10)
    
    X,Y = np.meshgrid(X, Y)
    import matplotlib.pyplot as plt
    plt.style.use("seaborn")
    #from matplotlib.pyplot import colorbar, pcolor, show
    Z = func.cost(X, Y)
    f , ax = plt.subplots(figsize=(8,6))
    plot = ax.pcolor(Y, X, Z,cmap='bwr')
    
    plt.xlabel('Falso Positivo')
    plt.ylabel('Falso negativo')
    plt.title('Heatmap de custo, relação falso negativo x falso positivo.')
    f.colorbar(plot,ax=ax)
    plt.plot([12270/10, 0], [0, 12270/500],color='black')
    plt.plot([37000/10,0],[0,37000/500],color='teal')
    plt.plot([20500/10,0],[0,20500/500],color='indigo')
    plt.show()
    
    
