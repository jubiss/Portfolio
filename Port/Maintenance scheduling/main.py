import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import functions as func
import plots 
 
"""
Importar dados/limpeza
"""
plt.style.use("seaborn")
 
df = pd.read_csv('base_vigente_anos_anteriores.csv')
test = pd.read_csv('base_vigente_2020.csv')

#na_threshold=50 remove todas as features que possuam mais de 50% de nan values 
df,test = func.data_clean(df,test,na_threshold=50)

"""
Resultados modelo sem Threshold
"""

custo_modelo_sem_thres = func.modelo_xgboost(df,test,threshold=0)

"""
Resultados modelo com Threshold
"""
##Slide 10
opt_thres = func.find_optimal_threshold(df,rept=200,plot=True)

##Slide 11 
custo_modelo_final, xgb_2 = func.modelo_xgboost(df,test,threshold=opt_thres,return_model = True)
"""
Resultado modelo com Threshold valor otimo
"""

threshold = np.linspace(0,0.45,800)
custo_thres = func.modelo_xgboost(df,test,threshold=threshold)
plt.plot(threshold,custo_thres)
custo_real_optimo = min(custo_thres)

#Slide 12
plt.xlabel('Threshold')
plt.ylabel('$')
plt.title('Threshold x Custo (Teste)')
plt.show()

"""
PLot Feature Importance
"""
#Slide 14,15
plots.feature_importance_plot(xgb_2,df)

"""
Plot Features não utilizadas pelo modelo
"""
#Slide 16
plots.feature_without_importance(xgb_2,df,test)