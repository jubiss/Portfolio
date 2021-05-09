 #Heat Map

import numpy as np
import pandas as pd

def cost(X,Y):
    return (X*500+Y*10)

def predict_random(neg_p): 
    
#    neg = train['class'].value_counts()[0]
#    neg_p = neg/len(train['class'])
    if round(np.random.random(),4) < neg_p:
        return 'neg'
    return 'pos'

def cost_random_model(true,predicted):
    if true == 'pos':
        return 500
    return 10


def random_model():
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
#Random model
random_cost = random_model()

best_val = 12270
sem_thres =20500

X,Y= np.linspace(0,375) , np.linspace(0,15625)

X,Y = np.meshgrid(X, Y)
import matplotlib.pyplot as plt
plt.style.use("seaborn")
#from matplotlib.pyplot import colorbar, pcolor, show
Z = cost(X, Y)
f , ax = plt.subplots(figsize=(8,6))
plot = ax.pcolor(Y, X, Z,cmap='bwr')

plt.xlabel('Falso Positivo')
plt.ylabel('Falso negativo')
plt.title('Heatmap de custo, relação falso negativo x falso positivo.')
f.colorbar(plot,ax=ax)
plt.plot([15625, 0], [0,312.5],color='brown')

plt.plot([best_val/10, 0], [0, best_val/500],color='black')
plt.plot([37000/10,0],[0,37000/500],color='teal')
plt.plot([sem_thres/10,0],[0,sem_thres/500],color='indigo')
plt.show()

X,Y= np.linspace(0,38000/500) , np.linspace(0,38000/10)

X,Y = np.meshgrid(X, Y)
import matplotlib.pyplot as plt
plt.style.use("seaborn")
#from matplotlib.pyplot import colorbar, pcolor, show
Z = cost(X, Y)
f , ax = plt.subplots(figsize=(8,6))
plot = ax.pcolor(Y, X, Z,cmap='bwr')

plt.xlabel('Falso Positivo')
plt.ylabel('Falso negativo')
plt.title('Heatmap de custo, relação falso negativo x falso positivo.')
f.colorbar(plot,ax=ax)
plt.plot([best_val/10, 0], [0, best_val/500],color='black')
plt.plot([37000/10,0],[0,37000/500],color='teal')
plt.plot([sem_thres/10,0],[0,sem_thres/500],color='indigo')
plt.show()



costs = np.array([random_cost,cost(375,0),cost(0,15625),37000,best_val])
label = ['Aleatória','Levar nenhum','Levar todos','Custo 2020','ML']
import seaborn as sns
plt.title('Custos para diferentes estratégias')
plt.ylabel('$')
sns.barplot(x=label,y=costs)
plt.show()

costs = np.array([random_cost,cost(375,0),cost(0,15625),37000])
label = ['Aleatória','Levar nenhum','Levar todos','Custo 2020']
import seaborn as sns
plt.title('Custos, estratégias simples e custo atual')
plt.ylabel('$')
sns.barplot(x=label,y=costs)
plt.show()

costs = np.array([37000,20500,best_val])
label = ['Custo 2020','ML sem threshold','ML com threshold']
import seaborn as sns
plt.title('Custo atual e com Machine Learning')
plt.ylabel('$')
sns.barplot(x=label,y=costs)
plt.show()

 
econ_costs =(1- np.array([20500,best_val])/37000)*(100)
labe_econl = ['ML sem threshold','ML com threshold']
plt.title('Redução de custos em relação ao custo de 2020')
plt.ylabel('%')
x1, x2 , y1, y2 = plt.axis()
plt.axis((x1,x2,0,100))
sns.barplot(x=labe_econl,y=econ_costs)
plt.show()

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

""" Não usar
costs = np.array([cost(375,15625),random_cost,cost(375,0),cost(0,15625),cost(25,385)])
label = ['Errar todos','Aleatória','Levar nenhum','Levar todos','ML']
import seaborn as sns
plt.title('Custos para diferentes estratégias')
plt.ylabel('R$')
sns.barplot(x=label,y=costs)
plt.show()

econ_costs =(1- np.array([random_cost,cost(375,0),cost(0,15625),37000,cost(best_val[0],best_val[1])])/random_cost)*(100)
labe_econl = ['Aleatória','Levar nenhum','Levar todos','Custo 2020','Usando ML']
economia = (1-costs/costs[1])*100
plt.title('Redução de custos em relação a escolher aleatóriamente')
plt.ylabel('%')
x1, x2 , y1, y2 = plt.axis()
plt.axis((x1,x2,0,100))
sns.barplot(x=labe_econl,y=econ_costs)
plt.show()

"""


