#  Previsão de necessidade de manutenção

Nesse trabalho eu mostro como prever a necessidade de manutenção de caminhões utilizando Machine Learning.

## Introdução

Quando se pensa em manutenção de automóveis, usualmente se segue as especificações do fabricante de períodos ideais para realizar uma revisão. As revisões prévias servem para reparar possíveis danos ou falhas que o automóvel possa ter adquirido com o tempo, isso leva uma redução de custo, já que a não reparação desses danos podem levar a danos posteriores mais custosos de se reparar, além de evitar uma possível demora para solução de um problema mais complicado.

Quando olhamos para a perspectiva de um dono de um automóvel individual realizar a manutenção nas datas previstas pelo fabricante não é um problema, mas quando olhamos para frotas de caminhões, por exemplo, isso pode acarretar em um custo extra de mandar para manutenção um automóvel que não necessita de manutenção, ou de automóveis que tiveram problemas antes mesmo das revisões recomendadas pelo fabricante.

Uma solução para esses problemas é a utilização de dados de funcionamento dos caminhões junto com modelos de Machine Learning, para prever quando esses caminhões vão necessitar de manutenção. Com isso conseguimos identificar quais caminhões necessitam de manutenção independente de indicações do fabricante, resolvendo assim o problema de enviar caminhões que não necessitavam de manutenção, e de caminhões que necessitavam, mas não foram enviados devido a recomendações do fabricante.

##Contexto

Com relação a esse problema, uma empresa que possui uma frota de caminhões necessita de ajuda para redução dos seus custos de manutenção, ela vem observando que seus custos extras vem aumentando anualmente chegando ao pico de $37000.00 no último ano. Os custos extras se dão por caminhões que são enviados para manutenção mas não precisavam (10$), e de caminhões que necessitavam de manutenção mas não foram enviados (500$), os outros custos relacionados a manutenção de caminhões que necessitavam de manutenção é visto como um custo necessário e irredutível. Também é importante entender quais são as principais variáveis responsáveis por mandar ou não um caminhão para manutenção.

##Dados

Os dados foram obtidos do dataset "Air pressure system failures in Scania trucks", contendo 171 variáveis diferentes com nome encriptados.
Se trata de um dataset assimétrico em que apenas 2.34% do total de dados eram falsos negativos (Necessitava de manutenção, mas não foi enviado).

##Limpeza de dados:

Para a parte de limpeza de dados foi realizada a limpeza de variáveis com mais de 50% de Missing Data e variáveis que só estavam no dataset de treino ou de teste.

##Modelo e métricas:

Para solução do problema utilizei o XGBoost, variando o threshold de classificação. Para encontrar o threshold ótimo treinei e testei diferentes amostras do dataset de treino, o threshold ótimo é então escolhido a partir da minimização de uma métrica customizada (False Positive * 10 + False Negative *500).

##Resultados

Utilizando a solução proposta é possível reduzir o custo total de $37000.00 para $12300.00, uma redução de 3x custo total, o recorde na "Industrial Challenge 2016 at The 15th International Symposium on Intelligent Data Analysis (IDA)" foi de 9920.
Também é apresentado as principais variáveis que possuem relação com um caminhão ir para a manutenção ou não, e as que não foram utilizadas.

### Scripts

1)Main: Programa principal.

2)Plots: Possui os plots utilizados para a confecção dos slides; as funções utilizadas para plotar feature importance e as que não foram utilizadas pelo modelo.

3)Functions: Funções de limpeza de dados; encontrar o threshold ótimo do dataset de treino e teste; implementação do modelo; feature importance; 


