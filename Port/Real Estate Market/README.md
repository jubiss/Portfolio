# Previsão de preço de imóveis

Nesse trabalho lido com o problema de predição do preço de imóveis na cidade de São Paulo.

## Introdução:

Obter o preço de mercado de um imóvel é etapa fundamental, para sua venda ou compra. 
Preços exorbitantes tendem a "Encalhar" o imóvel, fazendo com que sua venda seja demorada ou nunca ocorra. Já preços a baixo do preço de mercado levam prejuízos ao dono do imóvel poderia ter um maior lucro sobre sua venda.
O preço de um imóvel esta diretamente ligado com sua liquidez (velocidade de venda), quanto menor o preço de um imóvel maior sua liquidez.
Realizando uma predição mais acurada diminui as possibilidades de prejuízo tanto para o vendedor, ao estar ganhando o valor de mercado daquele imóvel, quanto para o comprador, por não estar sendo enganado com relação ao custo do imóvel.
Com uma previsão acurada de preços é possível gerar estratégias para vendas e compras de imóveis, como ajuste de preços para uma maior liquidez, e estratégias de compra e vendas de imóveis.

O processo de Valuation (Precificação de imóvel) normalmente é realizado por um corretor de imóveis, que utilizando de sua experiência e aprendizado avalia o preço do imóvel, esse tipo de avaliação possui alguns problemas como um forte viés baseado em experiências passadas do corretor, o alto custo desse profissional, avaliações limitadas a experiências do corretor e falta de escalabilidade para avaliação de múltiplos imóveis de uma vez.

Para resolver esse problema é possível utilizar modelos de Machine Learning para realizar essas avaliações, em que temos a vantagem de utilizar métodos estatísticos dando um menor viés a avaliação, custo menor que de um corretor, maior capacidade de generalização e escalabilidade do processo de avaliação. A desvantagem do modelo é a necessidade de dados para fazer essa avaliação e os limites que esses dados dão a precisão desse modelo.

## Dados:

Os dados foram obtidos em um site de anúncios de apartamentos ​(https://123i.uol.com.br​), e contavam com dados:
    
    Rua
    Nome do prédio em que o imóvel está localizado
    Número de quartos 
    Número de garagens
    Área do imóvel
    Valor máximo do imóvel
    Valor mínimo do imóvel
    Estimativa pontual do preço do imóvel
    Localização em coordenadas do imóvel

## Limpeza de dados:

A grande maioria dos dados de Rua, Nome do prédio estavam sujos e não existiam dados de bairros, então utilizei da localização de coordenadas para descobrir o bairro de cada um dos imóveis utilizando a API do Google Maps para fazer reverse Geocoding.
Com relação aos outros dados fiz a limpeza padrão e remoção de Outliers e verificação de Outliers naturais ou artificiais.

## Engenharia de variáveis:

Utilizei a variável bairro, obtida a partir do Geocoding, para gerar uma variável que separava os bairros em 10 baseado no valor/metro_quadrado. Mantive a variável bairro e mantive somente os bairros que apareciam em mais de 1% do data set, o resto foi armazenado em um novo bairro chamado "outros". 

## Modelo e métricas:

Utilizei dois modelos, primeiro o modelo Random Forest para servir como Benchmark e o XGBoost que foi meu modelo principal.
Por se tratarem de dois modelos de árvore eles possuem limitações semelhantes como baixa acurácia para extrapolar previsões não presentes nos dados.
Para métrica utilizei duas RMSQ (Root Mean Square) para avaliar como estava o erro do modelo em outliers e MAE (Mean Absolute Error) para avaliar o erro médio do modelo.

## Resultados:

Utilizando o modelo apresento quão melhor é o XGBoost comparado com o Random Forest.
Como as métricas se comportam para diferentes formas de tratamento de dados e remoção de outliers.
Distribuição de erros do modelo.
Quais são as variáveis que melhor explicam o valor de um imóvel, utilizando Gain e F-score.
Como avaliar o risco de um imóvel utilizando uma distribuição Normal, e quais são os pré-requisitos para isso.
Quão bom o modelo atuava com dados limitados a uma região.
Qual era o número mínimo de dados para obter performance semelhante a obtida nos resultados principais.

# Scripts:

1) main: Programa principal.

2) functions: Implementações das funções utilizadas no programa principal.

3) dados_bairros: Obtém os dados de geolocalização pela API do Google Maps.

4) Merge_dados_geo: Junta todos os dados obtidos pela API do Google Maps.
