#bibliotecas/funções de regressão
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.arima_model import ARMAResults
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
#outras bibliotecas (visuais e para vetores)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#ignorar avisos que não são erros
import warnings
warnings.filterwarnings('ignore')


#atribui os valores do arquivo csv ao vetor arma
arma = np.loadtxt('data.csv', skiprows=1, delimiter=',')

#pega o tamanho do arquivo e guarda numa variável
size_array = len(arma)
#define os valores limite do intervalo de confiança (1.96/√T)
top_limit = 1.96/(size_array**(1/2))
bottom_limit = -1.96/(size_array**(1/2))


'''atribui os valores de lag à variável acf_values
qstat = true significa que retornará também os valores do teste Ljung-Box e os p valores
Retorna uma matriz em que a primeira linha são os valores de FAC, a segunda os valores de Ljung e a última os p valores'''
acf_values = acf(arma, qstat=True, alpha=0.05)
#print(acf_values) #printa a matriz
#print('Número de lags: ', len(acf_values[0])-1) #-1 porque o 0 não conta

ar_level = 0
#para cada lag de 1 até o tamanho do vetor de lags, printe o valor na tela e, se estiver fora do intervalo de confiança, some 1 na variável para vermos o "nível" do MA
for lag in range(1, len(acf_values[0])):
	#print(acf_values[0][lag])
	if((acf_values[0][lag]>top_limit) or (acf_values[0][lag]<bottom_limit)):
		ar_level=+1

#print(ar_level)


pacf_values = pacf(arma, alpha=0.05)
#print(pacf_values)

ma_level = 0
#para cada lag de 1 até o tamanho do vetor de lags, printe o valor na tela e, se estiver fora do intervalo de confiança, some 1 na variável para vermos o "nível" do MA
for lag in range(1, len(pacf_values[0])):
	#print(pacf_values[0][lag])
	if((pacf_values[0][lag]>top_limit) or (pacf_values[0][lag]<bottom_limit)):
		ma_level=+1

#print(ma_level)


#Cria uma lista com vetores de duas posições (inicialmente zeradas) que serão os modelos ARMA que passarem no Ljung Box
validated_models = []

#range(start, stop, step): stop em intervalo aberto. Preciso das combinações incluindo o 0
for ar in range(ar_level,-1,-1):
	for ma in range(ma_level,-1,-1):
		if (ma==0) and (ar==0):
			break

		#estima o modelo ARIMA(ordem AR, ordem I, ordem MA)
		model = ARIMA(arma, order=(ar, 0, ma))
		#estima os parâmetros do modelo
		results = model.fit()
		#pega os resíduos
		residuals = pd.DataFrame(results.resid)

		#variável para controlar se há p-valores menores que 0.05
		pvalue_control = False

		#esse range são os lags que desejamos
		for lag in range(11,26):
			ljung_box = sm.stats.acorr_ljungbox(residuals, lags=[lag], return_df=True)
			pvalue = ljung_box['lb_pvalue'].values[0]

			#verifica se é menor que 0.05 (está 0.46 apenas como teste devido à minha base de dados)
			if pvalue < 0.05:
				pvalue_control = True
				break

			#print(str(ar) + ", " + str(ma))
			#print(pvalue)

		if pvalue_control == False:
			#print(str(ar) + ", " + str(ma))
			bic_value = results.bic
			#print(bic_value)
			validated_models.append([[ar,ma],bic_value])

#printa os modelos que passaram nos testes			
print(validated_models)

#Variável que será atribuída o menor valor de BIC
low_bic = 0
#Variável que será atribuída o modelo com o menor valor de BIC
final_model = [0,0]

for i in range(0,len(validated_models)):
	if i == 0:
		#Na primeira iteração tomamos como premissa que o menor valor é o primeiro, depois verificamos se há outros menores
		low_bic = validated_models[0][1]
		final_model = validated_models[0][0]
	else:
		if validated_models[i][1] < low_bic:
			low_bic = validated_models[i][1]
			final_model = validated_models[i][0]
			
print(low_bic)
print(final_model)

