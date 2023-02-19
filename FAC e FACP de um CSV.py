from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_process import ArmaProcess

import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.filterwarnings('ignore')

#defione o tamanho da imagem
plt.rcParams['figure.figsize'] = [10, 7.5]

#cria um sample com 1000 valores
arma1 = np.loadtxt('L5Q7.csv', skiprows=1, delimiter=',')
sizearray = arma1.size()
print(sizearray)

#adiciona o modelo no gráfico
plt.plot(arma1)
#define o título
plt.title('Simulador ARMA(1,1)')
#define o range do gráfico
plt.xlim([0,200])
#plt.show()

#coloca a FAC e FACP no gráfico
plot_acf(arma1)
plot_pacf(arma1)
plt.show()