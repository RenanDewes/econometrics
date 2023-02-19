from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_process import ArmaProcess

import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = [10, 7.5]

ar1 = np.array([1, 0.33])
ma1 = np.array([1, 0.9])

arma1 = ArmaProcess(ar1, ma1).generate_sample(nsample=1000)

plt.plot(arma1)
plt.title('Simulador ARMA(1,1)')
plt.xlim([0,200])
#plt.show()

plot_acf(arma1)
plot_pacf(arma1)
plt.show()