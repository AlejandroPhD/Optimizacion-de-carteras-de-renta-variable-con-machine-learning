

## Importar todos los paquetes necesarios.
import numpy as np # Para datos numéricos en forma de matriz
import pandas as pd # Para datos en forma de tabla
import matplotlib.pyplot as plt # Para fines de visualización (gráficos).
import matplotlib.pylab as pl # Para fines de visualización (gráficos).
import yfinance as yf
import random
import pandas_datareader as pdr

import csv # Para leer archivos
from datetime import datetime, timedelta

from matplotlib.collections import LineCollection # Para fines de visualización (gráficos).
from sklearn import cluster, covariance, manifold # Para análisis gráfico

from scipy.stats import norm # Scipy.stats contiene funciones estadísticas y probabilísticas, como la función de densidad de probabilidad y la función de distribución acumulada de la distribución normal (norm)
from scipy.optimize import minimize #La función "minimize" se utiliza para minimizar una función de una o varias variables, utilizando diversos métodos de optimización. Busca los valores de entrada de una función que minimicen su valor de salida




# Cargar conjunto de datos de información de empresas 
# Descargar la lista de empresas que cotizan en el S&P 500 desde Wikipedia
url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
sp500 = pd.read_html(url, header=0)[0]

# Filtrar solo los tickers, nombres y sectores de las empresas del S&P 500
sp500_tickers = sp500['Symbol'].tolist()
sp500_names = sp500['Security'].tolist()
sp500_sectors = sp500['GICS Sector'].tolist()

# Combinar los tres en una tabla
sp500_table = pd.DataFrame({'Ticker': sp500_tickers, 'Name': sp500_names, 'Sector': sp500_sectors})




# SUBMUESTREO ALEATORIO
# Seleccionar una SUBmuestra aleatoria de empresas de la lista completa de tickers
num_stocks = 50
random_stocks = random.sample(sp500_tickers, num_stocks )

# Listar las acciones aleatorias por orden alfabetico y guardarlas en una variable para asegurarse de que no cambien durante el resto del código:
sample_stocks = sorted(random_stocks)

# Fecha de inicio y fecha de finalización que se está considerando:
start_date = '2013-03-03'
end_date   = '2023-03-05'




# INFORMACIÓN RESUMIDA DE LAS ACCIONES
# Seleccionar las filas de la tabla que corresponden a los tickers seleccionados aleatoriamente
firms_info = sp500_table[sp500_table['Ticker'].isin(sample_stocks)]

# Establecer Ticker como índice
firms_info = firms_info.set_index('Ticker') 




# DESCARGA DE DATA (retornos)
# Descargar los precios de cierre ajustados por dividendo de Yahoo Finance para la muestra de acciones y calcular su retorno diario:
daily_returns = yf.download(sample_stocks, start=start_date, end=end_date)['Adj Close'].pct_change()

# Elimina el primer elemento, que es NaN
daily_returns = daily_returns[1:]

# PROCESAMIENTO DE DATOS (retornos)
# Reorganizar los datos y convertirlos a frecuencia semanal
# Se crea una lista llamada WEEKLY_DFS
WEEKLY_DFS = []
# En cada iteración de un bucle for, se selecciona una empresa del DataFrame
for i in range(len(daily_returns.columns)):
# Se eliminan las filas con valores faltantes 
    firm = daily_returns.columns[i]
    RET = daily_returns[firm].dropna(axis=0)
# Se convierte los rendimientos diarios a rendimientos semanales utilizando el método resample de Pandas
    RET_weekly = (RET+1).resample('W').prod() - 1 
# El rendimiento semanal se agrega a la lista WEEKLY_DFS.
    WEEKLY_DFS.append(RET_weekly)
    
# Se calcula el número de elementos en cada elemento de WEEKLY_DFS y se almacena en un array llamado len_WEEKLY_DFS
len_WEEKLY_DFS = np.array([len(temp) for temp in WEEKLY_DFS])
Names = daily_returns.columns
# Se crea un DataFrame que tendrá como columnas los nombres de las empresas contenidas en daily_returns. La fecha de índice de este DataFrame se establece como la fecha correspondiente al elemento de WEEKLY_DFS con el mayor número de elementos
data = pd.DataFrame(columns=Names, index=WEEKLY_DFS[np.argmax(len_WEEKLY_DFS)].index)
# Se asigna cada rendimiento semanal en WEEKLY_DFS a la columna correspondiente en data
for i in range(len(Names)):
    data[Names[i]] = WEEKLY_DFS[i]
 


    
# Cargar conjunto de datos de información de la tasa libre de riesgo (TBill)   
# Leer y editar datos en el formato correcto
file_name = 'WTB3MS.csv'
df = pd.read_csv(file_name, delimiter='\t',skiprows=1)

# Convertir la columna 'DATE' del dataframe df en formato datetime
df['DATE']=pd.to_datetime(df['DATE'])
# Convertir la columna 'WTB3MS' del dataframe df a un tipo de datos numérico, cualquier valor no numérico se convertirá en NaN
df['WTB3MS'] = pd.to_numeric(df['WTB3MS'], errors='coerce')

# Crear un nuevo dataframe con una única columna 'RET' y los mismos índices que df.DATE
RET_data=pd.DataFrame(columns=['RET'], index=df.DATE)
# Calcular los retornos semanales 
RET_data['RET'] = (df['WTB3MS'].values/100 + 1) ** (1/52) - 1 

RET_data_weekly = (RET_data['RET']+1).resample('W').prod() - 1 # Convertir los datos a la misma frecuencia semanal que arriba
# Crear un nuevo dataframe con una única columna 'T-Bill' y los mismos índices que RET_data_weekly.
TBill = pd.DataFrame(columns=['T-Bill'], index=RET_data_weekly.index)
#retornos T-Bill
start_date = data.index[0]
end_date = data.index[-1]
# Para igualar la fecha de inicio de los datos de retorno de acciones.
TBill = TBill[TBill.index >= start_date] 
# Para igualar la fecha de finalización de los datos de retorno de acciones.
TBill = TBill[TBill.index <= end_date]   
# Calcular los retornos semanales de los datos de la columna 'RET' y asignarlos a la columna 'T-Bill'
TBill['T-Bill'] = RET_data_weekly  
   
    
    
    
#Funciones utilizadas para obtener estadísticas resumidas 
def sr_annu(r_old,rf_old):
    """
    Este código calcula la relación de Sharpe anualizada, toma como entradas una serie de pandas que contiene los rendimientos de los activos denominada "r_old" y una serie de pandas que contiene la tasa libre de riesgo "rf_old". "r" es el rendimiento de los activos, "rf" es la tasa libre de riesgo. y "n" hace referencia al número de activos
   """
    # se identifican las filas que contienen valores faltantes
    index = r_old.index[r_old.apply(np.isnan)] 
    # Las filas identificadas se eliminan del conjunto de datos
    r = r_old.drop(index).values 
    rf = rf_old.drop(index).values
    
    excess_r = r - rf
    annu_ex_ret = annualize_rets(excess_r, periods_per_year = 52)
    annu_vol = annualize_vol(r, periods_per_year = 52)
    return annu_ex_ret/annu_vol


def annualize_rets(r, periods_per_year = 52):
    """
    Anualiza un conjunto de retornos (calcula el rendimiento promedio anualizado)
    """
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    return (compounded_growth**(periods_per_year/n_periods)-1)*100

        
def annualize_vol(r, periods_per_year = 52):
    """
    Anualiza la volatilidad (desviación estándar de un conjunto de retornos)

    """
    return (r.std()*np.sqrt(periods_per_year))*100     

        
def skewness(r):
    """
   Alternativa a scipy.stats.skew()
   Calcula la asimetría de la Serie o DataFrame proporcionado
   Devuelve un valor float o una Serie
   Skewness se utiliza para medir si los rendimientos tienen una distribución sesgada a la izquierda o a la derecha en relación a la media.
    """
    # Se resta la media de los rendimientos a cada valor para obtener la versión centrada en cero de la serie de rendimientos
    demeaned_r = r - r.mean()
    # Se calcula la desviación estándar de la población de los rendimientos utilizando el parámetro "ddof=0"
    sigma_r = r.std(ddof=0)
    # Se eleva cada valor de la versión centrada en cero al cubo y se calcula la media
    exp = (demeaned_r**3).mean()
    # Se divide este valor por el cubo de la desviación estándar de la población 
    return exp/sigma_r**3

        
def kurtosis(r):
    """
    Alternativa a scipy.stats.kurtosis()
    Calcula la curtosis de la serie o DataFrame proporcionado 
    Devuelve un valor flotante o una serie
    Kurtosis se utiliza para medir el grosor de la cola de dsitribución de los datos en comparación con la distribución normal
    """
    # Se resta la media de los rendimientos a cada valor para obtener la versión centrada en cero de la serie de rendimientos
    demeaned_r = r - r.mean()
    # Se calcula la desviación estándar de la población de los rendimientos utilizando el parámetro "ddof=0"
    sigma_r = r.std(ddof=0)
    # se elevan al cuadrado cada uno de los valores centrados y se calcula su media
    exp = (demeaned_r**4).mean()
    # Se divide este valor entre la desviación estándar de la población elevada a la cuarta potencia
    return exp/sigma_r**4        
    
        
def var_gaussian(r, level=5, modified=True):
    """
    Devuelve el Valor en Riesgo (VaR) gaussiano paramétrico de una Serie o DataFrame.
    Si "modified" es Verdadero, entonces se devuelve el VaR modificado utilizando la modificación de Cornish-Fisher
    """
    # calcula el valor de Z-score para un nivel de confianza asumiendo que es gaussiano
    z = norm.ppf(level/100)
    if modified:
        # el código modifica el valor de Z-score basado en la asimetría y la curtosis observada de la serie de rendimientos utilizando la modificación Cornish-Fisher
        s = skewness(r)
        k = kurtosis(r)
        z = (z +
                (z**2 - 1)*s/6 +
                (z**3 -3*z)*(k-3)/24 -
                (2*z**3 - 5*z)*(s**2)/36
            )
     # Calcula el VaR como la suma de la media de los rendimientos y el producto de la desviación estándar de los rendimientos y el valor Z-score.        
    return -(r.mean() + z*r.std(ddof=0))


def var_historic(r, level=5):
    """
    Devuelve el Valor en Riesgo histórico a un nivel específico, es decir, devuelve el número tal que el "nivel" por ciento de los rendimientos caen por debajo de ese número, y el (100-nivel) por ciento están por encima.
    """
    # Si la entrada es un DataFrame de Pandas, la función utiliza el método "aggregate" para aplicar la función "var_historic" a todas las columnas del DataFrame con el argumento "level" especificado
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    # Si la entrada es una Serie de Pandas, la función calcula el percentil negativo correspondiente al nivel especificado
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    # Si la entrada no es ni una serie ni un DataFrame, la función devuelve un mensaje de error.
    else:
        raise TypeError("Expected r to be a Series or DataFrame")    
    
    
def cvar_historic(r, level=5):
    """
    Calcula el Valor en Riesgo Condicional de una Serie o DataFrame
    """
    # Si es una serie, la función calcula el CVaR basado en la fórmula histórica utilizando la función var_historic
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, level=level)
        # Devuelve la media de los valores que están por debajo del VaR histórico
        return -r[is_beyond].mean()
    # Si es un DataFrame, se utiliza la función aggregate para aplicar el cálculo del CVaR a cada columna del marco de datos
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    # Si la entrada no es ni una serie ni un DataFrame, la función devuelve un mensaje de error.
    else:
        raise TypeError("Expected r to be a Series or DataFrame")        

        
def mdd(r):
    """
    Función utilizada para calcular el maximum Drawdown
    """
    # r es un vector de retorno
    wealth_index = (r+1).cumprod()
    # Determinar el valor máximo acumulativo (picos previos)
    previous_peaks = wealth_index.cummax()
    # Calcular el vector de disminución (drawdown).
    drawdown = wealth_index/previous_peaks - 1
    return (drawdown.min())*100


     

def getSumStat(data, rf = TBill['T-Bill'], rounding = 2):
    """
    Función para obtener información estadística resumida, incluyendo:
    - Retorno total
    - Retorno promedio
    - Retorno promedio anualizado
    - Desviación estándar anualizada
    - Ratio de Sharpe anualizado
    - Máxima disminución (drawdown)
    """  
    #se extrae la fecha de la primera fila de la serie de datos y se la almacena en la variable "date_obj"
    date_obj = data.index[0]
    #se utiliza la clase "timedelta" del módulo datetime para calcular el primer día de la semana, el cálculo se realiza restando los días de la semana de la fecha de "date_obj"
    start_of_week = date_obj - timedelta(days=date_obj.weekday())
    #se formatea la fecha de "start_of_week" en el formato "mes/día/año" utilizando el método "strftime"
    start = start_of_week.strftime("%m/%d/%Y")
    #la fecha de la última fila de la serie de datos se formatea de manera similar
    end =  data.index[-1].strftime("%m/%d/%Y")
        
    print('Summary Statistic Information from ' + start + ' to ' + end + ':')
    # Comprueba si hay valores faltantes en el conjunto de datos dentro del período de tiempo
    if(data.isnull().values.any()):
        print('WARNING: Some firms have missing data during this time period!')
        print('Dropping firms: ')
    # Si hay valores faltantes, entonces elimina esas empresas (columnas) antes de calcular sus estadísticas resumidas.
        for Xcol_dropped in list(data.columns[data.isna().any()]): print(Xcol_dropped)
        data = data.dropna(axis='columns')
    
    # Se extrae la información de sector de cada empresa del DataFrame "firms_info" y se almacena en la variable "sectors"
    sectors = firms_info.Sector 
    # Se crea un DataFrame vacío llamado "ss_temp" con las mismas columnas que se utilizarán para almacenar las medidas      estadísticas de resumen.
    ss_temp = pd.DataFrame(sectors, index = data.columns, columns=['Sector'])
    # se calculan las siguientes medidas estadísticas para cada empresa y se almacenan en las columnas correspondientes del DataFrame "ss_temp"
    # Total Return (%): el rendimiento total de la empresa durante todo el período de tiempo
    ss_temp['Total Return(%)'] = np.round((((data+1).cumprod()-1)*100).iloc[-1] , rounding)
    # Ave Return (%): el rendimiento promedio semanal de la empresa
    ss_temp['Weekly Ave Return(%)'] = np.round(data.mean()*100, rounding)
    # Annu. Ave Return (%): el rendimiento promedio anualizado de la empresa
    ss_temp['Annu. Ave Return(%)'] = np.round(data.apply(annualize_rets, periods_per_year = 52), rounding)  
    # Annu. Std (%): la desviación estándar anualizada del rendimiento de la empresa
    ss_temp['Annu. Vol(%)'] = np.round(data.apply(annualize_vol, periods_per_year = 52), rounding)
    # Annu. Sharpe Ratio: el índice de Sharpe anualizado de la empresa
    ss_temp['Sharpe Ratio'] = np.round(data.apply(sr_annu, rf_old=rf), rounding)
    # Max Drawdown (%): el mayor retroceso (drawdown) de la empresa durante el período de tiempo
    ss_temp['Max Drawdown(%)'] = np.round(data.apply(mdd), rounding)
    #El código devuelve el Dataframe "ss_temp" que contiene todas las medidas estadísticas de resumen calculadas
    return(ss_temp)
        
        

        
# FUNCIONES USADAS PARA EL ANÁLISIS GRÁFICO
# Reference: https://scikit-learn.org/stable/auto_examples/applications/plot_stock_market.html

def graphicalAnalysis(dataset, start_date, end_date, 
                      Sectors_chosen = [],
                      drop_firm = [], 
                      display_SumStat = True, display_IndRet = True, 
                      data_rf = TBill ):
    """
    Esta es la Función para realizar un análisis de red gráfico.Imprime información de los clusters, información de la red gráfica, estadísticas resumidas y gráfico del desempeño individual de las empresas. Retorna las matrices de correlación y precisión, así como información de configuración para el gráfico.
NOTA: eliminará las empresas que tengan datos faltantes durante el período de tiempo especificado.
    """
    # Verificar si las fechas de inicio y fin ingresadas son válidas en relación a las fechas del conjunto de datos proporcionado
    # Si la fecha de inicio es posterior a la fecha de finalización, imprime un mensaje de error y devuelve dos valores nulos.
    if(datetime.strptime(start_date, "%Y-%m-%d") > datetime.strptime(end_date, "%Y-%m-%d")):
        print('ERROR: Revision needed! The entered \"start_date\" should be before \"end_date\".')
        return 0,0
    # Si la fecha de inicio es anterior al primer valor de fecha en el conjunto de datos, ajusta la fecha de inicio a la fecha del primer valor de fecha en el conjunto de datos y emite una advertencia.
    if (dataset.index[0]- timedelta(days=dataset.index[0].weekday()) > datetime.strptime(start_date, "%Y-%m-%d")):
        print('WARNING: the entered \"start_date\" is outside of the range for the given dataset.')
        print('The \"start_date\" is adjusted to the earliest start_date, i.e. ',
              (dataset.index[0]-timedelta(days=dataset.index[0].weekday())).strftime("%Y-%m-%d"))
        print()
    # Si la fecha de finalización es posterior al último valor de fecha en el conjunto de datos, ajusta la fecha de finalización a la fecha del último valor de fecha en el conjunto de datos y emite una advertencia.    
    if (dataset.index[-1] < datetime.strptime(end_date, "%Y-%m-%d")):
        print('WARNING: the entered \"end_date\" is outside of the range for the given dataset.')
        print('The \"end_date\" is adjusted to the lastest end_date, i.e. ',
              dataset.index[-1].strftime("%Y-%m-%d"))
        print()
    
    # Este código extrae los datos de retornos para el período de tiempo especificado.
    # Crea una copia del conjunto de datos dataset para el período de tiempo que es posterior o igual a la fecha de inicio. 
    temp = dataset[dataset.index >= start_date].copy()
    # Selecciona las filas de la copia creada que son anteriores o iguales a la fecha de finalización para obtener el conjunto de datos final X.
    X = temp[temp.index <= end_date].copy()
    # Para igualar la fecha de inicio de los datos de retorno de acciones.
    temp = data_rf[data_rf.index >= start_date].copy()
    # Para igualar la fecha de finalización de los datos de retorno de acciones.
    data_rf2 = temp[temp.index <= end_date].copy()       
    # Comprobar si estamos utilizando todos los sectores o eliminando algunos sectores. Si se han especificado sectores específicos, entonces verifica que esos sectores estén presentes en el conjunto de datos.
    if ((not Sectors_chosen) == False):
        if(all([(s in firms_info.Sector.unique()) for s in Sectors_chosen])):
            f_in_sector_chosen = []
            #Si lo sectores especificados están presentes, filtra el conjunto de datos para incluir solo las empresas que están en los sectores especificados y muestra los sectores que se han seleccionado. 
            for s in Sectors_chosen:
                f_in_sector_chosen += list(firms_info[firms_info.Sector == s].index)
            X = X[f_in_sector_chosen]
            print('Sectors choosen in the Graphical Analysis are:')
            print(Sectors_chosen)
            print()
        # Si no están presentes, se imprime un mensaje de error y se devuelve un valor de 0 para la función
        else:
            print('ERROR: Revision needed! At Least 1 Sector entered in the \"Sectors_choosen\" option is NOT in the dataset!')
            print('Check your format!')
            return 0,0
    
    # Comprobar si estamos utilizando todas los empresas o eliminando algunas. 
    # Se verifica si se ha especificado la opción "drop_firm" para eliminar algunas empresas del conjunto de datos. Si se ha especificado, el código verifica si todas las empresas en la lista están presentes en el conjunto de datos X. Si todas las empresas están presentes, se eliminan de "X" y se muestra un mensaje que indica las empresas eliminadas. 
    if((not drop_firm) == False):
        if(all([(f in X.columns) for f in drop_firm])):
            print('The following Firms are dropped:')
            print(drop_firm)
            print()
            X.drop(columns = drop_firm, inplace = True)
     #Si al menos una empresa no está presente en "X", se muestra un mensaje de error y se devuelve (0, 0) como resultado.  
        else:
            print('ERROR: Revision needed! At Least 1 firm entered in the \"drop_firm\" option is NOT in the dataset!')
            print('Check your format!')
            return 0,0
    
    # Verifica si hay valores nulos "NAs" en el conjunto de datos dentro del período de tiempo dado
    # Si es así, entonces elimina esas empresas antes de realizar el análisis gráfico.
    if(X.isnull().values.any()):
        print('WARNING: Some firms have missing data during this time period!')
        print('Dropping firms: ')
        for Xcol_dropped in list(X.columns[X.isna().any()]): print(Xcol_dropped)
        X = X.dropna(axis='columns')
        print()
        
    #se extrae la fecha de la primera fila de la serie de datos y se la almacena en la variable "date_obj"    
    date_obj = X.index[0]
    #se utiliza la clase "timedelta" del módulo datetime para calcular el primer día de la semana, el cálculo se realiza restando los días de la semana de la fecha de "date_obj"
    start_of_week = date_obj - timedelta(days=date_obj.weekday())
    #se formatea la fecha de "start_of_week" en el formato "mes/día/año" utilizando el método "strftime"
    start = start_of_week.strftime("%m/%d/%Y")
    #la fecha de la última fila de la serie de datos se formatea de manera similar
    end =  X.index[-1].strftime("%m/%d/%Y")
    
    # Obtener los nombres de las empresas del conjunto de datos
    names = np.array(list(X.columns))
    
    # Mostrar el número de empresas examinadas
    print('Number of firms examined:', X.shape[1])
    
    # Se utiliza el algoritmo "Graphical Lasso" de la biblioteca "scikit-learn" para estimar la matriz de precisión. Se aprende una estructura gráfica a partir de las correlaciones. En particular, el algoritmo ajustará los parámetros del modelo hasta que el número de iteraciones sea 1000 o hasta que la convergencia sea alcanzada antes. 
    edge_model = covariance.GraphicalLassoCV(max_iter=1000)

   # Se estandariza la serie de tiempo utilizando las correlaciones en lugar de la covarianza. Esto se hace para mejorar la eficiencia de recuperación de la estructura del modelo, lo que significa que es más fácil y rápido recuperar la relación entre las variables utilizando las correlaciones en lugar de la covarianza.
    X_std = X / X.std(axis=0)
    edge_model.fit(X_std)
    
    # Se utiliza el algoritmo de clustering Affinity Propagation para agrupar las empresas (acciones). Se usa el objeto "edge_model.covariance_" como entrada, que representa la matriz de covarianza de los datos.
    _, labels = cluster.affinity_propagation(edge_model.covariance_)
    # Se determina el número total de etiquetas de cluster utilizando la función "max()" en la variable "labels".
    n_labels = labels.max()
    #Se itera a través de cada etiqueta de cluster y se imprime la lista de nombres de las muestras que pertenecen a ese cluster, utilizando la variable "names" para hacer referencia a los nombres de las empresas.
    for i in range(n_labels + 1):
        print('Cluster %i: %s' % ((i + 1), ', '.join(names[labels == i])))

    # Se busca una representación de baja dimensión para la visualización de los datos. En particular, la mejor posición para los nodos (acciones) en un plano 2D, lo que permitiría visualizar y analizar las relaciones entre las acciones de manera más efectiva
    # Se recurre al algoritmo MDS (Multidimensional Scaling) de la biblioteca "scikit-learn" de Python. Se crea un modelo estableciendo que la representación final tendrá dos dimensiones utilizando el parámetro "n_components=2".
    node_position_model = manifold.MDS(n_components=2, random_state=0)
    # se está utilizando este modelo para ajustar la matriz de datos estandarizados "X_std" transpuesta, utilizando el método "fit_transform()" que encuentra la mejor representación de baja dimensión para los datos en función de las similitudes entre ellos. Finalmente, se transpone la matriz resultante de nuevo para obtener una representación de dos dimensiones de los datos para su posterior análisis y visualización.
    embedding = node_position_model.fit_transform(X_std.T).T

    # Especificar los colores de los nodos mediante las etiquetas de los clusters. Se llama a la función "pl.cm.jet()" de la biblioteca "matplotlib" para crear una lista de colores, que va desde el color rojo hasta el color azul, utilizando el parámetro "np.linspace(0,1,n_labels+1)".
    color_list = pl.cm.jet(np.linspace(0,1,n_labels+1))
    # Se crea una lista llamada "my_colors" que contiene los colores para cada nodo en función de su etiqueta de cluster. 
    my_colors = [color_list[i] for i in labels]
    # Calcular las correlaciones parciales entre las variables del modelo y luego establece una matriz de correlaciones parciales umbralizada (non_zero) para identificar qué correlaciones son significativas.
    
    # Se crea una copia de la matriz de precisión para calcular las correlaciones parciales. 
    partial_correlations = edge_model.precision_.copy()
    # Se calcula la raíz cuadrada inversa de la diagonal de la matriz de correlaciones parciales 
    d = 1 / np.sqrt(np.diag(partial_correlations))
    # Se multiplica por la matriz de correlaciones parciales para normalizar la matriz.
    partial_correlations *= d
    # Se multiplica la matriz normalizada por la transpuesta de sí misma para obtener una matriz simétrica. 
    partial_correlations *= d[:, np.newaxis]
    # Se crea una matriz booleana que es verdadera para cada par de variables que tienen una correlación parcial mayor a 0.02 y falsa si es menor o igual a 0.02.
    non_zero = (np.abs(np.triu(partial_correlations, k=1)) > 0.02)
    
    # Este código primero extrae y almacena los valores absolutos de las correlaciones parciales que han sido umbralizadas en la matriz booleana (non_zero), para solo considerar las correlaciones parciales que son significativas. 
    values = np.abs(partial_correlations[non_zero])
    # Se busca la correlación parcial más fuerte entre las variables del modelo, que es útil para el análisis y la visualización del grafo de correlación
    val_max = values.max()
    
    # Título del gráfico
    title = 'Graphical Network Analysis of Selected Firms over the Period '+start+' to '+end+' (Weekly)'
    
    # Mostrar el grafo de correlación parcial
    graphicalAnalysis_plot(d, partial_correlations, my_colors,
                           names, labels, embedding, val_max, title)
    
    # la configuración del gráfico
    plot_config = [d, partial_correlations, my_colors, names, labels, embedding, val_max, title]
    
    # Para el rendimiento individual de las empresas durante el período dado
    if (display_IndRet):
        print('Individual Stock Performance over the Period '+ start+' to '+end+' (Weekly):')
        l_r = int(np.ceil(len(names)/4))
        l_c = 4
        f_hei = l_r * 2.5
        f_wid = l_c * 4
        ax = (X+1).cumprod().plot(subplots=True, layout=(l_r, l_c), figsize=(f_wid, f_hei),
                                  logy=True, sharex=True, sharey=True, x_compat=True,
                                  color = my_colors);
        for i in range(l_c):
            ax[0,i].xaxis.set_tick_params(which='both', top = True, labeltop=True, labelrotation=40)
        plt.show()

    # Mostrar estadísticas resumidas para cada empresa durante el período indicado
    if (display_SumStat):
        display(getSumStat(X, rf = data_rf2['T-Bill']))
    
    return [edge_model.covariance_, edge_model.precision_], plot_config




def graphicalAnalysis_plot(d, partial_correlations, my_colors,
                           names, labels, embedding, val_max, title):
    """
    Función utilizada para trazar el gráfico de red gráfica
    """
    # Se crea una matriz booleana que es verdadera para cada par de variables que tienen una correlación parcial mayor a 0.02 y falsa si es menor o igual a 0.02.
    non_zero = (np.abs(np.triu(partial_correlations, k=1)) > 0.02)
    # Se busca la correlación parcial más fuerte entre las variables del modelo, que es útil para el análisis y la visualización del grafo de correlación   
    n_labels = labels.max()
    
    # Para el gráfico de red de correlación
    fig = plt.figure(1, facecolor='w', figsize=(12, 5))
    # Se borra la figura para asegurarse de que no haya nada dibujado previamente en ella.
    plt.clf()
    ax = plt.axes([0., 0., 1., 1.])
    # Se desactiva los ejes para que no se muestren los marcos de los ejes en el gráfico final.   
    plt.axis('off')

    # Representa gráficamente los nodos utilizando las coordenadas de nuestra incrustación (embedding)
    # Se grafica un diagrama de dispersión utilizando las dos primeras dimensiones de un embedding (incrustación)
    plt.scatter(embedding[0], embedding[1], s=500 * d ** 2, c= my_colors)

    # Representa gráficamente los arcos (conexión entre nodos)
    start_idx, end_idx = np.where(non_zero)
    # a sequence of (*line0*, *line1*, *line2*), where::
    #           linen = (x0, y0), (x1, y1), ... (xm, ym)
    # Este código crea una colección de líneas  utilizando las coordenadas del embedding para cada par de nodos conectados por un arco. Los valores de "start_idx" y "end_idx" se usan para generar los segmentos de línea que conectan los nodos correspondientes 
    segments = [[embedding[:, start], embedding[:, stop]]
                for start, stop in zip(start_idx, end_idx)]
    # El ancho de cada arco está determinado por la magnitud de la correlación parcial entre los nodos correspondientes en la matriz de correlaciones parciales
    values = np.abs(partial_correlations[non_zero])
    # Los valores de values se utilizan para establecer el color de cada línea en la colección de líneas, y se asigna un color de mapa de colores (cmap) para mapear los valores a los colores.   
    lc = LineCollection(segments,
                        zorder=0, cmap=plt.cm.hot_r, 
                        norm=plt.Normalize(0, .7 * val_max))
    lc.set_array(values)
    temp = (15 * values)
    temp2 = np.repeat(5, len(temp))
    w = np.minimum(temp, temp2)
    lc.set_linewidths(w)
    # La variable ax se refiere al objeto de la figura del gráfico donde se agregará la colección de líneas.
    ax.add_collection(lc)
    # Se crea un gráfico de color de barras (colorbar) para mostrar la escala de colores y se agrega una etiqueta para indicar la fuerza de las correlaciones. 
    axcb = fig.colorbar(lc)
    axcb.set_label('Streght')

    # Agrega una etiqueta a cada nodo. Para evitar la superposición de las etiquetas, el código utiliza una técnica de posicionamiento que ajusta la posición de la etiqueta en función de la posición de los nodos vecinos.
    # El código utiliza un bucle for para iterar sobre los nodos del grafo y sus respectivas etiquetas, así como sus coordenadas en el espacio de embedding. El parámetro enumerate se utiliza para obtener el índice del nodo actual en la iteración.
    for index, (name, label, (x, y)) in enumerate(
            zip(names, labels, embedding.T)):
    # Luego, el código calcula la distancia entre el nodo actual y los demás nodos en la dirección "x" y en la dirección "y"
        dx = x - embedding[0]
        dx[index] = 1
        dy = y - embedding[1]
        dy[index] = 1
    # Se encuentra la distancia más cercana a los nodos vecinos (en la dirección "x" o "y") y se ajusta la posición de la etiqueta en esa dirección para evitar la superposición
        this_dx = dx[np.argmin(np.abs(dy))]
        this_dy = dy[np.argmin(np.abs(dx))]
        if this_dx > 0:
            horizontalalignment = 'left'
            x = x + .002
        else:
            horizontalalignment = 'right'
            x = x - .002
        if this_dy > 0:
            verticalalignment = 'bottom'
            y = y + .002
        else:
            verticalalignment = 'top'
            y = y - .002
    # Se agrega la etiqueta a la posición ajustada. Los parámetros "horizontalalignment" y "verticalalignment" se utilizan para alinear la etiqueta con el nodo correspondiente
        plt.text(x, y, name, size=10,
                 horizontalalignment=horizontalalignment,
                 verticalalignment=verticalalignment,
                 bbox=dict(facecolor='w',
                           edgecolor=plt.cm.nipy_spectral(label / float(n_labels)),
                           alpha=.6))
    # Se establece los límites de los ejes de la figura.
    plt.xlim(embedding[0].min() - .15 * embedding[0].ptp(),
             embedding[0].max() + .10 * embedding[0].ptp(),)
    plt.ylim(embedding[1].min() - .03 * embedding[1].ptp(),
             embedding[1].max() + .03 * embedding[1].ptp())
    # Se agrega un título y se muestra la figura resultante.   
    plt.title(title)
    plt.show()
    


    
def graphicalAnalysis_plot_ZOOM_bySector(Sectors_list, plot_config):
    """
    Función utilizada para trazar el grafo de red gráfica para sectores especificos. Puede ser utilizada para ver más detalles del grafo de red.
    """      
    # Se asigna  entrada de la configuración del trazado a distintas variables que entraran a la función "graphycalAnalysis_plot()"
    d = plot_config[0]
    pc = plot_config[1]
    my_colors = np.array(plot_config[2])
    names = plot_config[3]
    labels = plot_config[4]
    embedding = plot_config[5]
    val_max = plot_config[6]
    title = 'ZOOM IN VIEW: ' + plot_config[7]
    
    # Se verifica si se han especificado sectores para mostrar en el gráfico
    if ((not Sectors_list) == False):
    # Se verifica si todos los sectores especificados están presentes en el conjunto de datos "firms_info", iterando sobre la lista de sectores especificados.
        if(all([(s in firms_info.Sector.unique()) for s in Sectors_list])):
            f_in_sector_chosen = []
            for s in Sectors_list:
    # Agregar los índices de las empresas que pertenecen a cada sector a la lista "f_in_sector_chosen"
                f_in_sector_chosen += list(firms_info[firms_info.Sector == s].index)
    # Si no se cumple con la condición de que los sectores especificados están presentes en el conjunto de datos, imprime un mensaje de error
        else:
            print('ERROR: Revision needed! At Least 1 Sector entered in the \"Sectors_choosen\" option is NOT in the dataset!')
            print('Check your format!')
    # Si no se especificaron sectores, imprime un mensaje de error
    else:       
        print('Error: Need to enter the sectors you wanted to examine in the \"Sectors_list\" option!')
        return
    
    # Se seleccionan las empresas que pertenecen a los sectores elegidos previamente
    f_selected = list(set(f_in_sector_chosen).intersection(set(names)))
    
    # Se verifica si hay empresas en los sectores seleccionados. Si no hay sectores, muestra un mensaje de error      
    if(not f_selected):
        print('ERROR: Revision needed! No firms in the selected sectors!')
        print('Check your format!')
        print('Note that the sectors entered in the \"Sectors_list\" option should also be in the \"Sectors_choosen\" option!')
        return
    # Si la lista de sectores no está vacía, se crea una matriz de índices que corresponden a los sectores seleccionados (f_selected). El índice de cada sector en f_selected se busca en names utilizando np.where() y se agrega al array "ind"
    else:
        ind = np.array([np.where(names == i)[0][0] for i in f_selected])

    # Se llama a la función "graphicalAnalysis_plot()" para generar el gráfico de red
    graphicalAnalysis_plot(d[ind], pc[ind[:, None], ind], my_colors[ind],
                           names[ind], labels[ind], embedding[:,ind], val_max, title)

    
    

def graphicalAnalysis_plot_ZOOM_byFirm(firms_list, plot_config):
    """
    Función utilizada para graficar el grafo de red gráfica para empresas específicas. Se puede usar para ver más detalles del grafo de red.
    """
    # Se asigna  entrada de la configuración del trazado a distintas variables que entraran a la función "graphycalAnalysis_plot()"
    d = plot_config[0]
    pc = plot_config[1]
    my_colors = np.array(plot_config[2])
    names = plot_config[3]
    labels = plot_config[4]
    embedding = plot_config[5]
    val_max = plot_config[6]
    title = 'ZOOM IN VIEW: ' + plot_config[7]
    
    # Se verifica si todas las empresas especificadas en la lista están presentes en la base de datos de empresas "names"    
    if( all([(f in names) for f in firms_list]) ):        
    # Si no cumple la condición, se imprime un mensaje de error y se detiene el proceso    
        if (not firms_list):
            print('Error: Need to enter the firms you wanted to examine in the \"firms_list\" option!')
            return
    # Si todas las empresas en la lista están presentes en la base de datos de empresas, el código comprueba si la lista de empresas está vacía utilizando una condición booleana. Si la lista de empresas no está vacía, se crea una matriz de índices que corresponden a las empresas seleccionadas (firms_list). El índice de cada empresa en firms_list se busca en names utilizando np.where() y se agrega al array "ind"
        else:
            ind = np.array([np.where(names == i)[0][0] for i in firms_list])
    # Si la lista de empresas está vacía, se imprime un mensaje de error
    else:
        print('Error: Revision needed! At Least 1 firm entered in the \"firms_list\" are NOT in the dataset!')
        print('Check your format and also whether the firms are dropped due to missing data!')
        return
    
    # Se llama a la función graphicalAnalysis_plot con los datos filtrados para mostrar solo la información correspondiente a las empresas seleccionadas en la lista
    graphicalAnalysis_plot(d[ind], pc[ind[:, None], ind], my_colors[ind],
                           names[ind], labels[ind], embedding[:,ind], val_max, title)
    

    
    
# Función utilizada para selecccionar el activo con el Sharpe Ratio más alto de cada cluster una vez realizado el Análisis Gráfico  
def sortDataBySharpeRatio(data, rf=TBill['T-Bill']):
    ss_temp = getSumStat(data, rf)
    sharpe_order = ss_temp['Sharpe Ratio'].sort_values(ascending=False).index.tolist()
    sorted_columns = data.loc[:, sharpe_order].columns.tolist()
    return sorted_columns


# Función utilizada para selecccionar el activo con la menor volatilidad anualizada de cada cluster  una vez realizado el Análisis Gráfico (empleada solo para un caso en concreto)                                        
def sortDataByVolatility(data):
    ss_temp = getSumStat(data)
    vol_order = ss_temp['Annu. Vol(%)'].sort_values(ascending=False).index.tolist()
    sorted_columns = data.loc[:, vol_order].columns.tolist()
    return sorted_columns     
        
        
# Cargar conjunto de datos de información de la tasa libre de riesgo (TBill) para summary Stats de tal forma que la serie temporal para la tasa libre de riesgo coincida con la longitud del Back Test
# Leer y editar datos en el formato correcto
file_name2 = 'WTB3MS.csv'
df2 = pd.read_csv(file_name2, delimiter='\t',skiprows=1)

# Convertir la columna 'DATE' del dataframe df en formato datetime
df2['DATE']=pd.to_datetime(df2['DATE'])
# Convertir la columna 'WTB3MS' del dataframe df a un tipo de datos numérico, cualquier valor no numérico se convertirá en NaN
df2['WTB3MS'] = pd.to_numeric(df2['WTB3MS'], errors='coerce')

# Crear un nuevo dataframe con una única columna 'RET' y los mismos índices que df.DATE
RET_data2=pd.DataFrame(columns=['RET'], index=df2.DATE)
# Calcular los retornos semanales 
RET_data2['RET'] = (df2['WTB3MS'].values/100 + 1) ** (1/52) - 1 

RET_data_weekly2 = (RET_data2['RET']+1).resample('W').prod() - 1 # Convertir los datos a la misma frecuencia semanal que arriba
# Crear un nuevo dataframe con una única columna 'T-Bill' y los mismos índices que RET_data_weekly.
TBill2 = pd.DataFrame(columns=['T-Bill'], index=RET_data_weekly2.index)
# establecer el número de semanas para retroceder
num_weeks = 51
# establecer la variable 'start_date' para ser 156 semanas antes del primer índice en 'data'
start_date2 = data.index[0] + pd.Timedelta(weeks=num_weeks)
end_date2 = data.index[-1]
# Para igualar la fecha de inicio de los datos de retorno de acciones.
TBill2 = TBill2[TBill2.index > start_date2] 
# Para igualar la fecha de finalización de los datos de retorno de acciones.
TBill2 = TBill2[TBill2.index <= end_date2]   
# Calcular los retornos semanales de los datos de la columna 'RET' y asignarlos a la columna 'T-Bill'
TBill2['T-Bill'] = RET_data_weekly2  

        
    
    
def summary_stats(r, rf=TBill2['T-Bill'], rounding = 2):
    """
    Devuelve un DataFrame que contiene estadísticas resumidas agregadas para los rendimientos en las columnas de 'data'
    """
    ann_r = np.round(r.aggregate(annualize_rets, periods_per_year = 52), rounding)
    ann_vol = np.round(r.aggregate(annualize_vol, periods_per_year = 52), rounding)
    ann_sr = np.round(r.aggregate(sr_annu, rf_old=rf), rounding)
    dd = np.round(r.aggregate(mdd), rounding)
    skew = np.round(r.aggregate(skewness), rounding)
    kurt = np.round(r.aggregate(kurtosis), rounding)
    cf_var5 = np.round(r.aggregate(var_gaussian, modified=True), rounding)
    hist_cvar5 = np.round(r.aggregate(cvar_historic), rounding)
    return pd.DataFrame({
        "Annu. Ave Return(%)": ann_r,
        "Annu. Vol(%)": ann_vol,
        "Skewness": skew,
        "Kurtosis": kurt,
        "Cornish-Fisher VaR (5%)": cf_var5,
        "Historic CVaR (5%)": hist_cvar5,
        "Sharpe Ratio": ann_sr,
        "Max Drawdown(%)": dd}) 
 
    
    
### Modelos de Ponderación
### Equally Weighted
def weight_ew(data):
    """
    Devuelve los pesos del portafolio EW basados en los rendimientos de los activos "r" como una serie
    """
    # Determina el número de columnas en el DataFrame "data"
    n = len(data.columns)
    # crea una Serie de pandas (estructura de datos similar a un array o lista) donde cada activo tiene la misma ponderación en  el portafolio.
    return pd.Series(1/n, index=data.columns)  




### Maximum Sharpe Ratio optimization
def portfolio_return(weights, returns):
    """
    Calcula el retorno de una cartera a partir de los retornos y pesos de los constituyentes. Los pesos son una matriz de numpy de tamaño Nx1 y los rendimientos son una matriz de numpy de tamaño Nx1.
    """
    # Se transpone la matriz de pesos y luego el operador "@" realiza una multiplicación matricial entre la matriz transpuesta y la matriz de retornos
    return weights.T @ returns


def portfolio_vol(weights, covmat):
    """
    Calcula la volatilidad de una cartera a partir de una matriz de covarianza y los pesos de los constituyentes. Los pesos son una matriz de numpy de tamaño Nx1, y la matriz de covarianza es de tamaño NxN.
    La volatilidad mide la variabilidad de los rendimientos de la cartera en relación con su media
    """
    # Se realiza una operación de multiplicación matricial entre la matriz de pesos transpuesta, la matriz de covarianza y la matriz de pesos (resulta en un valor escalar que representa la varianza de la cartera). A partir de ese valor, se calcular la raíz cuadrada para obtener la desviación estándar de la cartera que se almacena en la variable "vol"
    vol = np.sqrt(weights.T @ covmat @ weights)
    # Se devulve como resultado de la función 
    return vol 


# Este código devuelve una asignación de peso optimizada para una cartera utilizando el Sharpe Ratio Máximo de la Teoría Moderna de Portafolios 
def weight_msr(er, cov, rf=TBill['T-Bill']):
    """
    Devuelve los pesos de la cartera que proporciona el máximo sharpe ratio, dados la tasa libre de riesgo "rf", los retornos esperados "er" y una matriz de covarianza "cov"
    """
    # Se extraen el número de filas de la matriz "er"
    n = er.shape[0]
    # Se crea una suposición inicial para la asignación de peso donde cada activo "n" tiene el mismo peso
    init_guess = np.repeat(1/n, n)
    # Se configura los límites para que los pesos se encuentren entre 0 y 0.15 para todos los activos "n"
    bounds = ((0.0, 0.15),) * n 
    # Se crea una restricción para garantizar que la suma de los pesos de los activos de la cartera sea igual a 1
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }  
    def neg_sharpe(weights, er, cov, rf=TBill['T-Bill']):
        """
        Devuelve el negativo del sharpe ratio de la cartera dada.
        Al devolver el negativo del ratio de Sharpe, se busca maximizar el ratio de Sharpe durante la optimización de la cartera, es decir, minimizar el riesgo y maximizar la rentabilidad en relación al riesgo.
        """  
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        rf_last_year = rf.tail(52).mean()
        sharpe = (r - rf_last_year) / vol
        return -sharpe
    
    # Se llama al método de optimización minimize con la función neg_sharpe como la función objetivo y la suposición inicial, la tasa libre de riesgo, los retornos esperados y la matriz de covarianza como argumentos. El método de optimización utilizado es Programación Secuencial de Mínimos Cuadrados (SLSQP)
    weights = minimize(neg_sharpe, init_guess,
                       args=(er, cov, rf), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,),
                       bounds=bounds)
    # La función devuelve la asignación de peso optimizada para la cartera
    return weights.x




### Minimum Global Variance
def sample_cov(r, **kwargs):
    """
    Devuelve la covarianza muestral de los rendimientos 
    """
    return r.cov()


def gmv(cov):
    """
    Devuelve los pesos de la cartera de Mínima de Volatilidad Global dada una matriz de covarianza
    """
    n = cov.shape[0]
    return weight_msr(np.repeat(1, n), cov, rf=TBill['T-Bill'])

# Este código devuelve una asignación de peso optimizada para una cartera utilizando la Varianza Global Mínima de la Teoría Moderna de Portafolios 
def weight_gmv(r, cov_estimator=sample_cov, **kwargs):
    """
    Produce los pesos de la cartera de Mínima de Volatilidad Global (GMV) dada una matriz de covarianza de los rendimientos
    """
    est_cov = cov_estimator(r, **kwargs)
    return gmv(est_cov)




# Pruebas retrospectivas (backtests) en un esquema de ponderación dado (weighting scheme)
def backtest_ws(data, estimation_window=52, weighting=weight_ew, verbose=False, **kwargs):
    """
    Realiza pruebas retrospectivas (backtests) en un esquema de ponderación dado (weighting scheme), con algunos parámetros:
    data: rendimientos de activos a utilizar para construir el portafolio
    estimation_window: la ventana a utilizar para estimar los parámetros
    weighting: el esquema de ponderación a utilizar, debe ser una función que tome "data" y un número variable de argumentos de    palabras clave y valores.
    **kwargs: cualquier argumento de palabras clave adicional que se hayan proporcionado
    """
    # Determina el número de períodos en el conjunto de datos "data"
    n_periods = data.shape[0]
    # Crea una lista de tuplas de ventanas de tiempo para estimar los pesos del portafolio en cada ventana de tiempo.
    windows = [(start, start+estimation_window) for start in range(n_periods-estimation_window)]
    # Utiliza una función de ponderación (weighting) para calcular los pesos del portafolio para cada ventana de tiempo
    weights = [weighting(data.iloc[win[0]:win[1]], **kwargs) for win in windows]
    # Transforma la lista de pesos en un Dataframe de pandas con los mismos índices que los rendimientos de los activos
    weights = pd.DataFrame(weights, index=data.iloc[estimation_window:].index, columns=data.columns)
    # Calcula los rendimientos del portafolio para cada período multiplicando los pesos del portafolio por los rendimientos de los activos y sumando a través de las columnas
    returns = (weights * data).sum(axis="columns",  min_count=1) # mincount genera valores 'NAs' si todas las entradas son 'NAs
    # Devuelve los rendimientos del portafolio como una Serie de pandas
    return returns
