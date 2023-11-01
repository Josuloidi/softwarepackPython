# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 15:49:51 2023

@author: Josu
"""
import pandas as pd
import numpy as np
import math
import sys


def discretize(x, cutPoints):
    """
    Esta función toma un vector numérico y una lista de puntos de corte y
    devuelve un vector categórico con los valores discretizados utilizando 
    esos puntos de corte.
    
    Args:
        x (lista de números): El vector numérico
        cutPoints (lista de números): Lista de puntos de corte
    
    Returns:
        vector categórico: Vector categórico con los valores discretizados
    """
    # Se comprueba que los dos vectores de entrada son numéricos
    if (not (np.issubdtype(np.array(x).dtype, np.number)) or not (np.issubdtype(np.array(cutPoints).dtype, np.number))):
        sys.exit("x y cutPoints deben tener valores numéricos")
    # A la lista de los puntos de corte se le añaden minus infinito al principio y infinito al final. Para poder definir
    # el primer y último rango.
    cutPointsAux = [float('-Inf')] + list(cutPoints) + [float("Inf")]
    # Se crean los nombres de todos los rangos siguiendo este formato: (a,b]
    levels = ["(" + str(round(cutPointsAux[i],2)) + "," + str(round(cutPointsAux[i+1], 2)) + "]" for i in range(len(cutPointsAux)-1)]
    # Al último nombre se le cambia el corchete por un parentesis: (a,inf] -> (a,inf)
    levels[-1] = levels[-1].replace("]", ")")
    # Gracias a la funcion cut de pandas, se consigue el rango al que pertenece cada numero de x
    xDiscreticed = pd.cut(x, bins=cutPointsAux, labels=levels, right=True)
    return(xDiscreticed)

def discretizeEW(x, numBins):
    """
    Esta función toma un vector de tipo numérico y un número de intervalos. 
    Como resultado la función da dos cosas, un vector de valores categóricos
    resultado de aplicar el algoritmo Equal Width y la lista de puntos de corte.
    
    Args:
        x (lista de números): El vector numérico
        numBins (int): El número de intervalos
    
    Returns:
        vector categórico: Vector categórico con los valores discretizados
        list: La lista de puntos de corte
    """
    # Se comprueba que la lista x contiene números
    if not (np.issubdtype(np.array(x).dtype, np.number)):
        sys.exit("x debe tener valores numéricos")
    # Se comprueba que numBins es un número entero
    if not isinstance(numBins, int):
        sys.exit("numBins debe ser un número entero")
    # Se calculan los puntos de corte. Desde el valor minimo de X, hasta el más grande
    # con el salto de (max - min)/numBins. Eso sí, se descartan el último y el primer 
    # valor.
    cutPoints = np.arange(min(x), max(x), (max(x) - min(x)) / numBins)[1:]
    # Se discretizan los números
    xDiscretized = discretize(x, cutPoints)
    return(xDiscretized, cutPoints)

def discretizeEF(x, numBins):
    """
    Esta función toma un vector de tipo numérico y un número de intervalos. 
    Como resultado la función da dos cosas, un vector de valores categóricos
    resultado de aplicar el algoritmo Eequal Frecuency y la lista de puntos de corte.
    
    Args:
        x (lista de números): El vector numérico
        numBins (int): El número de intervalos
    
    Returns:
        vector categórico: Vector categórico con los valores discretizados
        list: La lista de puntos de corte
    """
    # Se comprueba que la lista x contiene números
    if not (np.issubdtype(np.array(x).dtype, np.number)):
        sys.exit("x debe tener valores numéricos")
    # Se comprueba que numBins es un número entero
    if not isinstance(numBins, int):
        sys.exit("numBins debe ser un número entero")
    # La cantidad de numéros que habrá como mínimo en cada rango
    frecuency = len(x) // numBins 
    # La cantidad de números que habrá en los primeros rangos en el caso de 
    # que la cantidad de números no sea multiplo de numBins
    frecuency1 = math.ceil(len(x) / numBins)
    # Si la cantidad de números es múltiplo de numBins
    if (frecuency == frecuency1):
        # En cada rango habrá la misma cantidad de números, por lo que los indices
        # de estos números mantendrán la misma frecuencia 
        cutInd = np.arange(frecuency, len(x), frecuency) - 1
    else:
        # En los primeros rangos habrá más números, por lo que los indices seguirán
        # esta frecuencia y los siguientes una distinta.
        centro = len(x) % numBins * frecuency1
        cutInd = np.concatenate([np.arange(frecuency1, centro + 1, frecuency1), np.arange(centro + frecuency, len(x), frecuency)]) - 1
    # Se ordenan todos los números y se toman los cutPoints que indican los indices
    cutPoints = np.sort(x)[cutInd]
    # Se discretizan los números
    xDiscretized = discretize(x, cutPoints)
    return(xDiscretized, cutPoints)  

def discretizeJNB(x, numBins):
    """
    Esta función toma un vector de tipo numérico y un número de intervalos. 
    Como resultado la función da dos cosas, un vector de valores categóricos
    resultado de aplicar el algoritmo Jenks Natural Breaks y la lista de 
    puntos de corte.
    
    Args:
        x (lista de números): El vector numérico
        numBins (int): El número de intervalos
    
    Returns:
        vector categórico: Vector categórico con los valores discretizados
        list: La lista de puntos de corte
    """
    # Se comprueba que la lista x contiene números
    if not (np.issubdtype(np.array(x).dtype, np.number)):
        sys.exit("x debe tener valores numéricos")
    # Se comprueba que numBins es un número entero
    if not isinstance(numBins, int):
        sys.exit("numBins debe ser un número entero")
    # Se ordena el vector de números
    ordenado = np.sort(x)
    # Se calculan las diferencias entre valores
    diferencias = np.diff(ordenado)
    # Se toman las las mayores numBins-1 diferencias
    diferenciasMaximas = np.sort(diferencias)[::-1][:numBins-1]
    # Los cutPoints son los puntos en los que las direcias son máximas
    cutPoints = ordenado[np.where(np.isin(diferencias,diferenciasMaximas))][:numBins-1]
    # Se discretizan los números
    xDiscretized = discretize(x, cutPoints)
    return(xDiscretized, cutPoints)  


def discretizarColumnas(df, numBins, discretizacion = "TODAS"):
    """
    Esta función toma un DataFrame que contiene variables numéricas, un número de intervalos
    y una discretización (EW, EF, JNB o todas). Como salida da un DataFrame que contiene la 
    discretización de todas las variables.
    
    Args:
        df (DataFrame): El DataFrame con variables numéricas
        numBins (int): El número de intervalos
        discretizacion (str): Qué discretización se desea hacer (EW, EF, JNB o TODAS)
    Returns:
        DataFrame: DataFrame que contiene todas las discretizaciones
    """
    #Se comprueba que numBins es un número entero
    if not isinstance(numBins, int):
        sys.exit("numBins debe ser un número entero")
    #Se comprueba que df es un DataFrame
    if not isinstance(df, pd.DataFrame):
        sys.exit("df debe ser un DataFrame")
    #Se comprueba que el DataFrame solo tiene variables numéricas
    if len(df.select_dtypes(include=['number']).columns) != len(df.columns):
        sys.exit("Todas las variables de df deben ser numéricas")
    #Si se quiere hacer Equal Width
    if discretizacion == "EW":
        #Con cada columna se le llama a la funcion discretizeEW y se toma el primer resultado (el vector categórico)
        df = df.apply(lambda x: discretizeEW(x, numBins)[0])
    #Si se quiere hacer Equal Frecuency
    elif discretizacion == "EF":
        #Con cada columna se le llama a la funcion discretizeEF y se toma el primer resultado (el vector categórico)
        df = df.apply(lambda x: discretizeEF(x, numBins)[0])
    #Si se quiere hacer Equal Width
    elif discretizacion == "JNB":
        #Con cada columna se le llama a la funcion discretizeJNB y se toma el primer resultado (el vector categórico)
        df = df.apply(lambda x: discretizeJNB(x, numBins)[0])
    #Si se quiere hacer Jenks Natural Breaks
    else:
        #Con cada columna se le llama a la funcion discretizeEW, discretizeEF y discretizeJNB y se juntan los tres vectores categoricos en una columna
        df = df.apply(lambda x: pd.Series(list(zip(discretizeEW(x, numBins)[0],discretizeEF(x, numBins)[0],discretizeJNB(x, numBins)[0]))))
    return df












































