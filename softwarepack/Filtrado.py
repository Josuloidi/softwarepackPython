# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 19:37:40 2023

@author: Josu
"""

import pandas as pd
import numpy as np
import sys
from softwarepack.Metricas import *


def __correlacionUnaVariable(i, df):
    """
    Dado un dataFrame y un indice de columna, garantiza que la iesima columna 
    es de tipo correcto
    
    Args:
        df (dataFrame): El dataFrame
        i (int): indice de columna
    
    Returns:
        bool: Indicativo de columna correcta
    """
    # Se consigue la columna
    x = np.array(df.iloc[:,i])
    # Si la columna es boolear
    if (np.issubdtype(x.dtype, np.bool_)):
        # Se garantiza que por delante tiene una columna numerica
        if (i == 0 or not np.issubdtype(np.array(df.iloc[:,i-1]).dtype, np.number)):
            return(False)
    return(True)
    
def __calculoAUCColumna(i, df):
    """
    Dado un dataFrame y un indice de columna, calcula el AUC de la columna
    
    Args:
        df (dataFrame): El dataFrame
        i (int): indice de columna
    
    Returns:
        float: El valor AUC de la columna
    """
    # Si la columna es boolear
    if (np.issubdtype(np.array(df.iloc[:,i]).dtype, np.bool_)):
        # Se calcula el AUC utilizando la columna anterior
        return(calculoAUC(df.iloc[:,i-1:i+1]))
    # Si la columna es numerica
    elif (np.issubdtype(np.array(df.iloc[:,i]).dtype, np.number)):
        # Si esta seguida por una columna boolear
        if (i != (df.shape[1] - 1) and np.issubdtype(np.array(df.iloc[:,i+1]).dtype, np.bool_)):
            # Se calcula el AUC utilizando la columna siguiente
            return (calculoAUC(df.iloc[:,i:i+2]))
    # Si no, se devuelve -1
    return(-1)

def __calculoVarColumna(i, df):
    """
    Dado un dataFrame y un indice de columna, calcula la varianza de la columna
    
    Args:
        df (dataFrame): El dataFrame
        i (int): indice de columna
    
    Returns:
        float: La varianza de la columna
    """
    # Si la columna es boolear
    if (np.issubdtype(np.array(df.iloc[:,i]).dtype, np.bool_)):
        # Se calcula la varianza de la columna anterior
        return(varianza(df.iloc[:,i-1]))
    # Si es numerica
    elif (np.issubdtype(np.array(df.iloc[:,i]).dtype, np.number)):
        # Se calcula la varianza de la columna
        return (varianza(df.iloc[:,i]))
    # Si no, se devuelve -1
    return(-1)



def filtrar(df, varianza = False, AUC = False, ent = False,
                varUmbr = 5, AUCUmbr = 0.5, entUmbr = 0.5,
                varOP = "EQ", AUCOP = "EQ", entOP = "EQ"):
    """
    Dado un dataFrame filtra las variables segun la entropia, el AUC o la varianza
    
    Args:
        df (dataFrame): El dataFrame
        varianza (bool): Un valor logico que indica si se quiere filtrar por la varianza
        AUC (bool): Un valor logico que indica si se quiere filtrar por el AUC
        ent (bool): Un valor logico que indica si se quiere filtrar por la entropia
        varUmbr (float): El umbral de la varianza
        AUCUmbr (float): El umbral del AUC
        entUmbr (float): El umbral de la entropia
        varOP (str): El operador que se utilizara al comparar la varianza
        AUCOP (str): El operador que se utilizara al comparar el AUC
        entOP (str): El operador que se utilizara al comparar la entropia
    
    Returns:
        dataFrame: El dataFrame filtrado
    """
    # Se garantiza que es un dataFrame
    if not isinstance(df, pd.DataFrame):
        sys.exit("df debe ser un DataFrame")
    # Se garantiza que todas las columnas son de tipo correcto
    if not all([__correlacionUnaVariable(i,df) for i in range(df.shape[1])]):
        sys.exit("Las variables logicas solo pueden ir después de una variable numérica")
    # En un principio, todas las variables pasan el filtro
    resultados = np.array([True] * df.shape[1])
    # Si se quiere filtrar por la entropia
    if ent:
        # Se consiguen los indices de las columnas que no sean numericas ni logicas
        categoricos = df.select_dtypes(exclude=[int, float, bool]).columns
        indicesCat = [df.columns.get_loc(col) for col in categoricos]
        # Las columnas numericas y logicas pasan el filtro, los otros por ahora no
        resultadosEnt = np.array([True] * df.shape[1])
        resultadosEnt[indicesCat] = False
        # Se calculan las entropias de las columnas categoricas
        entropias = [entropy(df.iloc[:, i]) if i in indicesCat else -1 for i in range(df.shape[1])]
        # Se ensenan las entropias
        print("Entropias: ", entropias)
        # Se hace la comparacion. Si cumplen la condicion, pasan el filtro
        if entOP == "LO":
            comparacion = np.array([elem < entUmbr for elem in entropias])
        elif entOP == "HI":
            comparacion = np.array([elem > entUmbr for elem in entropias])
        else:
            comparacion = np.array([elem == entUmbr for elem in entropias])
        # Se actualizan los resultados generales
        resultados = resultados & (comparacion | resultadosEnt)
    # Si se quiere filtrar por el AUC
    if AUC:
        # Se calcula el valor AUC de las columnas numericas que van seguidas por una logica
        valoresAUC = [__calculoAUCColumna(i, df) for i in range(df.shape[1])]
        # Se enseñan los valores AUC
        print("Valores AUC: ", valoresAUC)
        # Las columnas categoricas pasan el filtro, los otros por ahora no
        resultadosAUC = np.array([elem == -1 for elem in valoresAUC])
        # Se hace la comparacion. Si cumplen la condicion, pasan el filtro
        if AUCOP == "LO":
            comparacion = np.array([elem < AUCUmbr for elem in valoresAUC])
        elif AUCOP == "HI":
            comparacion = np.array([elem > AUCUmbr for elem in valoresAUC])
        else:
            comparacion = np.array([elem == AUCUmbr for elem in valoresAUC])
        # Se actualizan los resultados generales
        resultados = resultados & (comparacion | resultadosAUC)
    if varianza:
        # Se calcula la varianza de las columnas numericas
        varianzas = [__calculoVarColumna(i, df) for i in range(df.shape[1])]
        # Se enseñan las varianzas
        print("Varianzas: ", varianzas)
        # Las columnas categoricas pasan el filtro, los otros por ahora no
        resultadosVar = np.array([elem == -1 for elem in varianzas])
        # Se hace la comparacion. Si cumplen la condicion, pasan el filtro
        if varOP == "LO":
            comparacion = np.array([elem < varUmbr for elem in varianzas])
        elif varOP == "HI":
            comparacion = np.array([elem > varUmbr for elem in varianzas])
        else:
            comparacion = np.array([elem == varUmbr for elem in varianzas])
        # Se actualizan los resultados generales
        resultados = resultados & (comparacion | resultadosVar)
    # Se devuelven solo las columnas que han pasado el filtro
    return(df.iloc[:,resultados])
        
















