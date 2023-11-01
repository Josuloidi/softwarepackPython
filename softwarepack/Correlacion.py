# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 22:15:37 2023

@author: Josu
"""

import pandas as pd
import numpy as np
import math
import sys
from softwarepack.Metricas import entropy



def correlacion(x, y):
    """
    Dado dos vectores calcula la correlacion si los dos vectores son numericos, 
    y la informacion mutua si uno de los dos vectores es categorico
    
    Args:
        x (lista): El primer vector
        y (lista): El segundo vector
    
    Returns:
        float: La correlacion o informacion mutua
    """
    aux1 = np.array(x)
    aux2 = np.array(y)
    # Se garantiza que los dos vectores tienen la misma longitud
    if (len(x) != len(y)):
        sys.exit("Los dos vectores deben tener la misma longitud")
    # Se garantiza que ningun vector es logico
    if (aux1.dtype == "bool" or aux2.dtype == "bool"):
        sys.exit("Las variables no pueden ser logicas")
    # Si los dos son numericos
    if (np.issubdtype(aux1.dtype, np.number) and np.issubdtype(aux2.dtype, np.number)):
        # Se computa la formula de la correlacion
        res = sum((aux1 - np.mean(aux1)) * (aux2 - np.mean(aux2))) / math.sqrt( sum((aux1 - np.mean(aux1))**2) * sum((aux2 - np.mean(aux2))**2)) 
    else:
        # Se computa la formula de la entropia
        res = entropy(aux1) + entropy(aux2) - entropy([str(aux1[i])+str(aux2[i]) for i in range(len(aux1))])
    return(res)    
        
def __correlacionUnaVariable(x, df):
    """
    Dado un vector x y un dataFrame, calcula las correlaciones que mantiene x 
    con todas las columnas del dataFrame
    
    Args:
        x (lista): El vector
        df (dataFrame): El dataFrame
    
    Returns:
        lista numerica: Las correlaciones que mantiene x con todas las columnas del dataFrame
    """
    # Para cada columna del df se calcula la correlacion con x
    return(df.apply(lambda y: correlacion(x, y)))

def correlacionDF(df):
    """
    Dado un dataFrame calcula la correlacion por pares entre variables
    
    Args:
        df (dataFrame): El dataFrame
    
    Returns:
        dataFrame: Las matriz de correlaciones
    """
    # Se garantiza que df es un dataFrame
    if not isinstance(df, pd.DataFrame):
        sys.exit("df debe ser un DataFrame")
    # Se garantiza que ninguna columna del dataFrame es logica
    if (len([col for col in df.columns if pd.api.types.is_bool_dtype(df[col])]) > 0):
        sys.exit("Las variables no pueden ser logicas")
    # Por cada columna del dataFrame, se calcula la correlacion que mantiene con todo el dataFrame
    return(df.apply(lambda x: __correlacionUnaVariable(x, df)))
