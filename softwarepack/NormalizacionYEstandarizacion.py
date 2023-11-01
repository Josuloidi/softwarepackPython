# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 19:12:49 2023

@author: Josu
"""

import pandas as pd
import numpy as np
import sys

def estandarizar(x):
    """
    Dado un vector de tipo numerico estandariza dicho vector
    
    Args:
        x (lista de numeros): El vector numérico
    
    Returns:
        vector numerico: El vector estandarizado
    """
    aux = np.array(x)
    # Se garantiza que es un vector numerico
    if not np.issubdtype(aux.dtype, np.number):
        sys.exit("x debe ser numérico")  
    # Se computa la formula de estandarizacion
    return((aux - np.mean(aux))/np.std(aux))

def normalizar(x):
    """
    Dado un vector de tipo numerico normaliza dicho vector
    
    Args:
        x (lista de numeros): El vector numérico
    
    Returns:
        vector numerico: El vector normalizado
    """
    aux = np.array(x)
    # Se garantiza que es un vector numerico
    if not np.issubdtype(aux.dtype, np.number):
        sys.exit("x debe ser numérico")   
    # Se computa la formula de normalizacion
    return((aux - min(aux))/(max(aux) - min(aux)))


def estandarizarDF(df):
    """
    Dado un dataFrame con columnas numericas estandariza dicho dataFrame
    
    Args:
        df (dataFrame): El dataFrame con columnas numericas
    
    Returns:
        dataFrame: El dataFrame estandarizado
    """
    # Se garantiza que es un dataFrame
    if (not isinstance(df, pd.DataFrame)):
        sys.exit("df debe ser un DataFrame")
    # Se garantiza que todas las columnas son numericas
    if (len(df.select_dtypes(include=[int, float]).columns) != df.shape[1]):
        sys.exit("Todas las columnas de df deben ser numéricas")
    # Se estandariza cada columna
    return(df.apply(lambda x: estandarizar(x)))


def normalizarDF(df):
    """
    Dado un dataFrame con columnas numericas normaliza dicho dataFrame
    
    Args:
        df (dataFrame): El dataFrame con columnas numericas
    
    Returns:
        dataFrame: El dataFrame normalizado
    """
    # Se garantiza que es un dataFrame
    if (not isinstance(df, pd.DataFrame)):
        sys.exit("df debe ser un DataFrame")
    # Se garantiza que todas las columnas son numericas
    if (len(df.select_dtypes(include=[int, float]).columns) != df.shape[1]):
        sys.exit("Todas las columnas de df deben ser numéricas")
    # Se normaliza cada columna
    return(df.apply(lambda x: normalizar(x)))




