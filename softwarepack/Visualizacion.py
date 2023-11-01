# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 22:51:19 2023

@author: Josu
"""

import pandas as pd
import numpy as np
import sys
from softwarepack.Metricas import *
from .Metricas import __curvaROC
from softwarepack.Correlacion import *
from softwarepack.Discretizacion import *
import matplotlib.pyplot as plt
import seaborn as sns



def plotROC(df):
    """
    Dado un dataFrame con una columna numerica y una boolear visualiza su curva ROC
    
    Args:
        df (dataFrame): El dataFrame
    """
    # Se garantiza que df es un dataFrame
    if not isinstance(df, pd.DataFrame):
        sys.exit("df debe ser un DataFrame")
    # Se garantiza que el dataFrame tiene una columna numerica y una logica
    if not (df.shape[1] == 2 and pd.api.types.is_numeric_dtype(df.iloc[:, 0]) and pd.api.types.is_bool_dtype(df.iloc[:, 1])):
        sys.exit("El df de entrada debe tener dos columnas, la primera numérica y la segunda lógica")
    # Se calcula la curva ROC
    puntos = __curvaROC(df)
    # Se calcula el AUC
    AUC = calculoAUC(df)
    # Se consiguen las coordenadas x e y
    x = [elem[0] for elem in puntos]
    y = [elem[1] for elem in puntos]
    # Se visualiza la curva ROC
    plt.plot(x,y,color='red', linewidth=2)
    plt.title('Curva ROC (' + str(round(AUC,2)) + ")")
    plt.xlabel('FPR')
    plt.ylabel('TPR')


def plotCorrelaciones(df):
    """
    Dado un dataFrame calcula la correlacion por pares entre variables
    
    Args:
        df (dataFrame): El dataFrame
    """
    # Se garantiza que df es un dataFrame
    if not isinstance(df, pd.DataFrame):
        sys.exit("df debe ser un DataFrame")
    # Se garantiza que en el dataFrame no hay ninguna columna logica
    if (len([col for col in df.columns if pd.api.types.is_bool_dtype(df[col])]) > 0):
        sys.exit("Las variables no pueden ser logicas")
    # Se calcula la matriz de correlaciones
    correlaciones = correlacionDF(df)
    # Se visualiza un heatmap utilizando la matriz de correlaciones
    sns.heatmap(correlaciones, annot=True, cmap='coolwarm', linewidths=0.5) 
    plt.title('Matriz de correlaciones')
    plt.show()

def plotBoxPlots(df):
    """
    Dado un dataFrame con variables numericas visualiza todos los boxplots de las variables
    
    Args:
        df (dataFrame): El dataFrame
    """
    # Se garantiza que df es un dataFrame
    if not isinstance(df, pd.DataFrame):
        sys.exit("df debe ser un DataFrame")
    # Se garantiza que todas las variables son numericas
    if not all([pd.api.types.is_numeric_dtype(df[col]) for col in df.columns]):
        sys.exit("Todas las variables deben ser numéricas")
    # Se visualizan los boxplots    
    data = pd.melt(df)
    sns.boxplot(x="variable", y="value", data=data)
    plt.xlabel("Variables")
    plt.ylabel("")
    plt.title("BoxPlot")
    plt.xticks(rotation=45)
    plt.show()
    

def plotBarrasDiscretizadas(x, numBins, metodo = "EW"):
    """
    Dado un vector numerico, un numero de intervalos y un metodo de discretizacion,
    discretiza el vector y muestra la frecuencia de todos los intervalos
    
    Args:
        df (dataFrame): El dataFrame
    """
    aux = np.array(x)
    # Se garantiza que el vector x es numerico
    if not np.issubdtype(aux.dtype, np.number):
        sys.exit("x debe ser númerico")
    # Se garantiza que el numero de intervalos es un unico numero entero
    if not isinstance(numBins, int) or numBins <= 0:
        sys.exit("numBins debe ser un número entero positivo")
    # Se discretiza el vector numerico segun el metodo indicado
    if metodo == "EF":
        xDiscretized, _ = discretizeEF(aux, numBins)
    elif metodo == "JNB":
        xDiscretized, _ = discretizeJNB(aux, numBins)
    else:
        xDiscretized, _ = discretizeEW(aux, numBins)
    # Se calculan las frecuencias
    table = xDiscretized.value_counts()
    # Se visualiza el diagrama de barras
    plt.bar(table.index, table.values, color="blue")
    plt.xlabel("Intervalo")
    plt.ylabel("Frecuencia")   
    plt.title("Vector X discretizado")
    
