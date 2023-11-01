# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 16:59:56 2023

@author: Josu
"""
import pandas as pd
import numpy as np
import sys



def entropy(xDiscretized):
    """
    Dado un vector de tipo numerico o categorico calcula la entropia
    
    Args:
        xDiscretized (vector categórico o numerico): El vector numerico
    
    Returns:
        número: La entropia
    """
    # El vector se convierte a numpy.array
    aux = np.array(xDiscretized)
    # Se garantiza que el vector no es logico
    if (aux.dtype == bool):
        sys.exit("xDiscretized no puede ser lógico")
    # Se calculan el numero de ocurrencias de cada elemento y se dividen por la longitud de la lista. 
    # Es decir, se calculan las probabilidades    
    p = np.unique(np.array(aux), return_counts=True)[1] / len(aux)
    # Con cada probabilidad se calcula -pi*log2(pi)
    entr = -p * np.log2(p)
    # Se suman todos los valores
    entr = sum(entr)
    return(entr)




def __TPRyFDR(valorCorte, df):
    """
    Dado un vector de tipo numerico o categorico calcula la entropia
    
    Args:
        xDiscretized (dataFrame): Dado un dataFrame con una columna numerica y una boolear y un valor de corte, calula el TPR y FPR
    
    Returns:
        numero: FPR
        numero: TPR
    """
    # Se garantiza que es un dataFrame
    if not isinstance(df, pd.DataFrame):
        sys.exit("df debe ser un DataFrame")
    # Se garantiza que la primera columna es numerica y la segunda logica
    if not (df.shape[1] == 2 and pd.api.types.is_numeric_dtype(df.iloc[:,0]) and pd.api.types.is_bool_dtype(df.iloc[:,1])):
        sys.exit("df tener dos columnas. La primera columna debe ser numerica y la segunda logica")
    # Se garantiza que valorCorte es un unico numero
    if not isinstance(valorCorte, (int, float)):
        sys.exit("El valor de corte debe ser un numero")
    # Se calcula la prediccion del modelo. Cuando el valor es menor o igual que el valor de corte devuelve TRUE, si no FALSE
    prediccion = df.iloc[:,0] <= valorCorte
    # Se calculan TP, FP, FN y FP contando cuantas veces ocurre cada convinacion
    TP = sum(df.iloc[:, 1] & prediccion)  
    TN = sum(~ df.iloc[:, 1] & ~ prediccion)  
    FP = sum(~ df.iloc[:, 1] & prediccion) 
    FN = sum(df.iloc[:, 1] & ~ prediccion) 
    # Se calculan TPR y FPR
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    return(FPR, TPR)


def __curvaROC(df):
    """
    Dado un dataFrame con una columna numerica y una boolear calcula los puntos de su curva ROC
    
    Args:
        df (dataFrame): Un dataFrame con una columna numerica y una boolear
    
    Returns:
        lista: Lista con todos los valores FPR y TPR
    """
    # Se garantiza que es un dataFrame
    if not isinstance(df, pd.DataFrame):
        sys.exit("df debe ser un DataFrame")
    # Se garantiza que la primera columna es numerica y la segunda logica
    if not (df.shape[1] == 2 and pd.api.types.is_numeric_dtype(df.iloc[:, 0]) and pd.api.types.is_bool_dtype(df.iloc[:, 1])):
        sys.exit("El df de entrada debe tener dos columnas, la primera numérica y la segunda lógica") 
    # Por cada numero del dataFrame, se calculan el TPR Y FPR
    # Solo se toman estos valores ya que son los unicos que cambian los valores.
    return([__TPRyFDR(valor, df) for valor in sorted(df.iloc[:,0])])


def calculoAUC(df):
    """
    Dado un dataFrame con una columna numerica y una boolear calcula su correspondiente area AUC
    
    Args:
        df (dataFrame): Un dataFrame con una columna numerica y una boolear
    
    Returns:
        numero: El valor AUC
    """
    # Se garantiza que es un dataFrame
    if not isinstance(df, pd.DataFrame):
        sys.exit("df debe ser un DataFrame")
    # Se garantiza que la primera columna es numerica y la segunda logica
    if not (df.shape[1] == 2 and pd.api.types.is_numeric_dtype(df.iloc[:, 0]) and pd.api.types.is_bool_dtype(df.iloc[:, 1])):
        sys.exit("El df de entrada debe tener dos columnas, la primera numérica y la segunda lógica")
    # Se calcular los puntos de la curva ROC
    puntos = __curvaROC(df)
    x = [elem[0] for elem in puntos]
    y = [elem[1] for elem in puntos]
    # Se calcula el area de la curva ROC
    delta_x = np.diff(x)
    a=np.delete(y,0) #Para quitar los 0s del principio
    b=np.delete(y,len(y)-1) # y del final
    mean_y=np.mean(np.column_stack((a,b)), axis=1)
    return(np.sum(np.dot(delta_x,mean_y)))


def varianza(x):
    """
    Calcular la varianza
    
    Args:
        x (vector numerico): Un vector numerico
    
    Returns:
        numero: La varianza
    """
    aux = np.array(x)
    # Se garantiza que el vector es numerico
    if not np.issubdtype(aux.dtype, np.number):
        sys.exit("x debe ser numérico")   
    # Se computa la formula de la varianza
    return(np.sum((aux - np.mean(aux)) ** 2) / (len(aux) - 1))


def __calculoColumna(i, df):
    """
    Dado un dataFrame y un indice i, esta funcion calcula las metricas de la columna i
    
    Args:
        i (int): indice de columna
        df (dataFrame): dataFrame
    
    Returns:
        metricas: Si la columna es numerica la varianza y el AUC, si no, la entropia
    """
    # Se consigue la columna
    x = np.array(df.iloc[:,i])
    # Si es boolear
    if (np.issubdtype(x.dtype, np.bool_)):
        # Se garantiza que por delante tiene una columna numerica
        if (i == 0 or not np.issubdtype(np.array(df.iloc[:,i-1]).dtype, np.number)):
            sys.exit("Antes de la columna logica "+str(i)+" debe haber una columna numérica")
        # No se devuelve nada
        return(None)
    # Si es numerica
    elif (np.issubdtype(x.dtype, np.number)):
        # Se garantiza que despues tiene una columna boolear
        if (i == (df.shape[1] - 1) or not np.issubdtype(np.array(df.iloc[:,i+1]).dtype, np.bool_)):
            sys.exit("Despues de la columna numerica "+str(i)+" debe haber una columna lógica")
        # Se calculan la varianza y el AUC
        var = varianza(x)
        AUC = calculoAUC(df.iloc[:,i:i+2])
        return (var, AUC)
    # Si es categorica
    else:
        # Se calcula la entropia
        return(entropy(x))
    
def calculoMetricas(df):
    """
    Dado un dataFrame con variables continuas y discretas, calcula la entropia de 
    las discretas y AUC y varianza de las continuas 
    
    Args:
        df (dataFrame): dataFrame
    
    Returns:
        metricas: Un vector con las varianzas, AUC y entropias
    """
    # Se garantiza que es un dataFrame
    if not isinstance(df, pd.DataFrame):
        sys.exit("df debe ser un DataFrame")
    # Se calculan los resultados para cada columna
    res = [__calculoColumna(i, df) for i in range(df.shape[1])]
    # Se eliminan los valores None
    return ([elem for elem in res if elem != None])



    

