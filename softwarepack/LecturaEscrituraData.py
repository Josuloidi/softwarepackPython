# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 11:04:22 2023

@author: Josu
"""

import pandas as pd
import numpy as np
import sys

            
def __cambiarClaseColumna(x):
    """
    Esta función toma una columna tipo string de un dataFrame y le asigna su correspondiente clase
    
    Args:
        x (lista): La columna
    
    Returns:
        lista: La columna con la clase cambiada
    """
    # Si todos los valores son True o False, la columna sera boolear 
    if all([elem in ["True", "False"] for elem in x]):
        x[x == "True"] = True
        x[x == "False"] = False
        return(x.astype(bool))
    # Si no
    else:
        try:
            # Se intenta convertir cada elemento en entero
            [int(elem) for elem in x]
            # Si no da error, se convierte a int toda la columna
            return(x.astype(int))
        except:
            try:
                # Se intenta convertir cada elemento en float
                [float(elem) for elem in x]
                # Si no da error, se convierte a float toda la columna
                return(x.astype(float))
            except:
                # Si no, se devuelve como habia llegado
                return(x)


def readDataFrame(filePath, header = False, sep = ";"):
    """
    Dado un directorio de un fichero lee la informacion y crea un dataFrame
    
    Args:
        filePath (str): Directorio del fichero
        header (bool): Indicativo de si el fichero guarda los nombres de las columnas
        sep (str): Separador que se ha utilizado en el fichero csv
    
    Returns:
        dataFrame: El dataFrame leido
    """
    # Se habre el fichero
    with open(filePath,'r') as f:
        # Se lee la primera linea
        line = f.readline()[:-1]
        # Consigue el numero de columnas
        n = len(line.split(sep))
        # Si se ha indicado que se ha guardado el header
        if header:
            # Se guardan los nombres de las columnas y se lee otra linea
            nombresColumnas = line.split(sep)
            line = f.readline()[:-1]
        else:
            # Si no, se crean nombres de las columnas: x1, x2, ...
            nombresColumnas = ["x" + str(i) for i in np.arange(n)+1]
        # Se crea un diccionario con los nombres de las columnas
        data = {nombre: [] for nombre in nombresColumnas}
        # Por cada linea del fichero
        while line:
            # Se separa la linea utilizando el separador
            valores = line.split(sep)
            # Se garantiza que tiene una longitud correcta
            if (len(valores) != n):
                sys.exit("Todas las líneas deben tener el mismo numero de datos")
            # Se añade la linea en el diccionario
            for key, valor in zip(data.keys(), valores):
                data[key].append(valor)
            line = f.readline()[:-1]
        # Se crea el dataFrame
        df = pd.DataFrame(data)
        # Se cambian las clases de las columnas
        df = df.apply(lambda x: __cambiarClaseColumna(x))
    return(df)


def writeDataFrame(df, filePath, header = False, sep = ";"):
    """
    Dado un directorio de un fichero y un dataFrame, escribe el dataFrame en el fichero
    
    Args:
        filePath (str): Directorio del fichero
        df (dataFrame): El dataFrame
        header (bool): Indicativo de si se quieren guardar los nombres de las columnas
        sep (str): Separador que se utilizara en el fichero csv
    """
    # Se garantiza que se ha dado un dataFrame
    if not isinstance(df, pd.DataFrame):
        sys.exit("df debe ser un DataFrame")
    # Se consiguen todas las lineas
    lineas = list(df.apply(lambda x: sep.join([str(elem) for elem in x])+"\n", axis=1))
    # Se crea el fichero
    f = open(filePath, 'w')
    # Si se quieren guardar los nombres de las columnas
    if header:
        # Se escriben los nombres de las columnas
        f.write(sep.join(df.columns) + "\n")
    # Se escriben todas las lineas
    f.writelines(lineas)
