# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 15:57:31 2023

@author: Josu
"""

import sys
import numpy as np
import random

        
class Dataset:
    '''
    Clase que representa Datasets
    '''
    def __init__(self, diccionario):
        """
        Este es el constructor del Dataset
        
        Args:
            diccionario (dict): Diccionario que guarda todos los valores
        
        Returns:
            Dataset: el Dataset creado
        """
        # Se garantiza que se ha dado un diccionario
        if not isinstance(diccionario, dict):
            sys.exit("Debe proporcionar un diccionario")
        # Se calcula cuantos valores hay en cada columna
        numerosVariables = [len(diccionario[elem]) for elem in diccionario]
        # Se garantiza que todas las columnas tienen la misma longitud
        if not np.all(np.array(numerosVariables) == numerosVariables[0]):
            sys.exit("Todas las variables deben tener el mismo número de elementos")
        # Se guarda el numero de lineas
        self.nrow = numerosVariables[0]
        # Se guarda el numero de columnas
        self.ncol = len(diccionario)
        # Se guarda el diccionario
        self.diccionario = {key: np.array(diccionario[key]) for key in diccionario}
    
    def print(self):
        """
        Esta función sirve para imprimir un Dataset
        """
        # Se imprimen los nombres de las columnas
        print("    ".join(self.diccionario.keys()))
        # Se consiguen todas las lineas
        lineas = ["  ".join([str(self.diccionario[key][i]) for key in self.diccionario]) for i in range(self.nrow)]
        # Se imprimen todas las lineas
        for linea in lineas:
            print(linea)

    def addRow(self, row):
        """
        Esta funcion sirve para añadir una linea a un Dataset
        
        Args:
            row (lista): La nueva linea
        """
        # Se garantiza que la linea tiene longitud correcta
        if len(row) != self.ncol:
            sys.exit("La fila debe tener " + str(self.ncol) + " valores")
        # Se añade la linea en el Dataset
        for i, key in enumerate(self.diccionario):
            self.diccionario[key] = np.append(self.diccionario[key], row[i])
        # Se actualiza el numero de lineas
        self.nrow += 1
        
    def addCol(self, nombre, valores):
        """
        Esta funcion sirve para añadir una columna a un Dataset
        
        Args:
            col (lista): La nueva columna
            nombre (str): El nombre de la nueva columna
        """
        nombre = str(nombre)
        # Se garantiza que la columna tiene una longitud correcta
        if len(valores) != self.nrow:
            sys.exit("La columna debe tener " + str(self.nrow) + " valores")
        # Se garantiza que el nombre de columna no esta utilizado
        if nombre in self.diccionario:
            sys.exit("El nombre ya está utilizado")
        # Se añade la columna
        self.diccionario[nombre] = np.array(valores)
        # Se actualiza el numero de columnas
        self.ncol += 1

    
    def getCols(self, indices):
        """
        Esta funcion devuelve las columnas indicadas de un Dataset
        
        Args:
            indices (lista): lista de los nombres de las columnas
        
        Returns:
            Lista: Una lista que contiene las columnas
        """
        # Se garantiza que todos los indices estan en el diccionario
        if not all([key in self.diccionario for key in indices]):
            sys.exit("Algún indice es erroneo")
        # Se consiguen y se devuelven las columnas
        return([self.diccionario[key] for key in indices] )
    
    def getRows(self, indices):
        """
        Esta funcion devuelve las lineas indicadas de un Dataset
        
        Args:
            indices (lista): lista de los indices de las lineas
        
        Returns:
            Lista: Una lista que contiene las lineas
        """
        indices = np.array(indices)
        # Se garantiza que los indices son números enteros
        if np.array(indices).dtype != int:
            sys.exit("Los indices deben ser enteros")
        # Se garantiza que los indices estan en el intervalo correcto
        if not all([i > 0 and i < self.nrow for i in indices]):
            sys.exit("Algún indice es erroneo")
        # Se consiguen y se devuelven las lineas
        lineas = [[self.diccionario[key][i] for key in self.diccionario] for i in range(self.nrow) if i in indices]
        return(lineas)