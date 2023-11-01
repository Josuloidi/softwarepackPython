from softwarepack.Discretizacion import *
from softwarepack.Metricas import *
from softwarepack.NormalizacionYEstandarizacion import *
from softwarepack.Filtrado import *
from softwarepack.Correlacion import *
from softwarepack.Visualizacion import *
from softwarepack.LecturaEscrituraData import *
from softwarepack.Objetos import *


import pandas as pd
import random




def test_all():
    x = [11.5, 10.2, 1.2, 0.5, 5.3, 20.5, 8.4, 4.4]
    numBins = 4
    
    xDiscretized, cutPoints = discretizeEW(x, numBins)
    xDiscretized, cutPoints = discretizeEF(x, numBins)
    xDiscretized, cutPoints = discretizeJNB(x, numBins)
    
    n = 10
    data = {
        'a': [round(random.uniform(0, 100),2) for i in range(n)],
        'c': [round(random.uniform(100, 200),2) for i in range(n)],
        'd': [round(random.uniform(0, 1000),2) for i in range(n)],
    }
    df = pd.DataFrame(data)
    
    res = discretizarColumnas(df, numBins, "EW")
    res = discretizarColumnas(df, numBins, "EF")
    res = discretizarColumnas(df, numBins, "JNB")
    res = discretizarColumnas(df, numBins)
    
    x = [round(random.uniform(0, 100),2) for i in range(n)]
    varianza(x)
    
    n = 20
    data = {
        'x': random.sample(range(1, 101), n),
        'y': [random.choice([True, False]) for i in range(n)]
    }
    df = pd.DataFrame(data)
    calculoAUC(df)
    
    x = ['a', 'a', 'c', 'c', 'c']
    entropy(x)
    
    n = 20
    data = {
        'a': random.sample(range(1, 101), n),
        'b': [random.choice([True, False]) for i in range(n)],
        'c': [random.choice(["a", "b", "c", "d"]) for i in range(n)],
        'd': random.sample(range(1, 51), n),
        'e': [random.choice([True, False]) for i in range(n)],
        'f': [random.choice(["a", "b", "c"]) for i in range(n)],
        'g': [random.choice(["a", "b"]) for i in range(n)]
    }
    df = pd.DataFrame(data)
    calculoMetricas(df)
    
    n = 20
    x = random.sample(range(1, 101), n)
    x_norm = normalizar(x)
    
    n = 20
    x = random.sample(range(1, 101), n)
    x_est = estandarizar(x)
    
    n = 20
    data = {
        'a': random.sample(range(1, 101), n),
        'b': random.sample(range(1, 51), n),
        'c': random.sample(range(1, 101), n),
        'd': random.sample(range(1, 51), n)
    }
    df = pd.DataFrame(data)
        
    df_norm = normalizarDF(df)
    df_est = estandarizarDF(df)
    
    data = {
        'a.x': random.sample(range(1, 101), n),
        'a.y': [random.choice([True, False]) for i in range(n)],
        'b': [random.choice(["a", "b", "c", "d"]) for i in range(n)],
        'c.x': random.sample(range(1, 51), n),
        'c.y': [random.choice([True, False]) for i in range(n)],
        'd': [random.choice(["a", "b", "c"]) for i in range(n)],
        'e': [random.choice(["a", "b"]) for i in range(n)],
        'f': random.sample(range(50, 101), n),
    }
    df = pd.DataFrame(data)
    
    filtrar(df, ent = True, entUmbr = 1.6, entOP = "LO")
    filtrar(df, AUC = True, AUCUmbr = 0.5, AUCOP = "HI")
    filtrar(df, varianza = True, varUmbr = 1000, varOP = "HI")
    
    filtrar(df, ent = True, entUmbr = 1.6, entOP = "LO",
        AUC = True, AUCUmbr = 0.5, AUCOP = "HI",
        varianza = True, varUmbr = 1000, varOP = "HI")
    
    n = 20
    x = random.sample(range(1, 101), n)
    y = random.sample(range(1, 101), n)
    correlacion(x, y)
    
    n = 10
    x = [random.choice(["a", "b", "c"]) for i in range(n)]
    y = [random.choice(["a", "b", "c"]) for i in range(n)]
    correlacion(x,y)
    
    data = {
        'a': random.sample(range(1, 101), 20),
        'b': random.sample(range(1, 51), 20),
        'c': random.sample(range(50, 101), 20),
        'd': random.sample(range(100, 201), 20),
        'h': random.sample(range(75, 125), 20)
    }
    df = pd.DataFrame(data)
    correlacionDF(df)
    
    data = {
        'a': [random.choice(["a", "b", "c", "d"]) for i in range(20)],
        'b': [random.choice(["a", "b"]) for i in range(20)],
        'c': [random.choice(["b", "c", "d"]) for i in range(20)],
        'd': [random.choice(["b", "c"]) for i in range(20)],
        'h': [random.choice(["a", "b", "c", "d"]) for i in range(20)]
    }
    df = pd.DataFrame(data)
    correlacionDF(df)
    
    data = {
        'a': [random.choice(["a", "b", "c", "d"]) for i in range(20)],
        'b': random.sample(range(1, 51), 20),
        'c': [random.choice(["b", "c", "d"]) for i in range(20)],
        'd': [random.choice(["b", "c"]) for i in range(20)],
        'h': random.sample(range(75, 125), 20)
    }
    df = pd.DataFrame(data)
    correlacionDF(df)
    
    n = 100
    data = {
        'x': random.sample(range(1, 101), n),
        'y': [random.choice([True, False]) for i in range(n)]
    }
    df = pd.DataFrame(data)
    plotROC(df)
        
    data = {
        'a': random.sample(range(1, 101), 20),
        'b': random.sample(range(1, 51), 20),
        'c': random.sample(range(50, 101), 20),
        'd': random.sample(range(100, 201), 20),
        'h': random.sample(range(75, 125), 20)
    }
    df = pd.DataFrame(data)
    
    plotCorrelaciones(df)
    
    data = {
        'a': [random.choice(["a", "b", "c", "d"]) for i in range(20)],
        'b': [random.choice(["a", "b"]) for i in range(20)],
        'c': [random.choice(["b", "c", "d"]) for i in range(20)],
        'd': [random.choice(["b", "c"]) for i in range(20)],
        'h': [random.choice(["a", "b", "c", "d"]) for i in range(20)]
    }
    df = pd.DataFrame(data)
    
    plotCorrelaciones(df)
    
    data = {
        'a': [random.choice(["a", "b", "c", "d"]) for i in range(20)],
        'b': random.sample(range(1, 51), 20),
        'c': [random.choice(["b", "c", "d"]) for i in range(20)],
        'd': [random.choice(["b", "c"]) for i in range(20)],
        'h': random.sample(range(75, 125), 20)
    }
    df = pd.DataFrame(data)
    plotCorrelaciones(df)
    
    data = {
        'a': random.sample(range(1, 101), 20),
        'b': random.sample(range(1, 501), 20),
        'c': random.sample(range(50, 151), 20),
        'd': random.sample(range(100, 501), 20),
        'h': random.sample(range(50, 301), 20)
    }
    df = pd.DataFrame(data)
    plotBoxPlots(df)
    
    x = random.sample(range(1, 2000), 500)
    numBins = 4
    plotBarrasDiscretizadas(x, numBins, "EW")
    plotBarrasDiscretizadas(x, numBins, "EF")
    plotBarrasDiscretizadas(x, numBins, "JNB")
    
    n = 10
    data = {
        'a': [round(random.uniform(0, 100),2) for i in range(n)],
        'b': [random.choice([True, False]) for i in range(n)],
        'c': [random.choice(["a", "b", "c", "d"]) for i in range(n)],
        'd': random.sample(range(1, 51), n),
        'e': [random.choice([True, False]) for i in range(n)],
        'f': [random.choice(["X", "Y", "Z"]) for i in range(n)],
        'g': [round(random.uniform(0, 100),2) for i in range(n)],
        'h': random.sample(range(50, 101), n),
    }
    df = Dataset(data)
    
    df.print()
    df.addRow([1,True,"A",34,"True","T",2.3,100])
    df.addCol("i", [random.choice([True, False]) for i in range(11)])
    df.getCols(["a", "i", "d"])
    df.getRows([2,6,10])
    
    
    n = 20
    data = {
        'a': [round(random.uniform(0, 100),2) for i in range(n)],
        'b': [random.choice([True, False]) for i in range(n)],
        'c': [random.choice(["a", "b", "c", "d"]) for i in range(n)],
        'd': random.sample(range(1, 51), n),
        'e': [random.choice([True, False]) for i in range(n)],
        'f': [random.choice(["X", "Y", "Z"]) for i in range(n)],
        'g': [round(random.uniform(0, 100),2) for i in range(n)],
        'h': random.sample(range(50, 101), n),
    }
    df = pd.DataFrame(data)
    df
    
    writeDataFrame(df, "tabla.csv", header = True, sep = "#")
    dfAux = readDataFrame("tabla.csv", header = True, sep = "#")
    