En 01_EsaTlmy.ipynb cargo un dataframe y me quedo con los campos "channel_44" y "anomaly".
1.Primero quiero normalizar la frecuencia a la frecuencia dominante que es 30 segundos
1.1 Interpolar para el campo channel_44
1.2 Hacer un llenado ffill o bfill para anomaly que debe ser entero 0 o 1.

2.Con la frecuencia normalizada quiero aplicar una ventana deslizante de n minutos y calcular Media, Desviación estándar, máximo,  Mínimo y tendencia o pendiente, para eso hay una funcion desarrollada.

3.Ahora me gustaria un pair plot de los features pero coloreando si es anomalia o no

4.Ahora tengo los campos channel_44_max, channel_44_std, anomaly y cluster. Tengo plotear e identificar la siguientes situaciones. 1.Puntos con cluster, puntos sin cluster que son anomalias y puntos con cluster que son anomalias
  