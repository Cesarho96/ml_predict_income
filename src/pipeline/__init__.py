"""Del microdato crudo de la ENIGH al dataset que entrena el modelo (Hito 5b).

    enigh.py      leer las tablas de una edición (texto, NA de INEGI normalizados)
    ingesta.py    tablas crudas → una fila por persona adulta con ingreso   (notebook 01)
    features.py   persona → las variables que el modelo usa                 (notebook 02)
    particion.py  UPM → train/validation/test, estable entre ediciones
    construir.py  lo anterior de punta a punta; es lo que corre el DAG

Qué NO es: una copia de los notebooks. Los notebooks exploraron ~90 columnas para decidir
cuáles servían; este paquete calcula sólo las que el modelo congelado usa (13 features +
objetivo + diseño muestral). La exploración vive en notebooks; producción calcula lo que
el producto consume. Menos código = menos superficie que se puede romper cuando INEGI
publique una edición nueva.

La garantía de que el recorte no cambió nada es la prueba de paridad
(`tests/test_paridad.py`): sobre la ENIGH 2024, la salida de este paquete es idéntica,
columna por columna y fila por fila, al parquet que produjeron los notebooks.
"""
