# Model card — predicción de ingreso mensual (ENIGH 2024)

**Versión** v1 · **Fuente** INEGI, ENIGH 2024 Nueva Serie · **Pesos de** agosto 2024

## Qué predice
La **mediana condicional** del ingreso corriente monetario mensual de una persona adulta en México,
junto con un intervalo del 80%. No predice el ingreso exacto de nadie.

## Dos modelos, ruteados por una pregunta
*¿Trabajaste el mes pasado?*

| | Ocupados | No ocupados |
|---|---|---|
| preguntas | 10 | 7 |
| familia | XGBoost | HistGradientBoosting |
| MdAPE (test) | 30.78% | 41.03% |
| dentro de ±20% | 34.35% | 26.18% |
| RMSE (test, MXN) | $13,105 | $9,734 |
| RMSE en log | 0.7046 | 0.8609 |
| cobertura del intervalo | 80.64% | 81.34% |
| baseline (tabla dinámica) | 38.57% | 59.61% |

## Cómo se evaluó
Partición de tres vías agrupada por `upm` (conglomerado muestral): 64% train, 16% validation, 20% test.
El test es exactamente el mismo que definió `split_upm.csv` en la selección de variables; validation se
recortó de train para no moverlo. Hiperparámetros y calibración del intervalo se eligieron contra
validation; el test se abrió una sola vez. Todas las métricas están ponderadas por `factor`.

El **RMSE en pesos** es alto por construcción: eleva al cuadrado errores de una distribución con cola
larga, así que lo dominan unas decenas de casos del extremo alto. Se reporta como diagnóstico de cola,
no como métrica de producto; la métrica de producto es el MdAPE junto con la cobertura del intervalo.

## Limitaciones
- **El error no se reparte parejo.** El modelo acierta en el centro de la distribución y falla en los
  extremos: sobreestima el decil más bajo y subestima el más alto. Es el comportamiento esperado de un
  estimador de mediana y la razón de publicar un intervalo.
- **Universo**: adultos de 18+ con ingreso monetario positivo. No cubre a quien no percibe ingreso.
- **Corte temporal**: refleja el mercado laboral de 2024. El ingreso nominal se mueve con la inflación y
  con el salario mínimo; el modelo no se actualiza solo.
- **No es una tasación individual.** Un intervalo con razón superior/inferior de ~3x no sirve para decidir
  un crédito ni un sueldo concreto; sirve para ubicar a alguien en la distribución.
- **Sesgos de la fuente**: el ingreso se autorreporta y en encuestas de hogares se subdeclara,
  especialmente en la parte alta.

## Qué NO usa
Ninguna variable de ingreso o gasto del hogar, ni activos del hogar. El detalle está en
`seleccion_v3.json`.
