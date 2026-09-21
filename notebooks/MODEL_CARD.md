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
| familia | LightGBM | LightGBM |
| MdAPE (test) | 30.68% | 41.22% |
| dentro de ±20% | 34.48% | 26.3% |
| cobertura del intervalo | 81.17% | 81.58% |
| baseline (tabla dinámica) | 38.53% | 57.72% |

## Cómo se evaluó
Partición 80/20 agrupada por `upm` (conglomerado muestral), definida en `split_upm.csv` y reutilizada sin
cambios desde la selección de variables. Todas las métricas están ponderadas por `factor`. El conjunto de
prueba se abrió una sola vez.

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
