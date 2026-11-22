# Preregistro del monitoreo (fase 6)

Proyecto CI-0148, CRISP-ML(Q) fase 6. Cada predicción se anota con fecha y hora **antes** de
correr el experimento correspondiente. Las entradas no se editan después; los resultados van a
`reports/fase6/` y al artículo.

## 0. Desviación del plan original

*Anotado el 2026-11-22 15:46 CST, antes de correr cualquier experimento de monitoreo.*

El documento de definición fija la regla «alerta si >10 % de predicciones caen fuera del rango
intercuartílico histórico». Aplicada al pie de la letra la alarma suena siempre: el IQR contiene
por construcción el 50 % central de la distribución, así que sin ningún cambio de dominio cerca
del 50 % de las cajas cae fuera, muy por encima del 10 %.

Corrección: se usan las cercas de Tukey, [Q1 − 1.5·IQR, Q3 + 1.5·IQR], sobre el ancho y el alto
de la caja predicha (píxeles de la imagen original). Se mantienen el umbral de 10 % (estricto,
`>`) y el monitoreo de tamaños de caja. Una caja está fuera si cualquiera de las dos features sale
de sus cercas (desigualdad estricta). Lotes de menos de 20 volúmenes no dan decisión.

Implementación: `src/deploy/monitor.py`, `scripts/monitor.py`.

## 1. Referencia: fit sobre los 764 volúmenes de desarrollo

*Anotado el 2026-11-22 15:46 CST, antes de correr `fit`.*

Predicción: `base_rate` entre 1 % y 5 % (unión de dos features con colas moderadas; para una
normal cada feature deja fuera ~0.7 %, colas más pesadas lo suben).

## 2. Test sin cambio de dominio (153 volúmenes)

*Anotado el 2026-11-22 15:46 CST, antes de correr `check` sobre test.*

Predicción: `alert = False`, con `frac_out` cerca de `base_rate` (dentro de ±3 puntos).

Salvedad: la referencia son predicciones dentro de la muestra (el modelo refit entrenó con esos
764 volúmenes), así que sus cajas pueden ser más «típicas» que las de test. Si `frac_out` en test
queda claramente por encima de `base_rate`, ese es el tamaño del sesgo.

## 3. Cambio de dominio simulado sobre los 153 volúmenes de test

*Anotado el 2026-11-22 15:50 CST, después de `fit` y de `check` sobre test, antes de correr
`scripts/simulate_shift.py`.*

Cercas conocidas: ancho [68.6, 114.8] px, alto [73.3, 114.2] px (mediana ~92 px en ambas).
Cota calculada sobre la referencia: si las cajas escalaran exactamente con el zoom, quedaría fuera
el 38.9 % con zoom 0.8 (ancho 24.2 %, alto 38.1 %) y el 65.1 % con zoom 1.25 (ancho 50.8 %,
alto 63.5 %). Un regresor tiende a encoger su respuesta hacia el tamaño típico de entrenamiento,
así que espero menos que esa cota.

| Transformación | Predicción `alert` | Predicción `frac_out` | Feature que más dispara |
|---|---|---|---|
| zoom 0.8 en H y W (`order=1`) | True | 10–39 % | alto |
| zoom 1.25 + recorte central a 320 | True | 20–65 % | alto |
| ganancia ×0.7, recorte a 4095 | False | ≈ 0 % (≤ 2 %) | — |

Sobre la ganancia: la normalización por percentiles (0.5–99.5) de `preprocess_volume` es
invariante a una escala global, salvo por el redondeo a uint16. Espero cajas casi idénticas a las
originales. Esto no prueba que el monitor «detecte» bien la intensidad: el cambio queda absorbido
antes de llegar al modelo. El punto ciego real (un cambio de intensidad no lineal que sí altere
las predicciones sin mover el tamaño de caja) no se cubre con este experimento.

Si un encogimiento del 20 % no basta para pasar el 10 %, se reporta tal cual como resultado sobre
la sensibilidad del monitor.
