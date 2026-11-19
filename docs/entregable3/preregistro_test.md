# Preregistro de la evaluación en el conjunto de prueba

Proyecto CI-0148, CRISP-ML(Q) fase 4. Redactado el 17 de noviembre de 2026, **antes de
cualquier predicción sobre el conjunto de prueba**. La fecha del commit que agrega este archivo
es la evidencia de que lo que sigue se fijó antes de ver el test. El archivo no se edita
después de la evaluación: los resultados van a `reports/fase4/` y al artículo.

## 1. Qué se evalúa

El modelo M1 (ResNet-18 2.5D con cabeza de clasificación y cabeza de caja), congelado al
cierre de la fase 3. Ninguno de sus valores se eligió con el test.

| Elemento | Valor | Origen |
|---|---|---|
| Configuración | `configs/train/m1_refit.yaml` (sha256 `299e255a…19caf5`) | `scripts/cv.py` a partir de `m1.yaml` |
| Hiperparámetros | los de `m1.yaml` | ablaciones en fold_0; Optuna descartado por no replicar (0.7179 < 0.7204) |
| Entrenamiento | 5 folds de desarrollo (764 volúmenes, 23 669 cortes), 8 épocas, coseno con `T_max` = 20 | `refit_epochs` = ⌈mediana(best_epoch + 1)⌉ de 10 runs del CV |
| Semillas | 42 y 0 | las mismas del CV |
| Checkpoints | `runs/refit/s42/last.pt` y `runs/refit/s0/last.pt` (hashes abajo) | `scripts/refit.py` |
| Agregación | z: Kadane sobre p − τ; xy: mediana ponderada por p (`wmedian`) | ablación de agregación en fold_0 y OOF con 2 semillas |
| Umbral | τ = 0.65, fijo | mediana de `select_tau` sobre el OOF de cada semilla (0.6 y 0.7) |
| Código | commit `9b56e68` (rama `fase3-modelado`) | |

Hashes completos (sha256):

```
c0d41e8058d9c0192c39e840ed3a1984554eb38200c2e3a3187655b8a6be8f11  runs/refit/s42/last.pt
34f57dc31616dbc467a51e03a4f6d5ad304f20a118c2019c8b68b83a1113e7b5  runs/refit/s0/last.pt
299e255ac0cb9edf9bf37cc4ddf9ca83c018013aa5b0cd676ee1f8353119caf5  configs/train/m1_refit.yaml
```

Nota de trazabilidad: el refit se ejecutó sobre el commit `096dafb` con `src/train/loop.py`
modificado y `scripts/refit.py` sin trackear (registrado en `runs/refit/s*/env.txt`); ese
código se commiteó sin cambios en `9b56e68`.

La línea base M0 es la caja fija promedio (fase 1) calculada con **todos** los volúmenes de
desarrollo y aplicada a cada volumen de prueba. El test no contribuye a la caja.

## 2. Conjunto de prueba

153 volúmenes (115 de clase 0, 29 de clase 1, 9 de clase 2), separados en la fase 2
(`datasets/interim/splits.csv`, sha1 `0877f2ee…`), agrupados por `examId` y sin intersección
con desarrollo. Hasta este commit ningún script ha predicho sobre ellos.

## 3. Métricas y análisis

**Métrica principal.** IoU 3D media sobre los 153 volúmenes. La IoU de cada volumen es la
media de las IoU obtenidas con las semillas 42 y 0 (la misma unidad que el resumen del CV).
No se promedian las predicciones de las dos semillas (eso sería otro modelo, no validado).
Intervalo de confianza al 95 %: bootstrap percentil sobre volúmenes, 2000 remuestreos,
semilla 42 (`bootstrap_mean`).

**Secundarias.**
- Diferencia pareada M1 − M0 por volumen, con IC 95 % por bootstrap pareado (`paired_bootstrap`).
- Fracción de volúmenes con IoU ≥ 0.5.
- IoU por clase de diagnóstico, con IC.
- Variabilidad entre semillas (`seed_std`) y, de forma descriptiva, AUPRC y F1 a nivel de corte.

## 4. Predicciones

| Cantidad | Predicción | Justificación |
|---|---|---|
| IoU media M1 (todas) | **0.71**, rango esperado **[0.69, 0.73]** | IoU OOF con un único τ: 0.709 (seed 42, τ 0.6) y 0.713 (seed 0, τ 0.7). Desviación estándar por volumen en el OOF ≈ 0.137, n = 153 → error estándar ≈ 0.011 → ±0.022 al 95 % |
| IoU media M0 | ≈ 0.39 | 0.391 en la fase 1; 0.393 OOF en la fase 3 |
| Δ M1 − M0 | ≈ +0.32, IC que excluye 0 | +0.323 [0.308, 0.339] en el OOF |
| Fracción IoU ≥ 0.5 | ≈ 0.93 | 0.937 en el OOF |
| Orden por clase | clase 0 > clase 1 > clase 2 | OOF: 0.726, 0.703, 0.643 |

El rango [0.69, 0.73] cubre solo el error de muestreo del test. No incluye el efecto de
entrenar con 5 folds en lugar de 4 (que debería ayudar levemente) ni el optimismo residual
de haber elegido τ y la época con los mismos datos de validación (≈ 0.006 según el CV).

## 5. Criterios de decisión

1. **Criterio de éxito de la fase 1:** IoU media en test ≥ 0.5. Se considera cumplido si la
   media lo supera; se reporta además si el límite inferior del IC también lo supera.
2. **Superioridad sobre la línea base:** el IC 95 % de Δ M1 − M0 excluye 0 con Δ > 0.
3. **Coherencia con el CV:** la media cae dentro de [0.69, 0.73]. Si cae fuera, no se
   reajusta nada: se reporta el valor y se discuten causas posibles (cambio de distribución
   entre folds y test, varianza del refit, optimismo del CV).
4. **Clase 2 (rotura completa):** con n = 9 su resultado es descriptivo. No se hace
   inferencia ni se declara éxito o fracaso por clase.

## 6. Compromisos

- `scripts/evaluate_test.py` se ejecuta **una vez**. Se niega a correr si este archivo no
  está commiteado o si ya existe una salida en `reports/fase4/`.
- No se cambia el modelo, τ, la agregación ni el preprocesamiento después de ver el test.
  Cualquier modelo posterior requeriría un conjunto de prueba nuevo.
- Todo análisis adicional sobre el test (Grad-CAM de errores, cortes por tamaño de caja,
  etc.) se rotula como **exploratorio**. El análisis cualitativo principal de errores se hace
  sobre las predicciones OOF del CV.
- Si una ejecución falla por un error de código (no por el resultado), se corrige el error,
  se documenta en el artículo y se vuelve a ejecutar con la misma configuración y los mismos
  checkpoints.
