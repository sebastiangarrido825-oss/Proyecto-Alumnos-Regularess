# Predicción de riesgo de reprobación y deserción escolar
Taller de Ciencia de Datos (EIN146A25) · UTFSM · Paralelo 701

Predice, al inicio del año escolar *t*, qué estudiantes tienen mayor riesgo de **reprobar o retirarse** al cierre de *t*,
usando solo información del año anterior (*t−1*) y la matrícula de *t*. Datos abiertos del MINEDUC (2023–2025).

## Estructura
```
01-obtencion.ipynb          inventario de archivos, descarga opcional, auditoría de volumen y muestras
02-preprocesamiento.ipynb   limpieza, integración Matrícula+Rendimiento, rezago t-1, dataset final
03-eda.ipynb                hallazgos con figuras, contraste de H1, ficha de variables
04-modelado.ipynb           baselines, 4 modelos, CV anidada, umbral por costo, prueba única, H2 y H3
src/proyecto.py            toda la lógica (los cuadernos solo la llaman)
data/sample/               muestra del 5 % (versionada): permite correr TODO desde un clon vacío
data/raw/                  archivos completos de MINEDUC (no se suben; ver más abajo)
results/                   tablas y cifras generadas (de aquí salen los números del informe)
figs/  models/             figuras y modelo final
tests/                     generador de datos SINTÉTICOS solo para pruebas de humo
```

## Cómo reproducir
**A. Rápido, con la muestra (cualquier PC, minutos).** Funciona desde un clon vacío:
```bash
git clone <URL-DEL-REPOSITORIO> && cd <repo>
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace 02-preprocesamiento.ipynb 03-eda.ipynb 04-modelado.ipynb
```
(en Colab: `!git clone ...`, `%cd <repo>`, `!pip install -r requirements.txt`, y *Entorno de ejecución → Ejecutar todo* en cada cuaderno, en orden.)
Con la muestra los números son un 5 % de los estudiantes: las tendencias se mantienen, pero **las cifras del informe son las del archivo completo**.

**B. Completo.** Descargue de <https://datosabiertos.mineduc.cl/> (Estudiantes → Educación básica y media) *Matrícula por estudiante*
y *Rendimiento por estudiante* de 2023, 2024 y 2025, descomprímalos en `data/raw/` (o apunte otra carpeta con la variable
de entorno `DIR_RAW`) y ejecute los cuatro cuadernos **en orden** con *Restart & Run All*. El cuaderno 01 también puede
descargarlos por código si se le entregan los enlaces directos (`URLS`) y vuelve a generar `data/sample/`.

Cada cuaderno se ejecuta con *Restart & Run All* sin errores y deja sus salidas en `results/` y `figs/`.

## Decisiones clave
* **Unidad:** estudiante-año. Llave de cruce `MRUN` (+`AGNO` por archivo); se verifica `one_to_one` (el cruce no duplica filas).
* **Objetivo:** `SIT_FIN` ∈ {R, Y} → 1; P → 0; otros códigos (traslado, vacío) se excluyen y se cuentan. Configurable en `src/proyecto.py`.
* **Sin fuga:** ninguna nota ni asistencia del año *t* entra al modelo; el cuaderno 02 demuestra con datos por qué.
* **Ceros:** un promedio 0,0 es ausencia de dato → faltante + bandera, no riesgo.
* **Validación:** partición **temporal** (entrenar con 2024, probar en 2025); CV anidada; transformaciones dentro de `Pipeline`;
  la prueba se evalúa **una sola vez**, al final.
* **Umbral:** elegido por costo (`COSTO_FN_FP` en el cuaderno 04, a justificar), solo con datos de entrenamiento.

## Fuentes y licencia
Centro de Estudios, Ministerio de Educación de Chile — Datos abiertos (uso con citación de la fuente).
Se evaluó y descartó la ENDDEIE (Encuesta Nacional de Desarrollo Digital e Innovación Educativa): muestral y sin identificador de estudiante.
