# Taller de Ciencia de Datos – Informes 3 y 4 corregidos
UTFSM · EIN146A25 · Sebastián Garrido y Benjamín Enríquez · 25/11/2026

Repositorio del equipo: [https://github.com/sebastiangarrido825-oss/Proyecto-Alumnos-Regularess](https://github.com/sebastiangarrido825-oss/Proyecto-Alumnos-Regularess)

## Reproducción desde un clon
1. Instalar Python 3.13 (versión utilizada 3.13.5) y ejecutar `python -m pip install -r requirements.txt`. Puede usarse un entorno virtual. Se requiere espacio libre para los CSV (3,6 GB) y las cohortes (aprox. 200 MB), más memoria para las uniones. Se recomienda 8 GB de RAM.
2. Descomprimir el RAR original `archivos_csv.rar` dentro de `data/raw/` de modo que los seis CSV estén directamente en esa carpeta. Alternativamente, el notebook 01 descarga las versiones oficiales mediante las URLs de `src/descarga.py` (requiere 7-Zip o unrar). Si los archivos descargados no coinciden con los SHA-256 registrados en `results/fuentes_referencia.csv`, se detiene la ejecución para no mezclar versiones. La descarga remota no fue probada en esta ejecución; sí se comprobaron los CSV reales proporcionados.
3. Abrir Jupyter desde la raíz del proyecto y ejecutar los notebooks `01-obtencion.ipynb`, `02-preprocesamiento.ipynb`, `03-eda.ipynb`, `04-modelado.ipynb` en orden con Restart & Run All. Las funciones se encuentran en `src/ejecucion.py`, `src/analisis.py` y `src/modelado.py`; se ejecutaron con los archivos reales y dejaron los resultados incluidos. Alternativa equivalente: `python ejecutar_todo.py`.
4. Ejecutar `python generar_informe.py` si cambian los resultados o el enlace registrado en `repositorio.txt`. La URL proporcionada por el equipo ya está incorporada en el PDF.

## Metodología y evidencia
- Auditoría y procesamiento de los seis CSV completos, con cardinalidad one_to_one, separación de traslados y exclusión de registros contradictorios. Objetivo R/Y frente a P; sin historial o sin matrícula se contabilizan y excluyen. Nota cero = faltante; asistencia cero se conserva.
- 2023 aporta el historial; cohorte 2024 se usa para exploración y entrenamiento; 2025 para evaluación final. Muestra aleatoria simple de 60.000 registros de 2024 para ajuste/CV (semilla 42). Prueba sobre todos los registros elegibles de 2025. CV interna estratificada de 2 pliegues y externa de 3 dentro de 2024. No hay suficientes cohortes anteriores para validación temporal interna.
- Modelo seleccionado por AP interna; umbral definido por costo ilustrativo FN/FP=10 sobre OOF. Las hipótesis se contrastan sin usar 2025 para elegir variables, hiperparámetros ni umbral.
- `results/` contiene resultados calculados y el manifiesto inmutable de datos, `figs/` los gráficos, `models/` el modelo entrenado. No contiene microdatos, identificadores individuales ni predicciones personales; los archivos crudos y cohortes locales se excluyen del repositorio. No se publican datos de alumnos.
- `Informe_3_y_4_corregido.pdf` se genera automáticamente leyendo los resultados. La sección de resultados de test se calculó después de fijar el procedimiento. El código se volvió a ejecutar para comprobar el ajuste de una variable constante y el cálculo exacto de costo; el resultado final de prueba se mantuvo. No se ajustó nada usando los resultados de test.
- Los cuatro notebooks se ejecutaron desde Jupyter en el orden 01–04 con datos reales; se conservan sus salidas y el registro `results/ejecucion_notebooks.log`. El equipo debe revisar y reproducir el procedimiento antes de entregar al profesor.

## Pendiente externo
Publicar los archivos corregidos en el GitHub del equipo y comprobar el acceso del profesor. La URL ya se encuentra incorporada; no se ha realizado ninguna publicación remota. La fuente ENDDEIE no se integra porque el RAR solo contiene el libro de códigos, no las respuestas. El equipo debe revisar la declaración de IA y la metodología.
