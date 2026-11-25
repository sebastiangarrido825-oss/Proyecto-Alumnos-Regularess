# Control de las observaciones del profesor

| Observación | Evidencia de la corrección |
|---|---|
| Cuadernos 01–04 ausentes y CSV no disponibles | Cuatro notebooks, descarga oficial programada, manifiesto SHA-256 y resultados. La prueba local usa los CSV reales entregados. |
| Cifras incompatibles | Informe regenerado con 5.813.341 registros elegibles y métricas de 2025; no se conserva el recall anterior. |
| Merge duplica filas | Un registro anual por MRUN, exclusión de casos discordantes y validate=one_to_one. Cobertura por etapa en results/cobertura.csv. |
| Notas 0,0 y riesgo | Nota inválida -> NaN + bandera, y riesgo académico solo si hay nota válida. Comparación H3 con igual cantidad de alertas. |
| Falta hallazgo t−1 y H6 trivial | Tabla nota_vs_objetivo.csv, figura 2; asistencia previa y cobertura en reemplazo de la correlación construida. |
| Hipótesis no contrastadas | H1 con RR, IC95% y segmentación; H2 y H3 con pliegues pareados de entrenamiento. |
| Ficha sin respaldo / faltantes | Ficha con n, medida, faltantes, segmentos favorables/evaluados, decisión y motivo. Cobertura cuantifica las exclusiones por falta de historial. |
| Faltan gráficos en PDF | Figuras de EDA y modelado incorporadas. |
| ENDDEIE y demografía | Solo se recibió libro de códigos de ENDDEIE, no respuestas. GEN_ALU se conserva para auditoría, no se utiliza como predictor. |
| Partición temporal y métrica de costo | 2024 entrenamiento, 2025 prueba; AP para selección, costo supuesto FN/FP=10 y análisis de sensibilidad con OOF. |
| Redundancia y falta de árbol/bosque | Se excluyen riesgo derivado, nivel agrupado y bandera de asistencia constante; se comparan árbol, bosque, logística, HistGradientBoosting y dos baselines. |
| Búsqueda no anidada | GridSearchCV interno (2 pliegues) dentro de cada pliegue externo (3), exclusivamente 2024. |
| get_dummies antes del split | Imputación, escalado y OneHotEncoder dentro de Pipeline. |
| Test antes de búsqueda | Se lee la cohorte de prueba en modelar() después de selección y calibración. El EDA predictivo usa 2024. |
| Precisión, matriz y falsos negativos | Informe sección 5.5, results/modelo.json y figs/06_matriz.png. |
| Dependencias y archivos ajenos | requirements.txt, estructura limpia y gitignore. |
| Fecha y URL del repositorio | Fecha actualizada, URL proporcionada por el equipo incorporada en repositorio.txt y portada del PDF regenerada. Publicación remota pendiente. |

**Importante:** se analizaron los seis CSV completos y se evaluaron todos los registros elegibles de 2025. El ajuste de modelos usa 60.000 registros de 2024 por costo computacional; no se presenta como entrenamiento de la población completa. Los resultados se generaron con datos reales, no sintéticos. No se garantiza una nota ni se afirma que el profesor haya aceptado las decisiones metodológicas.
