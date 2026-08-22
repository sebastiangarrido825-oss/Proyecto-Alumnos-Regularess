# Detección Temprana de Patrones de Riesgo Académico y Ausentismo Crítico en el Sistema Escolar Chileno

##  Descripción del Proyecto

Este proyecto aborda la problemática de la deserción, la reprobación y el ausentismo crítico en la educación regular chilena. En lugar de operar mediante un esquema reactivo, la propuesta transforma la gestión escolar hacia un modelo preventivo de alerta temprana. 

A través del análisis de microdatos históricos de rendimiento, asistencia acumulada y factores de entorno socioeducativo a nivel nacional (2023–2025), el sistema caracteriza de forma anticipada los perfiles de vulnerabilidad académica para orientar la asignación prioritaria de recursos pedagógicos, tutorías y acompañamiento psicosocial antes de que el deterioro sea irreversible.

---

## 📊 Tabla de Metadata (Fuentes de Datos Nacionales)

| Fuente / Portal | Versión / Período | Variables Clave | Descripción y Rol en el Proyecto |
| :--- | :--- | :--- | :--- |
| **Rendimiento Académico (Mineduc)** | 2025 | `ID_ESTUDIANTE`, `RBD`, `PROM_NOTAS`, `SIT_FIN`, `RAMO_REPROBADO` | Microdatos oficiales de desempeño académico individual, asignaturas reprobadas y condición final (Promovido, Reprobado, Retirado). (~3.120.500 registros). |
| **Matrícula Oficial (Centro de Estudios Mineduc)** | 2025 | `ID_ESTUDIANTE`, `RBD`, `GEN_ALU`, `EDAD_ALU`, `ASISTENCIA`, `IND_VULNERAB` | Registros sociodemográficos, dependencia del establecimiento, tipo de enseñanza y asistencia anual acumulada (~3.350.000 registros)
| **Encuesta Nacional de Deserción (ENDDEIE)** | 2023 | `COD_FACTOR_EXT`, `COD_REG_RBD`, `COD_COM_RBD`, `COD_DEPE` | Matriz contextual para homologar variables de entorno socioeducativo y factores externos de vulnerabilidad (~3.450 registros)

*Volumen Total Consolidado:* **6.473.950 registros** a nivel nacional

---

## 🛠️ Estructura del Repositorio

```text
proyecto/
├── README.md                  # Descripción general y tabla de metadata
├── data/
│   └── raw/                   # Archivos crudos de Mineduc y ENDDEIE (Frecuencia_Rendimiento_2025.xlsx, Frecuencias_Matrícula oficial 2025.xlsx, Libro_de_codigos_ENDDEIE_2023.xlsx)
└── notebooks/
    └── 01-obtencion.ipynb     # Script automatizado de carga, verificación de duplicados, limpieza y exploración
