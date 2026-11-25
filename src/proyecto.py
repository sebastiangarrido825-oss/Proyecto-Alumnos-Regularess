"""Funciones compartidas por los cuadernos 01-04.

Taller de Ciencia de Datos (EIN146A25) - Proyecto: riesgo de reprobacion/desercion escolar.
Todo el codigo con logica vive aqui para que los cuadernos sean cortos y cada cifra del
informe pueda rastrearse a una celda.
"""
from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- rutas
RAIZ = Path(__file__).resolve().parents[1]
DIR_RAW = Path(os.environ.get("DIR_RAW", RAIZ / "data" / "raw"))
DIR_MUESTRA = RAIZ / "data" / "sample"
DIR_PROC = RAIZ / "data" / "processed"
DIR_FIGS = RAIZ / "figs"
DIR_RES = RAIZ / "results"
DIR_MODELOS = RAIZ / "models"
for _d in (DIR_RAW, DIR_MUESTRA, DIR_PROC, DIR_FIGS, DIR_RES, DIR_MODELOS):
    _d.mkdir(parents=True, exist_ok=True)

SEMILLA = 42
MODO_RAPIDO = os.environ.get("MODO_RAPIDO") == "1"      # solo para pruebas de humo
FORMATO_TABLA = os.environ.get("FORMATO_TABLA", "parquet")
ANIOS = (2023, 2024, 2025)
ANIOS_OBJETIVO = (2024, 2025)                           # t; el rezago t-1 sale de 2023 y 2024

# ---------------------------------------------------- definicion del objetivo (CONFIGURABLE)
# IMPORTANTE: en SIT_FIN la categoria 'Y' mezcla retirados Y trasladados. SIT_FIN_R los separa
# (verificado con los datos: SIT_FIN='Y' = SIT_FIN_R 'Y' (retirado) + SIT_FIN_R 'T' (trasladado)).
COLUMNA_SIT = "SIT_FIN_R"
CODIGOS_POSITIVOS = {"R", "Y"}      # R = reprobado, Y = retirado
CODIGOS_NEGATIVOS = {"P"}           # P = promovido
# Cualquier otro codigo (T = trasladado, vacio, etc.) NO es fracaso ni exito: se excluye y se cuenta.

# Totales oficiales del Centro de Estudios MINEDUC (tabulaciones de frecuencia de Rendimiento),
# establecimientos en funcionamiento. Sirven para validar el archivo descargado.
OFICIAL_SIT = {
    2023: {"promovido": 3_182_094, "reprobado": 81_620, "trasladado": 227_282, "retirado": 91_356, "sin_informacion": 1_835},
    2024: {"promovido": 3_165_353, "reprobado": 77_238, "trasladado": 239_599, "retirado": 83_527, "sin_informacion": 3_176},
    2025: {"promovido": 3_146_183, "reprobado": 75_585, "trasladado": 235_867, "retirado": 76_484, "sin_informacion": 6_216},
}
MAPA_SIT_OFICIAL = {"P": "promovido", "R": "reprobado", "T": "trasladado", "Y": "retirado"}

# ----------------------------------------------------------------- variables del modelo
OBJETIVO = "estudiante_reprobado"
TIPOS = {
    "prom_notas_norm_prev": "num",
    "asistencia_prev": "num",
    "prom_prev_faltante": "bin",
    "asistencia_prev_faltante": "bin",
    "riesgo_academico_prev": "bin",
    "repro_prev": "bin",
    "retiro_prev": "bin",
    "GEN_ALU": "cat",
    "COD_ENSE": "cat",
    "COD_ENSE2": "cat",
}


# ------------------------------------------------------------------------------ lectura
def ubicar_archivo(tipo: str, anio: int):
    """tipo = 'rend' o 'mat'. Prefiere data/raw (archivo completo); si no, data/sample."""
    pat_raw = f"*Rendimiento_{anio}_*" if tipo == "rend" else f"*Matr*cula*_{anio}_*"
    for carpeta, pat in ((DIR_RAW, pat_raw), (DIR_MUESTRA, f"muestra_{tipo}_{anio}*")):
        hits = sorted(p for p in carpeta.glob(pat) if p.suffix.lower() in (".csv", ".gz"))
        if hits:
            return hits[0]
    return None


def _sep(ruta):
    abrir = gzip.open if str(ruta).endswith(".gz") else open
    with abrir(ruta, "rt", encoding="latin1") as f:
        cab = f.readline()
    return ";" if cab.count(";") > cab.count(",") else ","


def _norm(c: str) -> str:
    return c.replace("\u00ef\u00bb\u00bf", "").replace("\ufeff", "").strip().upper()


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False).str.strip(), errors="coerce")


def _tipar(ch: pd.DataFrame) -> pd.DataFrame:
    if "MRUN" in ch:
        ch["MRUN"] = pd.to_numeric(ch["MRUN"], errors="coerce")
    if "AGNO" in ch:
        ch["AGNO"] = pd.to_numeric(ch["AGNO"], errors="coerce").fillna(0).astype("int16")
    for c in ("PROM_GRAL", "ASISTENCIA"):
        if c in ch:
            ch[c] = _num(ch[c]).astype("float32")
    for c in ("GEN_ALU", "COD_ENSE", "COD_ENSE2"):
        if c in ch:
            ch[c] = pd.to_numeric(ch[c], errors="coerce").fillna(-1).astype("int16")
    for c in ("SIT_FIN", "SIT_FIN_R"):
        if c in ch:
            ch[c] = ch[c].astype(str).str.strip().str.upper().replace({"NAN": ""})
    return ch


def leer_csv(ruta, columnas, chunksize: int = 500_000, mrun_modulo: int | None = None,
             estadisticas: dict | None = None) -> pd.DataFrame:
    """Lee solo las columnas pedidas (nombres normalizados), por trozos y tipadas.
    Si se pasa `estadisticas`, acumula ahi las filas totales y el conteo de SIT_FIN del archivo COMPLETO
    (antes de aplicar el filtro de muestra)."""
    ruta = Path(ruta)
    sep = _sep(ruta)
    cab = pd.read_csv(ruta, sep=sep, encoding="latin1", nrows=0)
    mapa = {c: _norm(c) for c in cab.columns}
    usar = [c for c, n in mapa.items() if n in columnas]
    trozos = []
    for ch in pd.read_csv(ruta, sep=sep, encoding="latin1", usecols=usar, dtype=str, chunksize=chunksize):
        ch = _tipar(ch.rename(columns=mapa))
        if estadisticas is not None:
            estadisticas["filas"] = estadisticas.get("filas", 0) + len(ch)
            if "SIT_FIN" in ch:
                cuenta = estadisticas.setdefault("sit_fin", {})
                for k, v in ch["SIT_FIN"].value_counts().items():
                    cuenta[k] = cuenta.get(k, 0) + int(v)
        if mrun_modulo:
            ch = ch[ch["MRUN"] % mrun_modulo == 0]
        trozos.append(ch)
    df = pd.concat(trozos, ignore_index=True)
    df = df[df["MRUN"].notna()].copy()
    df["MRUN"] = df["MRUN"].astype("int64")
    return df


def guardar_tabla(df: pd.DataFrame, nombre: str) -> Path:
    ruta = DIR_PROC / f"{nombre}.{'parquet' if FORMATO_TABLA == 'parquet' else 'pkl'}"
    (df.to_parquet(ruta, index=False) if FORMATO_TABLA == "parquet" else df.to_pickle(ruta))
    return ruta


def leer_tabla(nombre: str) -> pd.DataFrame:
    ruta = DIR_PROC / f"{nombre}.{'parquet' if FORMATO_TABLA == 'parquet' else 'pkl'}"
    return pd.read_parquet(ruta) if FORMATO_TABLA == "parquet" else pd.read_pickle(ruta)


def guardar_json(obj, nombre: str) -> Path:
    ruta = DIR_RES / nombre
    ruta.write_text(json.dumps(obj, indent=2, ensure_ascii=False,
                               default=lambda o: float(o) if hasattr(o, "__float__") else str(o)),
                    encoding="utf-8")
    return ruta


# -------------------------------------------------------------------------- limpieza
def normalizar_promedio(p: pd.Series) -> pd.Series:
    """Notas guardadas sin separador (55 -> 5.5) se dividen por 10. Fuera de 1.0-7.0 -> NaN.
    Un 0.0 NO se trata como nota: es 'sin dato' (en 2024+ la Educacion Especial viene sin notas)."""
    p = p.astype("float32")
    p = p.where(~(p > 10), p / 10.0)
    return p.where((p >= 1.0) & (p <= 7.0))


def objetivo_desde_sit(sit: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=sit.index, dtype="float32")
    out[sit.isin(CODIGOS_NEGATIVOS)] = 0.0
    out[sit.isin(CODIGOS_POSITIVOS)] = 1.0
    return out


def cargar_rendimiento(anio: int):
    ruta = ubicar_archivo("rend", anio)
    if ruta is None:
        raise FileNotFoundError(f"No hay archivo de Rendimiento {anio} en {DIR_RAW} ni en {DIR_MUESTRA}")
    df = leer_csv(ruta, ["MRUN", "AGNO", "PROM_GRAL", "SIT_FIN", "SIT_FIN_R", "ASISTENCIA", "COD_ENSE2"])
    if "COD_ENSE2" not in df:
        df["COD_ENSE2"] = np.int16(-1)
    col_sit = COLUMNA_SIT if COLUMNA_SIT in df else "SIT_FIN"
    df["sit"] = df[col_sit]
    df["prom"] = normalizar_promedio(df["PROM_GRAL"])
    df["asis"] = df["ASISTENCIA"].where(df["ASISTENCIA"] > 0) if "ASISTENCIA" in df else np.nan
    df["objetivo"] = objetivo_desde_sit(df["sit"])
    info = {
        "anio": anio, "archivo": ruta.name, "filas": int(len(df)),
        "prom_cero_en_origen": int((df["PROM_GRAL"] == 0).sum()),
        "prom_sin_dato_tras_normalizar": int(df["prom"].isna().sum()),
        "asistencia_sin_dato": int(df["asis"].isna().sum()),
    }
    return df, info


def resolver_duplicados(df: pd.DataFrame):
    """Un estudiante puede tener varios registros en un anio (traslados, doble matricula).
    Regla: se prefiere un registro con situacion final valida (P/R/Y); si hay varios, el primero del archivo."""
    vc = df["MRUN"].value_counts()
    valido = df["objetivo"].notna()
    conflicto = df[valido].groupby("MRUN")["objetivo"].nunique()
    info = {
        "mrun_unicos": int(len(vc)),
        "mrun_con_mas_de_un_registro": int((vc > 1).sum()),
        "registros_sobrantes": int(len(df) - len(vc)),
        "mrun_con_registros_validos_contradictorios": int((conflicto > 1).sum()),
    }
    out = (df.assign(_v=valido).sort_values("_v", ascending=False, kind="stable")
             .drop_duplicates("MRUN", keep="first").drop(columns="_v"))
    return out, info


def cargar_matricula(anio: int):
    ruta = ubicar_archivo("mat", anio)
    if ruta is None:
        raise FileNotFoundError(f"No hay archivo de Matricula {anio} en {DIR_RAW} ni en {DIR_MUESTRA}")
    df = leer_csv(ruta, ["MRUN", "AGNO", "GEN_ALU", "COD_ENSE", "COD_ENSE2"])
    if "COD_ENSE2" not in df:
        df["COD_ENSE2"] = np.int16(-1)
    n0 = len(df)
    df = df.drop_duplicates("MRUN", keep="first")
    return df, {"anio": anio, "archivo": ruta.name, "filas": int(n0), "registros_sobrantes": int(n0 - len(df))}


def construir_cohorte(anio: int):
    """Cohorte del anio t: objetivo en t, predictores de rendimiento en t-1, estructura de matricula en t.
    Devuelve (dataset, info, extras)."""
    rt, i_t = cargar_rendimiento(anio)
    rt, d_t = resolver_duplicados(rt)
    rp, i_p = cargar_rendimiento(anio - 1)
    rp, d_p = resolver_duplicados(rp)
    mt, i_m = cargar_matricula(anio)

    n_unicos_t = len(rt)
    audit = rt[["MRUN", "prom", "objetivo"]].rename(columns={"prom": "prom_t"})  # solo para demostrar la fuga
    rt = rt[rt["objetivo"].notna()]
    n_con_obj = len(rt)

    base = rt[["MRUN", "objetivo"]].merge(
        mt[["MRUN", "GEN_ALU", "COD_ENSE", "COD_ENSE2"]], on="MRUN", how="left", validate="one_to_one",
        indicator="_mat")
    # Quien tiene rendimiento en t pero NO matricula en t suele haberse retirado antes de la fecha de corte:
    # su ausencia depende del resultado, asi que usarlo como categoria "-1" filtraria el objetivo. Se excluye y se cuenta.
    sin_mat = base["_mat"] == "left_only"
    n_sin_mat = int(sin_mat.sum())
    tasa_sin_mat = float(base.loc[sin_mat, "objetivo"].mean()) if n_sin_mat else float("nan")
    base = base[~sin_mat].drop(columns="_mat")
    prev = pd.DataFrame({
        "MRUN": rp["MRUN"].values,
        "prom_notas_norm_prev": rp["prom"].values,
        "prom_prev_faltante": rp["prom"].isna().astype("int8").values,
        "riesgo_academico_prev": (rp["prom"] < 4.0).astype("int8").values,   # NaN < 4 -> False; la ausencia va aparte
        "asistencia_prev": rp["asis"].values,
        "asistencia_prev_faltante": rp["asis"].isna().astype("int8").values,
        "repro_prev": (rp["sit"] == "R").astype("int8").values,
        "retiro_prev": (rp["sit"] == "Y").astype("int8").values,
    })
    m = base.merge(prev, on="MRUN", how="left", validate="one_to_one", indicator=True)
    sin_prev = m["_merge"] == "left_only"
    final = m[~sin_prev].drop(columns="_merge").copy()
    final.insert(1, "AGNO", np.int16(anio))
    final = final.rename(columns={"objetivo": OBJETIVO})
    final[OBJETIVO] = final[OBJETIVO].astype("int8")
    for c in ("prom_prev_faltante", "asistencia_prev_faltante", "riesgo_academico_prev", "repro_prev", "retiro_prev"):
        final[c] = final[c].astype("int8")
    for c in ("GEN_ALU", "COD_ENSE", "COD_ENSE2"):
        final[c] = final[c].fillna(-1).astype(int).astype(str)

    tasa = lambda x: float(x.mean()) if len(x) else float("nan")
    info = {
        "anio": anio,
        "filas_rendimiento_t": i_t["filas"], "mrun_unicos_t": n_unicos_t,
        "registros_sobrantes_t": d_t["registros_sobrantes"],
        "contradictorios_t": d_t["mrun_con_registros_validos_contradictorios"],
        "excluidos_sin_situacion_valida_(traslado_u_otro)": n_unicos_t - n_con_obj,
        "con_objetivo": n_con_obj,
        "sin_registro_en_t_menos_1": int(sin_prev.sum()),
        "dataset_final": int(len(final)),
        "tasa_objetivo_dataset_final": tasa(final[OBJETIVO]),
        "tasa_objetivo_sin_t_menos_1": tasa(m.loc[sin_prev, "objetivo"]),
        "prom_t_menos_1_faltante_pct": float(final["prom_prev_faltante"].mean() * 100),
        "excluidos_sin_matricula_t": n_sin_mat,
        "tasa_objetivo_sin_matricula_t": tasa_sin_mat,
        "dup_t_menos_1": d_p, "dup_t": d_t, "matricula": i_m,
    }
    extras = {
        "sin_prev_por_nivel": m.loc[sin_prev, "COD_ENSE2"].value_counts(dropna=False),
        "auditoria_fuga": audit[audit["objetivo"].notna()].copy(),
        "info_rend_t": i_t, "info_rend_prev": i_p,
    }
    return final, info, extras


# ------------------------------------------------------------------------ estadistica
def cramers_v(a: pd.Series, b: pd.Series) -> float:
    from scipy.stats import chi2_contingency
    tabla = pd.crosstab(a, b)
    if min(tabla.shape) < 2:
        return 0.0
    chi2 = chi2_contingency(tabla, correction=False)[0]
    n = tabla.values.sum()
    return float(np.sqrt(chi2 / (n * (min(tabla.shape) - 1))))


def auc_orientado(x: pd.Series, y: pd.Series) -> float:
    from sklearn.metrics import roc_auc_score
    ok = x.notna()
    if ok.sum() < 50 or y[ok].nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y[ok], x[ok]))


def razon_de_riesgo(y: pd.Series, expuesto: pd.Series) -> dict:
    """RR de tener objetivo=1 entre expuestos vs no expuestos, con IC95% (log) y chi2."""
    from scipy.stats import chi2_contingency
    a, n1 = int(y[expuesto == 1].sum()), int((expuesto == 1).sum())
    c, n0 = int(y[expuesto == 0].sum()), int((expuesto == 0).sum())
    rr = (a / n1) / (c / n0)
    se = np.sqrt(1 / a - 1 / n1 + 1 / c - 1 / n0)
    chi2, p, _, _ = chi2_contingency(pd.crosstab(expuesto, y), correction=False)
    return {"tasa_expuestos": a / n1, "tasa_no_expuestos": c / n0, "RR": float(rr),
            "IC95": [float(rr * np.exp(-1.96 * se)), float(rr * np.exp(1.96 * se))],
            "n_expuestos": n1, "n_no_expuestos": n0, "chi2": float(chi2), "p_valor": float(p)}


def _estratos_de(df: pd.DataFrame, var: str):
    anio = df["AGNO"].astype(str)
    if var == "GEN_ALU":
        return anio + "|nivel" + df["COD_ENSE2"]
    return anio + "|gen" + df["GEN_ALU"]


def ficha_candidatas(df: pd.DataFrame, min_estrato: int = 5000, semilla: int = SEMILLA) -> pd.DataFrame:
    """Ficha de variables candidatas CALCULADA sobre el dataset (no escrita a mano)."""
    y = df[OBJETIVO]
    muestra = df.sample(min(300_000, len(df)), random_state=semilla)
    num_bin = [v for v, t in TIPOS.items() if t in ("num", "bin")]
    rho = muestra[num_bin].rank().corr()  # Spearman
    filas = []
    for var, tipo in TIPOS.items():
        x = df[var]
        faltante = float(x.isna().mean() * 100) if tipo != "cat" else float((x == "-1").mean() * 100)
        if tipo in ("num", "bin"):
            medida, valor = "AUC univariado", auc_orientado(x, y)
            fuerza_global = abs(valor - 0.5)
            signo = np.sign(valor - 0.5)
        else:
            medida, valor = "V de Cramer", cramers_v(x, y)
            fuerza_global, signo = valor, 1.0
        # sobrevive a la segmentacion: misma direccion y >= 50% de la fuerza global en todos los estratos grandes
        est = _estratos_de(df, var)
        buenos, total = 0, 0
        for _, idx in df.groupby(est).groups.items():
            sub = df.loc[idx]
            if len(sub) < min_estrato or sub[OBJETIVO].sum() < 200 or sub[OBJETIVO].nunique() < 2:
                continue
            total += 1
            if tipo in ("num", "bin"):
                v = auc_orientado(sub[var], sub[OBJETIVO])
                ok = (not np.isnan(v)) and np.sign(v - 0.5) == signo and abs(v - 0.5) >= 0.5 * fuerza_global
            else:
                ok = cramers_v(sub[var], sub[OBJETIVO]) >= 0.5 * fuerza_global
            buenos += int(ok)
        if fuerza_global < 0.01:
            sobrevive = "n/d (senal global casi nula)"
        else:
            sobrevive = "Si" if total and buenos == total else f"No ({buenos}/{total} estratos)" if total else "n/d"
        # redundancia
        if tipo in ("num", "bin"):
            otros = rho[var].drop(var).abs()
            red = ", ".join(f"{k} (|rho|={v:.2f})" for k, v in otros[otros > 0.9].items()) or "Ninguna"
        elif var == "COD_ENSE2":
            red = f"COD_ENSE (V={cramers_v(muestra['COD_ENSE'], muestra['COD_ENSE2']):.2f})"
        else:
            red = "Ninguna"
        filas.append({"variable": var, "tipo": tipo, "medida": medida, "valor": round(valor, 4),
                      "n": int(len(df)), "pct_faltante": round(faltante, 2),
                      "sobrevive_segmentacion": sobrevive, "redundante_con": red,
                      "riesgo_fuga": "Bajo (disponible al inicio del anio t)"})
    ficha = pd.DataFrame(filas)
    # decision: sale la derivada (COD_ENSE2 se usa solo para segmentar) y la mas debil de un par redundante
    fuerza = {r.variable: abs(r.valor - 0.5) if r.tipo != "cat" else r.valor for r in ficha.itertuples()}
    dec, why = [], []
    for r in ficha.itertuples():
        if r.variable == "COD_ENSE2":
            dec.append("Fuera"); why.append("Derivada de COD_ENSE; se usa solo para segmentar")
        elif r.redundante_con != "Ninguna" and any(fuerza.get(p.split(" ")[0], -1) > fuerza[r.variable] for p in r.redundante_con.split(", ")):
            dec.append("Fuera"); why.append("Redundante con una variable mas informativa")
        elif fuerza[r.variable] < 0.01:
            dec.append("Entra"); why.append("Senal marginal; se mantiene como control (sin riesgo de fuga)")
        else:
            dec.append("Entra"); why.append("Informativa y sin riesgo de fuga")
    ficha["decision"], ficha["por_que"] = dec, why
    return ficha


# ------------------------------------------------------------------------------ modelado
def columnas_por_tipo(variables):
    return ([v for v in variables if TIPOS[v] == "num"], [v for v in variables if TIPOS[v] == "bin"],
            [v for v in variables if TIPOS[v] == "cat"])


def preprocesador(variables, escalar: bool = False):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    num, bin_, cat = columnas_por_tipo(variables)
    trans = []
    if num:
        pasos = [("imp", SimpleImputer(strategy="median"))] + ([("esc", StandardScaler())] if escalar else [])
        trans.append(("num", Pipeline(pasos), num))
    if bin_:
        trans.append(("bin", "passthrough", bin_))
    if cat:
        trans.append(("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.005,
                                           sparse_output=False, dtype=np.float32), cat))
    return ColumnTransformer(trans, sparse_threshold=0.0)


def modelos(variables, semilla: int = SEMILLA):
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.tree import DecisionTreeClassifier
    return {
        "Baseline (clase mayoritaria)": make_pipeline(preprocesador(variables), DummyClassifier(strategy="most_frequent")),
        "Baseline (aleatorio estratificado)": make_pipeline(preprocesador(variables), DummyClassifier(strategy="stratified", random_state=semilla)),
        "Regresion logistica": make_pipeline(preprocesador(variables, escalar=True),
                                             LogisticRegression(C=1.0, class_weight="balanced", max_iter=300)),
        "Arbol de decision": make_pipeline(preprocesador(variables),
                                           DecisionTreeClassifier(max_depth=8, min_samples_leaf=200, class_weight="balanced", random_state=semilla)),
        "Bosque aleatorio": make_pipeline(preprocesador(variables),
                                          RandomForestClassifier(n_estimators=100, max_depth=14, min_samples_leaf=100,
                                                                 class_weight="balanced_subsample", n_jobs=-1, random_state=semilla)),
        "HistGradientBoosting": make_pipeline(preprocesador(variables),
                                              HistGradientBoostingClassifier(class_weight="balanced", early_stopping=True, random_state=semilla)),
    }


def metricas_en_umbral(y, p, umbral: float) -> dict:
    y = np.asarray(y); pred = (np.asarray(p) >= umbral).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"umbral": float(umbral), "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": prec, "recall": rec, "f1": f1, "tasa_alertas": (tp + fp) / len(y)}


def umbral_por_costo(y, p, razon_fn_fp: float):
    """Umbral que minimiza razon*FN + FP (un FN cuesta 'razon' veces un FP). Se elige SOLO con datos de entrenamiento."""
    y = np.asarray(y); p = np.asarray(p)
    grid = np.unique(np.quantile(p, np.linspace(0.01, 0.99, 197)))
    costos = []
    for t in grid:
        pred = p >= t
        costos.append(razon_fn_fp * int(((~pred) & (y == 1)).sum()) + int((pred & (y == 0)).sum()))
    return float(grid[int(np.argmin(costos))]), float(min(costos))


def recall_precision_top_k(y, p, k: float) -> dict:
    y = np.asarray(y); p = np.asarray(p)
    n = max(1, int(round(k * len(y))))
    top = np.argsort(-p)[:n]
    tp = int(y[top].sum())
    return {"k": k, "alertas": n, "recall": tp / max(1, int(y.sum())), "precision": tp / n}


def tamanos():
    """Tamanos de muestra para validacion cruzada y busqueda (en modo rapido son mucho menores)."""
    return {"cv": 3_000 if MODO_RAPIDO else 300_000,
            "anidada": 2_000 if MODO_RAPIDO else 150_000,
            "n_iter": 2 if MODO_RAPIDO else 8,
            "folds": 3 if MODO_RAPIDO else 5}
