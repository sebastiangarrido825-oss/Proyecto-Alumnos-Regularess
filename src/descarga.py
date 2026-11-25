"""Descarga opcional de las versiones oficiales. Los CSV entregados ya se usaron para calcular el informe."""
import shutil, subprocess, tempfile, urllib.request, hashlib
from .ejecucion import RAW, YEARS, file, RES
import pandas as pd
URLS={('Matr',2023):'https://datosabiertos.mineduc.cl/wp-content/uploads/2023/10/Matricula-por-estudiante-2023.rar',('Matr',2024):'https://datosabiertos.mineduc.cl/wp-content/uploads/2024/11/Matricula-por-estudiante-2024.rar',('Matr',2025):'https://datosabiertos.mineduc.cl/wp-content/uploads/2025/10/Matricula-por-estudiante-2025.rar',('Rendimiento',2023):'https://datosabiertos.mineduc.cl/wp-content/uploads/2024/09/Rendimiento-2023.rar',('Rendimiento',2024):'https://datosabiertos.mineduc.cl/wp-content/uploads/2025/04/Rendimiento_2024.rar',('Rendimiento',2025):'https://datosabiertos.mineduc.cl/wp-content/uploads/2026/03/Rendimiento-por-estudiante-2025.rar'}
def descargar_faltantes():
    RAW.mkdir(parents=True,exist_ok=True)
    for (tipo,anio),url in URLS.items():
        try: file(tipo,anio); continue
        except ValueError: pass
        cmd=shutil.which('7z') or shutil.which('unrar')
        if not cmd: raise RuntimeError('Instale 7-Zip o unrar, o descomprima archivos_csv.rar en data/raw.')
        with tempfile.TemporaryDirectory() as tmp:
            from pathlib import Path
            tmp=Path(tmp); rar=tmp/'origen.rar'; urllib.request.urlretrieve(url,rar)
            if 'unrar' in cmd: subprocess.run([cmd,'x','-o+',str(rar),str(tmp)+'/'],check=True)
            else: subprocess.run([cmd,'x','-y',f'-o{tmp}',str(rar)],check=True)
            csvs=[p for p in tmp.rglob('*') if p.suffix.lower()=='.csv' and f'_{anio}_' in p.name and tipo.lower() in p.name.lower()]
            if len(csvs)!=1: raise ValueError(f'Archivos ambiguos: {csvs}')
            shutil.copy2(csvs[0],RAW/csvs[0].name)
    # Identidad de versiones: manifiesto inmutable de la ejecución de referencia.
    ref=RES/'fuentes_referencia.csv'
    if ref.exists():
        for r in pd.read_csv(ref).itertuples():
            p=file(r.tipo,r.anio); h=hashlib.sha256()
            with p.open('rb') as f:
                for b in iter(lambda:f.read(8388608),b''): h.update(b)
            if h.hexdigest()!=r.sha256: raise ValueError(f'Versión diferente de {p.name}; use los CSV entregados originalmente.')
