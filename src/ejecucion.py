from pathlib import Path
import os, json, hashlib, gc
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]; RAW=Path(os.environ.get('DIR_RAW',ROOT/'data/raw')); RES=ROOT/'results'; FIG=ROOT/'figs'; PROC=ROOT/'data/processed'; MOD=ROOT/'models'
for p in (RES,FIG,PROC,MOD): p.mkdir(parents=True,exist_ok=True)
Y='estudiante_reprobado'; YEARS=[2023,2024,2025]; SEED=42
VARS=['prom_notas_norm_prev','prom_prev_faltante','asistencia_prev','repro_prev','retiro_prev','COD_ENSE']
def save(x,n):
    (RES/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=lambda v:v.item() if hasattr(v,'item') else str(v)),encoding='utf8')
def file(tipo,a):
    fs=[p for p in RAW.glob('*') if p.suffix.lower()=='.csv' and f'_{a}_' in p.name and tipo.lower() in p.name.lower()]
    if len(fs)!=1: raise ValueError(f'Se requiere un CSV de {tipo} {a}: {fs}')
    return fs[0]
def obtener():
    out=[]
    for a in YEARS:
        for tipo in ['Matr','Rendimiento']:
            p=file(tipo,a); h=hashlib.sha256(); n=0
            with p.open('rb') as f:
                for b in iter(lambda:f.read(8388608),b''): h.update(b); n+=b.count(b'\n')
            out.append(dict(anio=a,tipo=tipo,archivo=p.name,filas=n-1,sha256=h.hexdigest(),bytes=p.stat().st_size))
    t=pd.DataFrame(out); t.to_csv(RES/'fuentes.csv',index=False); print(t[['anio','tipo','filas']].to_string(index=False),flush=True)
def read(tipo,a,cols):
    p=file(tipo,a); nums=['MRUN','AGNO','GEN_ALU','COD_ENSE','COD_ENSE2']; types={c:'int32' for c in cols if c in nums}; types.update({c:'category' for c in cols if c in ['SIT_FIN','SIT_FIN_R']}); types.update({c:'str' for c in cols if c in ['PROM_GRAL','ASISTENCIA']})
    d=pd.read_csv(p,sep=';',encoding='utf-8-sig',usecols=cols,dtype=types); assert d.AGNO.eq(a).all(); assert d.MRUN.gt(0).all(); return d.drop(columns='AGNO')
def rendimiento(a):
    d=read('Rendimiento',a,['AGNO','MRUN','PROM_GRAL','ASISTENCIA','SIT_FIN','SIT_FIN_R','COD_ENSE2'])
    for c in ['PROM_GRAL','ASISTENCIA']: d[c]=pd.to_numeric(d[c].str.replace(',','.',regex=False),errors='coerce').astype('float32')
    p=d.PROM_GRAL; d['prom']=p.where(p<10,p/10).where(lambda s:s.between(1,7)); d['asis']=d.ASISTENCIA.where(d.ASISTENCIA.between(0,100)); d['sit']=d.SIT_FIN_R; d['objetivo']=d.sit.map({'P':0,'R':1,'Y':1}).astype('float32')
    pd.crosstab(d.SIT_FIN,d.SIT_FIN_R,dropna=False).to_csv(RES/f'situaciones_{a}.csv'); pd.crosstab(d.COD_ENSE2,p.eq(0)).to_csv(RES/f'ceros_nivel_{a}.csv'); pd.crosstab(d.sit,p.eq(0)).to_csv(RES/f'ceros_situacion_{a}.csv')
    dup=d.MRUN.duplicated(False); multi=d.loc[dup,['MRUN','sit','prom','asis','objetivo']].copy(); valid=multi[multi.objetivo.notna()]; conflict=valid.groupby('MRUN',observed=True).sit.nunique(); bad=conflict[conflict>1].index
    multi['valid']=multi.objetivo.notna(); multi['maxvalid']=multi.groupby('MRUN').valid.transform('max'); candidates=multi[multi.valid==multi.maxvalid]; discord=candidates.groupby('MRUN')[['prom','asis','sit']].nunique(dropna=False).max(axis=1); bad2=discord[discord>1].index; exclude=bad.union(bad2)
    chosen=candidates[~candidates.MRUN.isin(exclude)].drop_duplicates('MRUN').drop(columns=['valid','maxvalid']); u=pd.concat([d.loc[~dup,['MRUN','sit','prom','asis','objetivo']],chosen],ignore_index=True); assert u.MRUN.is_unique
    info=dict(anio=a,filas=len(d),unicos=d.MRUN.nunique(),sobrantes=len(d)-d.MRUN.nunique(),mrun_duplicados=multi.MRUN.nunique(),conflicto_situacion=len(bad),discordantes=len(exclude),ceros=int(p.eq(0).sum()),notas_faltantes=int(d.prom.isna().sum()),situaciones=d.sit.value_counts(dropna=False).to_dict(),asistencia_faltante=int(d.asis.isna().sum()))
    return u,info
def matricula(a):
    d=read('Matr',a,['AGNO','MRUN','GEN_ALU','COD_ENSE','COD_ENSE2']); n=len(d); d=d.drop_duplicates(); dup=d.MRUN.duplicated(False); k=int(d.loc[dup,'MRUN'].nunique()); d=d[~dup]; return d,dict(filas=n,unicos=len(d),conflictos=k)
def preparar():
    diag=[]; cob=[]; prev=None
    for a in YEARS:
        r,i=rendimiento(a); m,mi=matricula(a); i['matricula']=mi; diag.append(i); save(diag,'diagnostico.json'); print('Auditado',a,i,flush=True)
        if prev is not None:
            b=r[r.objetivo.notna()][['MRUN','objetivo']].rename(columns={'objetivo':Y}); n0=len(b); b=b.merge(m,on='MRUN',how='left',validate='one_to_one',indicator=True); no=b._merge.eq('left_only'); nm=int(no.sum()); b=b.loc[~no].drop(columns='_merge')
            hist=prev.rename(columns={'prom':'prom_notas_norm_prev','asis':'asistencia_prev'})[['MRUN','sit','prom_notas_norm_prev','asistencia_prev']].copy(); hist['repro_prev']=hist.sit.eq('R').astype('int8'); hist['retiro_prev']=hist.sit.eq('Y').astype('int8'); hist=hist.drop(columns='sit')
            d=b.merge(hist,on='MRUN',how='left',validate='one_to_one',indicator=True); no=d._merge.eq('left_only'); npv=int(no.sum()); ts=float(d.loc[no,Y].mean()); d=d.loc[~no].drop(columns='_merge'); d['AGNO']=np.int16(a); d[Y]=d[Y].astype('int8'); d['prom_prev_faltante']=d.prom_notas_norm_prev.isna().astype('int8'); d['asistencia_prev_faltante']=d.asistencia_prev.isna().astype('int8'); d['riesgo_academico_prev']=np.where(d.prom_notas_norm_prev.isna(),np.nan,d.prom_notas_norm_prev.lt(4).astype(float)).astype('float32')
            for c in ['GEN_ALU','COD_ENSE','COD_ENSE2']: d[c]=d[c].astype('int16')
            for c in ['repro_prev','retiro_prev']: d[c]=d[c].astype('int8')
            assert d.MRUN.is_unique and len(d)==n0-nm-npv
            c=dict(anio=a,rend_unico=len(r),objetivo_valido=n0,sin_matricula=nm,sin_historial=npv,tasa_sin_historial=ts,n=len(d),positivos=int(d[Y].sum()),tasa=float(d[Y].mean()),notas_prev_faltantes=int(d.prom_prev_faltante.sum()),asistencia_prev_faltante=int(d.asistencia_prev_faltante.sum()),columnas=len(d.columns)); cob.append(c); d.to_pickle(PROC/f'cohorte_{a}.pkl'); print(c,flush=True); del d,b,hist; gc.collect(); save(cob,'cobertura.json'); pd.DataFrame(cob).to_csv(RES/'cobertura.csv',index=False)
        prev=r; del m; gc.collect()
if __name__=='__main__': obtener(); preparar()
