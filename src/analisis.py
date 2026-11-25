from .ejecucion import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from scipy.stats import chi2_contingency

def graph(tab,name,title,xlabel='',ylabel='Tasa positiva (%)'):
    ax=tab.plot(kind='bar',color='#35688c',figsize=(6.8,3.2)); ax.set(title=title,xlabel=xlabel,ylabel=ylabel); ax.tick_params(axis='x',labelrotation=15); ax.spines[['top','right']].set_visible(False); ax.figure.tight_layout(); ax.figure.savefig(FIG/name,dpi=150); plt.close(ax.figure)
def table(d,group):
    return d.groupby(group,observed=True)[Y].agg(n='size',positivos='sum',tasa=lambda x:x.mean()*100).reset_index()
def eda():
    d=pd.read_pickle(PROC/'cohorte_2024.pkl'); y=d[Y]; H={}
    H['n']=len(d); H['positivos']=int(y.sum()); H['tasa']=float(y.mean()); graph(pd.Series({'Promovido':100*(1-y.mean()),'Reprobado / retirado':100*y.mean()}),'01_objetivo.png','Distribución del objetivo · 2024',ylabel='Registros (%)')
    g=pd.Series(np.where(d.prom_prev_faltante.eq(1),'Sin nota',np.where(d.prom_notas_norm_prev.lt(4),'Nota < 4','Nota ≥ 4')),index=d.index,name='grupo'); t=table(d,g); t.to_csv(RES/'nota_vs_objetivo.csv',index=False); graph(t.set_index('grupo').tasa,'02_nota_previa.png','Nota del año anterior y desenlace · 2024'); H['nota']=t.to_dict('records')
    a=t.set_index('grupo'); x=a.loc['Nota < 4']; z=a.loc['Nota ≥ 4']; rr=(x.positivos/x.n)/(z.positivos/z.n); se=np.sqrt(1/x.positivos-1/x.n+1/z.positivos-1/z.n); H['H1']={'RR':rr,'IC95':[rr*np.exp(-1.96*se),rr*np.exp(1.96*se)],'n':int(x.n+z.n)}
    t=table(d,'COD_ENSE2'); t.to_csv(RES/'niveles.csv',index=False); graph(t.set_index('COD_ENSE2').tasa,'03_nivel.png','Tasa por nivel de enseñanza · 2024','Código COD_ENSE2'); H['niveles']=t.to_dict('records')
    f=d.groupby('COD_ENSE2').prom_prev_faltante.agg(n='size',faltantes='sum',pct=lambda x:x.mean()*100); f.to_csv(RES/'faltantes_nivel.csv'); graph(f.pct,'04_faltantes.png','Notas previas faltantes por nivel · 2024','Código COD_ENSE2','Faltantes (%)')
    tramo=pd.cut(d.asistencia_prev,[-1,60,80,90,95,100],labels=['0–60','>60–80','>80–90','>90–95','>95–100']); t=table(d,tramo.rename('tramo')); t.to_csv(RES/'asistencia.csv',index=False); graph(t.set_index('tramo').tasa,'05_asistencia.png','Asistencia previa y desenlace · 2024','Asistencia (%)'); H['asistencia']=t.to_dict('records')
    # H1 por nivel: se comprueba en cada estrato con al menos 200 registros por grupo.
    seg=[]
    for nivel,s in d.groupby('COD_ENSE2'):
        a=s[s.prom_notas_norm_prev.lt(4)][Y]; b=s[s.prom_notas_norm_prev.ge(4)][Y]; seg.append(dict(nivel=int(nivel),n_bajo=len(a),n_alto=len(b),tasa_bajo=float(a.mean()),tasa_alto=float(b.mean()),cumple=bool(a.mean()>b.mean()) if min(len(a),len(b))>=200 else None))
    pd.DataFrame(seg).to_csv(RES/'segmentacion_H1.csv',index=False)
    # Ficha descriptiva: muestra fija 100k para asociaciones, faltantes sobre toda cohorte.
    s=d.sample(min(100000,len(d)),random_state=SEED); rows=[]
    for v in VARS+['asistencia_prev_faltante','riesgo_academico_prev','GEN_ALU','COD_ENSE2']:
        cat=v in ['GEN_ALU','COD_ENSE','COD_ENSE2']; ok=s[v].notna(); x=s.loc[ok,v]; yy=s.loc[ok,Y]
        if cat:
            ct=pd.crosstab(x,yy); val=np.sqrt(chi2_contingency(ct,correction=False)[0]/ct.values.sum()/max(1,min(ct.shape)-1)); med='V Cramer'
        else: val=roc_auc_score(yy,x) if x.nunique()>1 else .5; med='ROC-AUC univ.'
        seg_ok=0; seg_n=0
        for _,sub in s.groupby('COD_ENSE2' if v not in ['COD_ENSE2','COD_ENSE'] else 'GEN_ALU'):
            sub=sub.dropna(subset=[v]);
            if len(sub)<1000 or sub[Y].nunique()<2 or sub[Y].sum()<30 or sub[v].nunique()<2: continue
            if cat:
                ct=pd.crosstab(sub[v],sub[Y]); w=np.sqrt(chi2_contingency(ct,correction=False)[0]/ct.values.sum()/max(1,min(ct.shape)-1)); good=w>=val/2
            else:
                w=roc_auc_score(sub[Y],sub[v]); good=(w-.5)*(val-.5)>0 and abs(w-.5)>=abs(val-.5)/2
            seg_n+=1; seg_ok+=int(good)
        why={'asistencia_prev_faltante':'Constante en entrenamiento (sin faltantes); se excluye','riesgo_academico_prev':'Derivada de nota; excluida para evitar redundancia','GEN_ALU':'Solo auditoría demográfica, no predictor','COD_ENSE2':'Derivada del tipo de enseñanza; solo segmentación'}.get(v,'Disponible antes del desenlace; indicador complementario' if 'faltante' in v else 'Información previa o de matrícula')
        rows.append(dict(variable=v,medida=med,valor=val,n=int(ok.sum()),faltantes_pct=float(d[v].isna().mean()*100),segmentos_apoyan=seg_ok,segmentos_evaluados=seg_n,decision='Entra' if v in VARS else 'Fuera',motivo=why))
    pd.DataFrame(rows).to_csv(RES/'ficha_candidatas.csv',index=False); save(H,'hallazgos.json'); print('EDA',H['H1'],flush=True)
if __name__=='__main__': eda()
