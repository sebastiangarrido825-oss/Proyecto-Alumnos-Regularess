from .ejecucion import *
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold,GridSearchCV,cross_val_predict
from sklearn.metrics import average_precision_score,roc_auc_score,confusion_matrix,precision_recall_curve
from sklearn.base import clone
import joblib
from .analisis import graph,plt

def pipe(name,vs):
    cats=[v for v in vs if v=='COD_ENSE']; nums=[v for v in vs if v not in cats]; prep=ColumnTransformer([('num',make_pipeline(SimpleImputer(strategy='median'),StandardScaler()),nums),('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False,dtype=np.float32),cats)],remainder='drop')
    m={'Logistica':LogisticRegression(max_iter=200,C=1,random_state=SEED),'Arbol':DecisionTreeClassifier(random_state=SEED,min_samples_leaf=100),'Bosque':RandomForestClassifier(n_estimators=35,max_depth=10,min_samples_leaf=60,n_jobs=2,random_state=SEED),'HistGradientBoosting':HistGradientBoostingClassifier(max_iter=70,early_stopping=False,random_state=SEED),'Mayoritaria':DummyClassifier(strategy='most_frequent'),'Estratificado':DummyClassifier(strategy='stratified',random_state=SEED)}[name]
    return make_pipeline(prep,m)
GRIDS={'Logistica':{'logisticregression__C':[.1,1]},'Arbol':{'decisiontreeclassifier__max_depth':[5,9]},'Bosque':{'randomforestclassifier__max_depth':[7,10]},'HistGradientBoosting':{'histgradientboostingclassifier__max_leaf_nodes':[15,31]}}
def search(name,vs,cv): return GridSearchCV(pipe(name,vs),GRIDS[name],scoring='average_precision',cv=cv,n_jobs=1,refit=True)
def metrics(y,p,t):
    tn,fp,fn,tp=confusion_matrix(y,p>=t,labels=[0,1]).ravel(); prec=tp/max(1,tp+fp); rec=tp/max(1,tp+fn); return dict(umbral=float(t),TN=int(tn),FP=int(fp),FN=int(fn),TP=int(tp),precision=float(prec),recall=float(rec),f1=float(2*prec*rec/max(1e-15,prec+rec)),tasa_alertas=float((tp+fp)/len(y)),FNR=float(fn/max(1,fn+tp)))
def cost_threshold(y,p,r):
    order=np.argsort(-p,kind='stable'); ps=p[order]; ys=y[order]; ends=np.r_[np.flatnonzero(ps[:-1]!=ps[1:]),len(ps)-1]; tp=np.cumsum(ys)[ends]; fp=ends+1-tp; costs=r*(y.sum()-tp)+fp; j=int(np.argmin(costs)); return float(ps[ends[j]]) if costs[j]<r*y.sum() else float(np.nextafter(p.max(),np.inf))
def top(p,k):
    n=int(round(len(p)*k)); idx=np.argsort(-p,kind='stable')[:n]; a=np.zeros(len(p),bool); a[idx]=True; return a

def modelar():
    train=pd.read_pickle(PROC/'cohorte_2024.pkl'); # muestra aleatoria simple reproducible para ajuste y CV, universo completo en test
    n=min(int(os.environ.get('N_TRAIN','60000')),len(train)); d=train.sample(n,random_state=SEED).reset_index(drop=True); del train; X=d[VARS]; y=d[Y].to_numpy(); outer=StratifiedKFold(3,shuffle=True,random_state=SEED); splits=list(outer.split(X,y)); inner=StratifiedKFold(2,shuffle=True,random_state=43)
    allnames=list(GRIDS)+['Mayoritaria','Estratificado']; scores=[]; oof=np.zeros(n); selected=[]
    for f,(tr,va) in enumerate(splits,1):
        best=None; best_score=-1
        for name in allnames:
            model=search(name,VARS,inner) if name in GRIDS else pipe(name,VARS); model.fit(X.iloc[tr],y[tr]); p=model.predict_proba(X.iloc[va])[:,1]; met=metrics(y[va],p,.5); scores.append(dict(fold=f,modelo=name,AP=average_precision_score(y[va],p),ROC_AUC=roc_auc_score(y[va],p),**met));
            if name in GRIDS and model.best_score_>best_score: best_score=model.best_score_; best=(name,p)
        oof[va]=best[1]; selected.append(best[0]); print('Fold',f,selected[-1],flush=True)
    sc=pd.DataFrame(scores); sc.to_csv(RES/'cv_pliegues.csv',index=False); summary=sc.groupby('modelo')[['AP','ROC_AUC','f1','recall','precision']].agg(['mean','std']); summary.to_csv(RES/'cv_resumen.csv');
    finals={name:search(name,VARS,inner).fit(X,y) for name in GRIDS}; winner=max(finals,key=lambda k:finals[k].best_score_); final=finals[winner].best_estimator_; print('Ganador',winner,finals[winner].best_params_,flush=True)
    # Umbral calibrado con OOF de todo el procedimiento, no con test.
    sens=[]
    for r in (2,5,10,20):
        t=cost_threshold(y,oof,r); sens.append(dict(costo_FN_FP=r,**metrics(y,oof,t)))
    t=sens[2]['umbral']; pd.DataFrame(sens).to_csv(RES/'sensibilidad.csv',index=False)
    # H2/H3: comparaciones pareadas con regresión logística fija, sin selección por 2025.
    def pred(vs,xx): return cross_val_predict(pipe('Logistica',vs),xx[vs],y,cv=splits,method='predict_proba',n_jobs=1)[:,1]
    A=['prom_notas_norm_prev','prom_prev_faltante']; B=A+['COD_ENSE']; pa=pred(A,d); pb=pred(B,d); pc=pred(VARS,d); bad=d.copy(); bad['prom_notas_norm_prev']=bad.prom_notas_norm_prev.fillna(0); vb=[v for v in VARS if v!='prom_prev_faltante']; pi=pred(vb,bad); neg=(y==0)&d.prom_prev_faltante.eq(1).to_numpy(); pp=top(pc,.15); ip=top(pi,.15)
    h2=[dict(fold=i+1,solo=average_precision_score(y[va],pa[va]),nivel=average_precision_score(y[va],pb[va])) for i,(_,va) in enumerate(splits)]; h3=dict(n_neg_sin_nota=int(neg.sum()),FPR_propuesto=float(pp[neg].mean()),FPR_cero=float(ip[neg].mean()),AP_propuesto=average_precision_score(y,pc),AP_cero=average_precision_score(y,pi),alertas=int(pp.sum()));
    # La cohorte 2025 se abre por primera vez para evaluar el modelo y umbral congelados.
    test=pd.read_pickle(PROC/'cohorte_2025.pkl'); yt=test[Y].to_numpy(); probs=[]
    for start in range(0,len(test),200000): probs.append(final.predict_proba(test.iloc[start:start+200000][VARS])[:,1])
    p=np.concatenate(probs); mt=metrics(yt,p,t); m5=metrics(yt,p,.5); ap=average_precision_score(yt,p); roc=roc_auc_score(yt,p); fn=(yt==1)&(p<t); valid=test.prom_notas_norm_prev.notna().to_numpy();
    base=metrics(yt,np.zeros(len(yt)),.5); rng=np.random.default_rng(SEED); bs=metrics(yt,(rng.random(len(yt))<y.mean()).astype(float),.5)
    out=dict(n_train=n,positivos_train=int(y.sum()),n_test=len(test),positivos_test=int(yt.sum()),modelo=winner,hiperparametros=finals[winner].best_params_,seleccion_externa=selected,AP_oof=average_precision_score(y,oof),ROC_oof=roc_auc_score(y,oof),AP_test=ap,ROC_test=roc,umbral_costo=mt,umbral_05=m5,baselines={'mayoritaria':base,'estratificado':bs},H2=h2,H3=h3,fn_con_nota_valida=int((fn&valid).sum()),fn_nota_mayor_igual_4=int((fn&test.prom_notas_norm_prev.ge(4).to_numpy()).sum()))
    audit=[]
    for g,inds in test.groupby('GEN_ALU').indices.items():
        audit.append(dict(GEN_ALU=int(g),n=len(inds),positivos=int(yt[inds].sum()),**metrics(yt[inds],p[inds],t)))
    pd.DataFrame(audit).to_csv(RES/'auditoria_genero_test.csv',index=False)
    save(out,'modelo.json'); joblib.dump(final,MOD/'modelo_final.joblib');
    fig,ax=plt.subplots(figsize=(4.8,3.5)); mat=np.array([[mt['TN'],mt['FP']],[mt['FN'],mt['TP']]]); ax.imshow(mat,cmap='Blues');
    for (i,j),v in np.ndenumerate(mat): ax.text(j,i,f'{v:,}',ha='center',va='center',color='white' if v>mat.max()/2 else 'black')
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Pred. 0','Pred. 1'],yticklabels=['Real 0','Real 1'],title='Matriz de confusión · 2025'); fig.tight_layout(); fig.savefig(FIG/'06_matriz.png',dpi=150); plt.close(fig)
    pr,rc,_=precision_recall_curve(yt,p); fig,ax=plt.subplots(figsize=(5.8,3.5)); ax.plot(rc,pr,label=f'AP = {ap:.3f}'); ax.axhline(yt.mean(),ls='--',color='gray',label='Prevalencia'); ax.set(xlabel='Recall',ylabel='Precisión',title='Curva precisión-recall · 2025'); ax.legend(); fig.tight_layout(); fig.savefig(FIG/'07_pr.png',dpi=150); plt.close(fig); print('TEST',out,flush=True)
if __name__=='__main__': modelar()
