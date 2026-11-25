"""Comprobaciones de consistencia de las tablas generadas, sin acceder a microdatos."""
import json
from pathlib import Path
import pandas as pd
R=Path(__file__).resolve().parent/'results'
c=pd.read_csv(R/'cobertura.csv'); m=json.loads((R/'modelo.json').read_text())
assert (c.objetivo_valido-c.sin_matricula-c.sin_historial==c.n).all()
assert (c.n>c.positivos).all()
assert c.loc[c.anio.eq(2025),'n'].iat[0]==m['n_test']
for clave in ('umbral_costo','umbral_05'):
    t=m[clave]; tp,fp,fn,tn=[t[k] for k in ('TP','FP','FN','TN')]
    assert tp+fp+fn+tn==m['n_test'] and tp+fn==m['positivos_test']
    assert abs(tp/max(1,tp+fn)-t['recall'])<1e-12
    assert abs(tp/max(1,tp+fp)-t['precision'])<1e-12
    assert abs(fn/max(1,tp+fn)-t['FNR'])<1e-12
    assert abs(2*tp/max(1,2*tp+fp+fn)-t['f1'])<1e-12
assert pd.read_csv(R/'fuentes.csv').sha256.tolist()==pd.read_csv(R/'fuentes_referencia.csv').sha256.tolist()
print('OK: cobertura, identidades de versiones, matrices, recall, precisión, FNR y F1.')
