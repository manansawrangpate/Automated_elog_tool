import sys, warnings, logging, pickle
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
logging.disable(logging.CRITICAL)

import pandas as pd, numpy as np, joblib
from pipeline import TriagePipeline

with open('frames.pkl', 'rb') as f:
    frames = pickle.load(f)

clf     = joblib.load('models_6mo/classifier.joblib')
tfidf   = joblib.load('models_6mo/tfidf.joblib')
final_t = joblib.load('models_6mo/threshold.joblib')

pipe = TriagePipeline(model=clf, tfidf=tfidf, threshold=final_t, use_llm=False)

train_months = ['2025-10','2025-11','2025-12','2026-01']
all_months = ['2025-10','2025-11','2025-12','2026-01','2026-02','2026-03']

print('%-12s %-6s %8s %7s %4s %6s %9s %8s' % ('Month','Set','Recall','Prec','FN','Flags','Cleared','Reduce%'))
print('-' * 68)

eval_rows = []
for m in all_months:
    tag = 'TRAIN' if m in train_months else ('VAL' if m=='2026-02' else 'TEST')
    res = pipe.evaluate(frames[m])
    print('  %-10s %-6s %8.4f %7.4f %4d %6d %9d %7.1f%%' % (
        m, tag, res['recall'], res['precision'],
        res['fn'], res['flags_sent_to_human'], res['logs_cleared'], res['reduction_pct']))
    eval_rows.append({'month': m, 'set': tag, **res})

# Save detailed output for Val + Test
df_valtest = pd.concat([frames['2026-02'], frames['2026-03']], ignore_index=True)
results = pipe.run(df_valtest)

out_cols = ['source_month','EVENTDATE','DESCRIPTION','PERSONGROUP','LDTEXT',
            'triage_decision','triage_stage','triage_score','triage_reason',
            'matching_keywords','clf_proba','stage1_label','is_issue']
out_cols = [c for c in out_cols if c in results.columns]

from pathlib import Path
Path('outputs').mkdir(exist_ok=True)

with pd.ExcelWriter('outputs/6month_eval_results.xlsx', engine='openpyxl') as writer:
    results[out_cols].to_excel(writer, sheet_name='All (Val+Test)', index=False)
    results[results['triage_decision']=='FLAG'][out_cols].to_excel(writer, sheet_name='Flagged', index=False)
    results[results['triage_decision']=='CLEAR'][out_cols].to_excel(writer, sheet_name='Cleared', index=False)
    # Issues only
    if 'is_issue' in results.columns:
        results[results['is_issue']==1][out_cols].to_excel(writer, sheet_name='Known Issues', index=False)
    # Summary
    pd.DataFrame(eval_rows).to_excel(writer, sheet_name='Summary', index=False)

print()
print('Saved: /mnt/user-data/outputs/6month_eval_results.xlsx')

# False negative analysis
fn_rows = results[(results.get('is_issue',pd.Series(0,index=results.index))==1) & (results['triage_decision']!='FLAG')]
print()
print('FALSE NEGATIVES on Val+Test: %d' % len(fn_rows))
if len(fn_rows) > 0:
    for _, r in fn_rows.iterrows():
        print('  [%s] prob=%.4f stage=%s kw=[%s]' % (
            str(r.get('LDTEXT',''))[:90], r.get('clf_proba',0),
            r.get('triage_stage','?'), r.get('matching_keywords','')[:50]))

# Stage distribution
print()
print('Stage breakdown (Val+Test):')
for stage, cnt in results['triage_stage'].value_counts().items():
    dec = results[results['triage_stage']==stage]['triage_decision'].value_counts().to_dict()
    print('  %-30s n=%4d  %s' % (stage, cnt, dec))
