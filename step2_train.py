import sys, warnings, logging, pickle
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
logging.disable(logging.CRITICAL)

import pandas as pd, numpy as np, joblib
from pathlib import Path
from classifier import build_feature_matrix, find_recall_threshold
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, classification_report

with open('/tmp/frames.pkl', 'rb') as f:
    frames = pickle.load(f)

train_months = ['2025-10','2025-11','2025-12','2026-01']
df_tr = pd.concat([frames[m] for m in train_months], ignore_index=True).sort_values('date_str').reset_index(drop=True)
print('TRAIN: %d rows  %d issues' % (len(df_tr), int(df_tr['is_issue'].sum())))

X, tfidf = build_feature_matrix(df_tr, fit=True)
y = df_tr['is_issue'].values
w = min((y==0).sum() / max((y==1).sum(), 1), 50)
print('Class weight ratio: %.0f:1' % w)

base = LogisticRegression(C=1.0, class_weight={0:1, 1:w}, max_iter=1000, solver='saga', n_jobs=1, random_state=42)
clf = CalibratedClassifierCV(base, method='sigmoid', cv=3)

cv_r, cv_a = [], []
for fold, (ti, vi) in enumerate(TimeSeriesSplit(n_splits=4).split(X)):
    if y[vi].sum() == 0: continue
    clf.fit(X[ti], y[ti])
    p = clf.predict_proba(X[vi])[:, 1]
    t = find_recall_threshold(y[vi], p)
    yp = (p >= t).astype(int)
    r2 = classification_report(y[vi], yp, output_dict=True, zero_division=0).get('1', {}).get('recall', 0)
    a = roc_auc_score(y[vi], p)
    cv_r.append(r2); cv_a.append(a)
    print('  Fold %d: recall=%.3f  auc=%.3f  threshold=%.4f' % (fold, r2, a, t))

clf.fit(X, y)
p_all = clf.predict_proba(X)[:, 1]
final_t = find_recall_threshold(y, p_all)
print()
print('CV Recall:  %.3f +/- %.3f' % (np.mean(cv_r), np.std(cv_r)))
print('CV AUC:     %.3f' % np.mean(cv_a))
print('Threshold:  %.4f' % final_t)

Path('models_6mo').mkdir(exist_ok=True)
joblib.dump(clf,     'models_6mo/classifier.joblib')
joblib.dump(tfidf,   'models_6mo/tfidf.joblib')
joblib.dump(final_t, 'models_6mo/threshold.joblib')

# top issue terms
coef = clf.calibrated_classifiers_[-1].estimator.coef_[0]
names = list(tfidf.get_feature_names_out())
top = [names[i] for i in np.argsort(coef[:len(names)])[-25:][::-1]]
print()
print('Top 25 issue-predictive terms (learned from 4 months of data):')
for i in range(0, 25, 5):
    print('  ' + ', '.join(top[i:i+5]))

joblib.dump({'top_terms': top}, 'models_6mo/top_terms.joblib')
print()
print('Model saved to models_6mo/')
