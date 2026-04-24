import sys, warnings, logging, pickle
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
logging.disable(logging.CRITICAL)

import pandas as pd
from data_loader import load_month_pair
from features import engineer_features

all_cfg = [
    (
        '2025-10',
        r"C:\Users\Sawrangpatem\OneDrive - York Region\Documents\github\elog_ui\Training Data\2025-10 RAW.csv",
        r"C:\Users\Sawrangpatem\OneDrive - York Region\Documents\github\elog_ui\Training Data\2025-10 SORTED.xlsx"
    ),
    (
        '2025-11',
        r"C:\Users\Sawrangpatem\OneDrive - York Region\Documents\github\elog_ui\Training Data\2025-11 RAW.csv",
        r"C:\Users\Sawrangpatem\OneDrive - York Region\Documents\github\elog_ui\Training Data\2025-11 SORTED.xlsx"
    ),
    (
        '2025-12',
        r"C:\Users\Sawrangpatem\OneDrive - York Region\Documents\github\elog_ui\Training Data\2025-12 RAW.csv",
        r"C:\Users\Sawrangpatem\OneDrive - York Region\Documents\github\elog_ui\Training Data\2025-12 SORTED.xlsx"
    ),
    (
        '2026-01',
        r"C:\Users\Sawrangpatem\OneDrive - York Region\Documents\github\elog_ui\Training Data\2026-01 RAW.csv",
        r"C:\Users\Sawrangpatem\OneDrive - York Region\Documents\github\elog_ui\Training Data\2026-01 SORTED.xlsx"
    ),
    (
        '2026-02',
        r"C:\Users\Sawrangpatem\OneDrive - York Region\Documents\github\elog_ui\Training Data\2026-02 RAW.csv",
        r"C:\Users\Sawrangpatem\OneDrive - York Region\Documents\github\elog_ui\Training Data\2026-02 SORTED.xlsx"
    ),
    (
        '2026-03',
        r"C:\Users\Sawrangpatem\OneDrive - York Region\Documents\github\elog_ui\Training Data\2026-03 RAW.csv",
        r"C:\Users\Sawrangpatem\OneDrive - York Region\Documents\github\elog_ui\Training Data\2026-03 SORTED.xlsx"
    ),
]

frames = {}
for m, r, s in all_cfg:
    df = load_month_pair(r, s)
    df['source_month'] = m
    df = engineer_features(df)
    frames[m] = df
    print('%s: %d rows %d issues (%.1f%%)' % (m, len(df), int(df['is_issue'].sum()), 100*df['is_issue'].mean()))

with open('frames.pkl', 'wb') as f:
    pickle.dump(frames, f)
print('Frames saved to /tmp/frames.pkl')
