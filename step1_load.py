import sys, warnings, logging, pickle
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
logging.disable(logging.CRITICAL)

import pandas as pd
from data_loader import load_month_pair
from features import engineer_features

all_cfg = [
    ('2025-10','/mnt/user-data/uploads/2025-10_RAW.csv','/mnt/user-data/uploads/2025-10_SORTED.xlsx'),
    ('2025-11','/mnt/user-data/uploads/2025-11_RAW.csv','/mnt/user-data/uploads/2025-11_SORTED.xlsx'),
    ('2025-12','/mnt/user-data/uploads/2025-12_RAW.csv','/mnt/user-data/uploads/2025-12_SORTED.xlsx'),
    ('2026-01','/mnt/user-data/uploads/2026-01_RAW.csv','/mnt/user-data/uploads/2026-01_SORTED.xlsx'),
    ('2026-02','/mnt/user-data/uploads/2026-02_RAW.csv','/mnt/user-data/uploads/2026-02_SORTED.xlsx'),
    ('2026-03','/mnt/user-data/uploads/2026-03_RAW.csv','/mnt/user-data/uploads/2026-03_SORTED.xlsx'),
]

frames = {}
for m, r, s in all_cfg:
    df = load_month_pair(r, s)
    df['source_month'] = m
    df = engineer_features(df)
    frames[m] = df
    print('%s: %d rows %d issues (%.1f%%)' % (m, len(df), int(df['is_issue'].sum()), 100*df['is_issue'].mean()))

with open('/tmp/frames.pkl', 'wb') as f:
    pickle.dump(frames, f)
print('Frames saved to /tmp/frames.pkl')
