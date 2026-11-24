import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

results = [json.loads(f.read_text()) for f in sorted((Path(sys.argv[1]) / 'scenes').rglob('*.json'))]
print('episodes', len(results))
groups = defaultdict(list)
for row in results:
    groups[row['kind'], row['mode']].append(row)

for (kind, mode), rows in sorted(groups.items()):
    print(kind, mode, len(rows), 'safe', sum(round(r['success_rate']*r['agents']) for r in rows))
    firsts = []
    for r in rows:
        seq = np.array(list(r['first_contact_ticks'].values()))
        firsts.extend(np.min(np.where(seq < 0, np.inf, seq), axis=0) * r['dt'])
    v = np.asarray(firsts)
    count = len(v)
    print('  first_any_counts at zero, by0.5, by1, by1.5, by3, by15 / totals:',
          [int((v <= t+1e-9).sum()) for t in (0,.5,1,1.5,3,15.01)], count)
    print('  first_by1 neighbor, obstacle, room:',
          [sum(sum(0 <= t*r['dt'] <= 1+1e-9 for t in r['first_contact_ticks'][c]) for r in rows)
           for c in ('neighbor','obstacle','room')])
    starts = np.concatenate([r['nearest_start'] for r in rows])
    print('  typical nearest_start', np.min(starts), np.median(starts), 'threshold', rows[0]['collision_threshold'],
          'minimum wall clearance', min(min(r['start_clearance']) for r in rows))
    if mode.startswith('peer'):
        d = np.sum([r['signal'] for r in rows], axis=0)
        ticks = sum(r['signal_ticks'] for r in rows)
        print('  decisions on/off/different/only_messages/replace_neighbor_messages/saturation',
              (100*d[[1,2,3,4,5,8]]/d[0]).round(4).tolist())
        print('  logit_change messages/flag, mean raw bid std and abs',
              (d[6:8]/d[0]).round(4).tolist(), (d[9:]/ticks).round(4).tolist())
        events = 0
        pre = 0
        for r in rows:
            ts = r['first_contact_ticks']['neighbor']
            events += sum(0 <= t*r['dt'] <= 1.5+1e-9 for t in ts)
            pre += sum(0 <= h < t and t*r['dt'] <= 1.5+1e-9
                       for h, t in zip(r['first_hold_ticks'], ts))
        print('  drone first neighbor by1.5, had already requested a hold', events, pre)
for r in results:
    if r['layout_id'] in ('narrow_open','narrow_deadend','straight_open','zigzag_open'):
        first = np.array(r['first_contact_ticks']['neighbor'])
        ts = first[first>=0]*r['dt']
        print(r['layout_id'],r['mode'], 'span', np.round(r['start_span_xyz'], 3).tolist(),
              'first neighbor', min(ts,default=-1), 'by1', int((ts<=1+1e-9).sum()),
              'by1.5', int((ts<=1.5+1e-9).sum()), 'safe',round(r['success_rate']*r['agents']))
