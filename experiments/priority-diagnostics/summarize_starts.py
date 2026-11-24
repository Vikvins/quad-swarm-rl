import json
from pathlib import Path
from statistics import median
import sys

root = Path(sys.argv[1])
for path in sorted(root.glob('*.json')):
    v = json.loads(path.read_text())
    dt = v['dt']
    neighboring = v['first_contact_ticks']['neighbor']
    contacts = [min((xs[i] for xs in v['first_contact_ticks'].values() if xs[i] >= 0), default=-1)
                for i in range(v['agents'])]
    pairs = [r for r in v['early_pairs'] if r['tick'] * dt <= 1.0+1e-9]
    result = dict(mode=v['mode'], spacing=v['wait_schedule'],
                  first_neighbor=round(min((t*dt for t in neighboring if t>=0),default=99),2),
                  drones_with_neighbor_contact=[sum(0 <= t*dt<=lim+1e-9 for t in neighboring)
                                                 for lim in (1.,1.5,3.,15.01)],
                  any_contact_by3=sum(0 <= t*dt <=3+1e-9 for t in contacts),
                  any_contact_full=sum(t>=0 for t in contacts),
                  safe=round(v['success_rate']*v['agents']),
                  reached=round(v['reached_goal_rate']*v['agents']))
    print(result)
    if v['mode']=='probe_base':
        assert result['safe'] == 10 and result['drones_with_neighbor_contact'][:2] == [42,61]
        print('initial kinematics', v['snapshots_kinematics'])
        for lead in ('before_10ms','before_100ms'):
            print(lead,'pair count',len(pairs),'visibility none/one/both',
                  [sum(sum(row[lead]['visible'])==k for row in pairs) for k in range(3)],
                  'median radial speed', round(median(row[lead]['separation_rate'] for row in pairs),3),
                  'median along y of pair center', round(median(sum(p[1] for p in row[lead]['positions'])/2 for row in pairs),3))
        print('first pair', pairs[0])
