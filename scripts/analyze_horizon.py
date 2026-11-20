"""Compare matched 15s and 30s episodes and verify identical trajectory prefixes."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--main',type=Path,required=True)
    p.add_argument('--extended',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    before=pd.read_csv(a.main/'episodes.csv');after=pd.read_csv(a.extended/'episodes.csv')
    if len(after)!=16:raise ValueError(f'Expected 16 completed long-horizon episodes, found {len(after)}')
    paired=after.merge(before,on=['layout_id','agents','seed'],suffixes=('_30s','_15s'),validate='one_to_one')
    checks=[]
    for row in paired.itertuples():
        rel=Path(f'n{row.agents}')/row.layout_id/f'seed{row.seed:03d}'
        x=np.load(a.main/rel/'trajectory.npz')['positions'];y=np.load(a.extended/rel/'trajectory.npz')['positions']
        error=float(np.max(np.abs(x-y[:len(x)])))
        exact=x.tobytes()==y[:len(x)].tobytes()
        checks.append(dict(agents=int(row.agents),layout_id=row.layout_id,seed=int(row.seed),max_prefix_error=error,exact_bytes_equal=exact))
        if error!=0 or not exact:raise ValueError('15-second prefixes are not identical')
    metrics=['success_rate','reached_goal_rate','collision_rate','mean_final_distance','stalled_rate']
    rows=[]
    for layout,g in paired.groupby('layout_id'):
        r=dict(layout_id=layout,n_pairs=len(g))
        for key in metrics:
            r[key+'_15s']=float(g[key+'_15s'].mean());r[key+'_30s']=float(g[key+'_30s'].mean())
            r[key+'_delta']=r[key+'_30s']-r[key+'_15s']
        rows.append(r)
    result=pd.DataFrame(rows)
    result.to_csv(a.output/'horizon_comparison.csv',index=False,encoding='utf-8-sig')
    paired.to_csv(a.output/'horizon_by_seed.csv',index=False,encoding='utf-8-sig')
    (a.output/'horizon_prefix_checks.json').write_text(json.dumps(checks,indent=2))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    fig,axes=plt.subplots(1,2,figsize=(14,6));y=np.arange(len(result))
    for ax,metric,title in zip(axes,['success_rate','mean_final_distance'],['Успех без контактов, %','Конечное расстояние, м']):
        scale=100 if metric=='success_rate' else 1
        ax.barh(y-.18,result[metric+'_15s']*scale,height=.36,label='15 секунд',color='#2d8194')
        ax.barh(y+.18,result[metric+'_30s']*scale,height=.36,label='30 секунд',color='#c75050')
        ax.set(yticks=y,yticklabels=result.layout_id,title=title);ax.invert_yaxis();ax.grid(axis='x',alpha=.15)
    axes[0].legend();fig.suptitle('Проверка горизонта, N=8, по 2 парных seed. Диагностическая серия.');fig.tight_layout()
    fig.savefig(a.output/'horizon_comparison.png',dpi=170);fig.savefig(a.output/'horizon_comparison.svg');fig.savefig(a.output/'horizon_comparison.pdf');plt.close(fig)
    closed=result[result.layout_id.str.endswith('deadend')]
    text=('# Дополнительный горизонт\n\n16 эпизодов по 30 секунд на 8 дронах: 4 семейства, открытая и закрытая версия, 2 seed.\n'
          'Сравнение выполнено с теми же seed основной серии. Все первые 1502 позиции траектории совпали побайтово.\n'
          'Удлинение горизонта не меняло первые 15 секунд управления. Два seed не дают точной оценки редких успехов.\n\n')
    for row in closed.itertuples():text+=f'- {row.layout_id}: успех {row.success_rate_15s:.3f} -> {row.success_rate_30s:.3f}.\n'
    (a.output/'HORIZON_RESULTS_RU.md').write_text(text,encoding='utf-8')
    print(result.to_string(index=False))


if __name__=='__main__':main()
