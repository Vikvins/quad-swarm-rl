"""Generate complete descriptive statistics, paired contrasts, charts and report."""
import argparse
import base64
import html
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from gym_art.quadrotor_multi.obstacles.suite_layouts import catalogue,FAMILIES

METRICS={
 'success_rate':('Успех без контактов','доля',True),
 'reached_goal_rate':('Достижение цели','доля',True),
 'collision_rate':('Любой контакт','доля',False),
 'obstacle_collision_rate':('Контакт с препятствием','доля',False),
 'neighbor_collision_rate':('Контакт с соседом','доля',False),
 'room_collision_rate':('Контакт с комнатой','доля',False),
 'timeout_rate':('Цель не достигнута','доля',False),
 'stalled_rate':('Застревание','доля',False),
 'swarm_success':('Успех всего роя','доля',True),
 'mean_final_distance':('Конечное расстояние до цели','м',False),
 'mean_path_length':('Длина траектории','м',None),
 'mean_arrival_time':('Время достижения среди достигших','с',None),
 'official_agent_success':('Успех исходного сценария','доля',True),
 'official_collision':('Исходная метрика столкновений проекта','доля',False),
 'mean_sensitivity_xy':('Чувствительность к шуму цели','RMS команды [0,1]',None),
 'mean_command_change_rate':('Скорость изменения команд','1/с',None),
 'mean_command_second_difference':('Вторая разность команд','1/с²',None),
 'mean_mean_change_rate_raw':('Изменение исходного среднего действия','1/с',None),
 'mean_saturation_fraction':('Насыщение команд','доля',None),
 'mean_neighbors_deviation':('Чувствительность к соседям','RMS команды [0,1]',None),
 'mean_obstacles_deviation':('Чувствительность к препятствиям','RMS команды [0,1]',None),
 'mean_total_deviation':('Сумма чувствительностей групп','RMS команды [0,1]',None),
 'mean_obstacle_share':('Доля чувствительности к препятствиям','доля',None),
 'mean_influence_valid':('Доля валидных проб влияния','доля',None),
 'mean_reward':('Награда эпизода','reward',None),
 'inference_batch_ms':('Время actor+critic для батча под параллельной нагрузкой','мс',None),
 'wall_seconds':('Время эпизода на компьютере под параллельной нагрузкой','с',None),
}


def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,(np.integer,)):return int(value)
    if isinstance(value,(float,np.floating)) and not np.isfinite(value):return None
    return value


def ci(values,seed=2026):
    values=np.asarray(values,float);values=values[np.isfinite(values)]
    if not len(values):return np.nan,np.nan
    rng=np.random.default_rng(seed)
    means=rng.choice(values,size=(4000,len(values)),replace=True).mean(axis=1)
    return np.quantile(means,[.025,.975])


def save(fig,path):
    fig.savefig(path.with_suffix('.png'),dpi=170,bbox_inches='tight')
    fig.savefig(path.with_suffix('.svg'),bbox_inches='tight')
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--expected-episodes',type=int,default=432)
    p.add_argument('--agents',type=int,nargs='+',help='Filter completed runs by swarm size')
    p.add_argument('--allow-partial',action='store_true')
    args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    tables=out/'tables';charts=out/'charts';tables.mkdir(exist_ok=True);charts.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    files=sorted(args.input.glob('n*/*/seed*/episode.json'))
    episodes=pd.DataFrame([json.loads(file.read_text()) for file in files])
    if args.agents:
        episodes=episodes[episodes.agents.isin(args.agents)].copy()
    if not args.allow_partial and len(episodes)!=args.expected_episodes:
        raise ValueError(f'Expected {args.expected_episodes} completed episodes, got {len(episodes)}')
    cat=pd.DataFrame(catalogue());order=list(cat.id);names=dict(zip(cat.id,cat.title))
    episodes=episodes.merge(cat[['id','family','has_dead_end','paired_with']],left_on='layout_id',right_on='id',validate='many_to_one').drop(columns='id')
    episodes['official_goal_radius_m']=np.where(episodes.layout_id=='random_baseline',.5,.3)
    if episodes.duplicated(['layout_id','agents','seed']).any():raise ValueError('Duplicate episodes')
    required={'steps','duration','mean_sensitivity_xy','mean_obstacle_share','official_agent_success'}
    if not required.issubset(episodes.columns):raise ValueError('Incomplete diagnostics')
    episodes['duration_error']=np.abs(episodes.duration-15.01)
    if not args.allow_partial and (episodes.duration_error>1e-7).any():raise ValueError('Incomplete episode duration')
    # The unit of uncertainty is an entire episode/seed, never individual drones.
    summary=[];wide=[]
    for (n,layout),group in episodes.groupby(['agents','layout_id'],sort=True):
        row=dict(agents=int(n),layout_id=layout,title=names[layout],episodes=len(group),agent_episodes=int(n*len(group)))
        for metric,(label,unit,direction) in METRICS.items():
            values=group[metric].dropna().to_numpy(float)
            low,high=ci(values)
            mean=float(values.mean()) if len(values) else np.nan
            summary.append(dict(agents=int(n),layout_id=layout,metric=metric,label=label,unit=unit,
                                mean=mean,sd=float(values.std(ddof=1)) if len(values)>1 else np.nan,
                                ci_low=low,ci_high=high,n_episodes=len(values),missing_episodes=int(len(group)-len(values))))
            row[metric]=mean;row[metric+'_ci_low']=low;row[metric+'_ci_high']=high
        wide.append(row)
    stats=pd.DataFrame(summary);wide=pd.DataFrame(wide)
    contrasts=[];paired_seeds=[]
    for n in sorted(episodes.agents.unique()):
        for family in FAMILIES:
            opened=episodes[(episodes.agents==n)&(episodes.layout_id==family+'_open')].set_index('seed')
            closed=episodes[(episodes.agents==n)&(episodes.layout_id==family+'_deadend')].set_index('seed')
            common=opened.index.intersection(closed.index)
            if not len(common):continue
            for metric,(label,unit,_) in METRICS.items():
                before=opened.loc[common,metric];after=closed.loc[common,metric]
                valid=before.notna()&after.notna()
                delta=(after-before)[valid]
                low,high=ci(delta)
                contrasts.append(dict(agents=int(n),family=family,title=FAMILIES[family][0],metric=metric,label=label,unit=unit,
                                      before_mean=float(before[valid].mean()),after_mean=float(after[valid].mean()),
                                      delta=float(delta.mean()),ci_low=low,ci_high=high,n_pairs=len(delta)))
                for seed in delta.index:
                    paired_seeds.append(dict(agents=int(n),family=family,metric=metric,seed=int(seed),before=float(before[seed]),after=float(after[seed]),delta=float(delta[seed])))
    contrast=pd.DataFrame(contrasts)
    agents=[];time_series=[];occupancy={};geometries={}
    for row in episodes.itertuples():
        folder=args.input/f'n{row.agents}'/row.layout_id/f'seed{row.seed:03d}'
        a=pd.read_csv(folder/'agents.csv');a.insert(0,'seed',row.seed);a.insert(0,'layout_id',row.layout_id);a.insert(0,'agents',row.agents);agents.append(a)
        t=pd.read_csv(folder/'timeseries.csv');t.insert(0,'seed',row.seed);t.insert(0,'layout_id',row.layout_id);t.insert(0,'agents',row.agents);time_series.append(t)
        positions=np.load(folder/'trajectory.npz')['positions']
        key=(row.agents,row.layout_id)
        counts=np.histogram2d(positions[...,0].ravel(),positions[...,1].ravel(),bins=140,range=[[-7,7],[-7,7]])[0]
        if key not in occupancy:
            occupancy[key]=counts
            geometries[key]=json.loads((folder/'layout.json').read_text())
        else:occupancy[key]+=counts
    agent_data=pd.concat(agents,ignore_index=True);time_data=pd.concat(time_series,ignore_index=True)
    for frame,name in [(episodes,'episodes'),(stats,'all_metrics_summary'),(wide,'summary'),(contrast,'paired_differences'),
                       (pd.DataFrame(paired_seeds),'paired_by_seed'),(agent_data,'agent_results')]:
        frame.to_csv(tables/f'{name}.csv',index=False,encoding='utf-8-sig')
    time_data.to_csv(tables/'swarm_timeseries.csv.gz',index=False,compression='gzip')
    dictionary=pd.DataFrame([dict(metric=k,label=v[0],unit=v[1],higher_better=v[2]) for k,v in METRICS.items()])
    dictionary.to_csv(tables/'metrics_dictionary.csv',index=False,encoding='utf-8-sig')
    baselines=[]
    for n,g in wide.groupby('agents'):
        b=g[g.layout_id=='random_baseline']
        if not len(b):continue
        b=b.iloc[0]
        for row in g.itertuples():
            for metric in METRICS:
                baselines.append(dict(agents=int(n),layout_id=row.layout_id,metric=metric,
                                      baseline=float(b[metric]),value=getattr(row,metric),difference=getattr(row,metric)-b[metric],
                                      interpretation='Contextual contrast; starts/goals differ. Use paired_differences.csv for matched controls.'))
    pd.DataFrame(baselines).to_csv(tables/'baseline_comparison.csv',index=False,encoding='utf-8-sig')
    for n in sorted(episodes.agents.unique()):
        present=[s for s in order if ((wide.agents==n)&(wide.layout_id==s)).any()]
        from matplotlib.colors import LogNorm
        from matplotlib.patches import Circle
        for layout in present:
            counts=occupancy[n,layout];density=counts/max(1,counts.sum());plan=geometries[n,layout]
            fig,ax=plt.subplots(figsize=(7,7))
            image=ax.imshow(np.ma.masked_where(density.T==0,density.T),origin='lower',extent=(-7,7,-7,7),
                            cmap='magma',norm=LogNorm(vmin=1e-6,vmax=1.))
            if layout!='random_baseline':
                for x,y,_ in plan['obstacles']:
                    ax.add_patch(Circle((x,y),.3,color='#8f9ba6',alpha=.65))
                goals=np.asarray(plan['goals']);ax.scatter(*goals[:,:2].T,c='#23a478',marker='x',s=12,label='Пример целей, seed=0')
                ax.legend(loc='upper right',fontsize=7)
            ax.set(xlim=(-7,7),ylim=(-7,7),aspect='equal',xlabel='x, м',ylabel='y, м',title=f'{names[layout]}\nN={n}: плотность положений по всем seed, вид сверху')
            fig.colorbar(image,ax=ax,shrink=.8,label='Доля сохранённых состояний в ячейке, лог. шкала')
            fig.tight_layout();save(fig,charts/f'n{n}_occupancy_{layout}')
        for metric,(label,unit,_) in METRICS.items():
            g=stats[(stats.agents==n)&(stats.metric==metric)].set_index('layout_id').loc[present]
            fig,ax=plt.subplots(figsize=(10,8))
            values=g['mean'].to_numpy(float)
            mult=100 if unit=='доля' else 1
            colors=['#c75050' if s.endswith('deadend') else '#2d8194' for s in present]
            err=np.maximum(0,np.vstack((values-g.ci_low.to_numpy(),g.ci_high.to_numpy()-values)))*mult
            if np.isfinite(values).any():
                ax.barh(np.arange(len(present)),values*mult,xerr=np.nan_to_num(err,nan=0.),color=colors,alpha=.85,error_kw=dict(lw=.8,capsize=2))
            else:
                ax.text(.5,.5,'Нет наблюдений для этой метрики',transform=ax.transAxes,ha='center')
            ax.set(yticks=np.arange(len(present)),yticklabels=[names[s] for s in present],xlabel=('%' if unit=='доля' else unit),title=f'{label}, N={n}')
            ax.invert_yaxis();ax.grid(axis='x',alpha=.15)
            fig.text(.02,.01,'95% bootstrap по эпизодам. Красный: тупики. Синий: контроль.',fontsize=8)
            fig.tight_layout(rect=(0,.03,1,1));save(fig,charts/f'n{n}_{metric}')
        for metric in ['success_rate','collision_rate','mean_sensitivity_xy','mean_command_change_rate','mean_obstacle_share']:
            g=contrast[(contrast.agents==n)&(contrast.metric==metric)]
            if not len(g):continue
            fig,ax=plt.subplots(figsize=(10,5))
            unit=METRICS[metric][1];mult=100 if unit=='доля' else 1
            values=g.delta.to_numpy(float)
            err=np.maximum(0,np.vstack((values-g.ci_low.to_numpy(),g.ci_high.to_numpy()-values)))*mult
            ax.errorbar(values*mult,np.arange(len(g)),xerr=err,fmt='o',c='#9b3552',capsize=3)
            ax.axvline(0,c='#64748b',lw=1)
            ax.set(yticks=np.arange(len(g)),yticklabels=g.title,xlabel='Изменение: с тупиком минус открытый контроль'+(' (п.п.)' if mult==100 else ''),title=f'{METRICS[metric][0]}: парная разница, N={n}')
            ax.invert_yaxis();ax.grid(axis='x',alpha=.15);fig.tight_layout();save(fig,charts/f'n{n}_paired_{metric}')
        series_metrics=['goal_distance','physical_speed','collision_so_far','sensitivity_xy','command_change_rate','command_second_difference',
                        'saturation_fraction','neighbors_deviation','obstacles_deviation','obstacle_share','total_deviation','influence_valid']
        series_labels=['Расстояние до цели, м','Скорость дрона, м/с','Накопленная доля контактов','Чувствительность к цели','Изменение команд, 1/с',
                       'Вторая разность, 1/с²','Насыщение команд','Влияние соседей','Влияние препятствий','Доля препятствий','Сумма влияний','Валидные пробы']
        for family in FAMILIES:
            data=time_data[(time_data.agents==n)&time_data.layout_id.isin([family+'_open',family+'_deadend'])]
            if not len(data):continue
            fig,axes=plt.subplots(4,3,figsize=(14,13))
            for ax,key,label in zip(axes.flat,series_metrics,series_labels):
                for suffix,color,title in [('open','#2d8194','Открытый'),('deadend','#c75050','Тупик')]:
                    g=data[data.layout_id==family+'_'+suffix].groupby('time_s')[key]
                    mean=g.mean();sem=g.std()/np.sqrt(g.count())
                    ax.plot(mean.index,mean.values,c=color,label=title,lw=1.4)
                    ax.fill_between(mean.index,mean-sem,mean+sem,color=color,alpha=.15)
                ax.set(title=label,xlabel='Время, с');ax.grid(alpha=.15)
            axes[0,0].legend();fig.suptitle(f'{FAMILIES[family][0]}, N={n}: среднее по seed и стандартная ошибка',fontsize=15)
            fig.tight_layout(rect=(0,0,1,.97));save(fig,charts/f'n{n}_timeseries_{family}')
        fig,ax=plt.subplots(figsize=(10,6))
        for suffix,color,label in [('open','#2d8194','Открытые проходы'),('deadend','#c75050','Тупики')]:
            data=agent_data[(agent_data.agents==n)&agent_data.layout_id.str.endswith(suffix)]
            arrival=data.arrival_time.to_numpy(float)
            xs=np.linspace(0,15,151);ys=np.array([np.mean(arrival<=x) for x in xs])
            ax.plot(xs,ys*100,c=color,label=label)
        ax.set(xlabel='Время, с',ylabel='Достигли цели, %',title=f'Достижение цели независимо от контактов, N={n}');ax.legend();ax.grid(alpha=.15)
        fig.tight_layout();save(fig,charts/f'n{n}_arrival_cdf')
        fig,axes=plt.subplots(2,3,figsize=(14,8))
        for ax,key in zip(axes.flat,['mean_sensitivity_xy','mean_command_change_rate','mean_command_second_difference',
                                    'mean_mean_change_rate_raw','mean_obstacle_share','mean_total_deviation']):
            for suffix,color,label in [('open','#2d8194','Открытые'),('deadend','#c75050','Тупики')]:
                values=agent_data[(agent_data.agents==n)&agent_data.layout_id.str.endswith(suffix)][key].dropna()
                ax.hist(values,bins=35,density=True,histtype='step',lw=1.5,color=color,label=label)
            ax.set(title=METRICS[key][0],xlabel=METRICS[key][1],ylabel='Плотность');ax.grid(alpha=.15)
        axes[0,0].legend();fig.suptitle(f'Распределение средних по агент-эпизодам, N={n}. Описательное сравнение.',fontsize=14)
        fig.tight_layout(rect=(0,0,1,.96));save(fig,charts/f'n{n}_diagnostic_distributions')
        corr_keys=['success','collision','final_distance','mean_sensitivity_xy','mean_command_change_rate','mean_obstacle_share']
        corr=agent_data[agent_data.agents==n][corr_keys].corr()
        fig,ax=plt.subplots(figsize=(9,7));im=ax.imshow(corr,vmin=-1,vmax=1,cmap='RdBu_r')
        labels=['Успех','Контакт','Расстояние','Чувствительность','Изменение команд','Доля препятствий']
        ax.set(xticks=range(len(labels)),yticks=range(len(labels)),xticklabels=labels,yticklabels=labels,title=f'Описательные корреляции по агент-эпизодам, N={n}')
        plt.setp(ax.get_xticklabels(),rotation=35,ha='right')
        for i in range(len(labels)):
            for j in range(len(labels)):ax.text(j,i,f'{corr.iloc[i,j]:.2f}',ha='center',va='center',color='white' if abs(corr.iloc[i,j])>.55 else '#243447')
        fig.colorbar(im,ax=ax);fig.tight_layout();save(fig,charts/f'n{n}_correlations')
    payload=dict(episodes=clean(episodes.to_dict('records')),summary=clean(wide.to_dict('records')),
                 statistics=clean(stats.to_dict('records')),paired=clean(contrast.to_dict('records')),
                 dictionary=clean(dictionary.to_dict('records')),catalogue=catalogue())
    (out/'workbook_data.json').write_text(json.dumps(payload,ensure_ascii=False,allow_nan=False))
    # An offline HTML index retains searchable tables and filterable chart galleries.
    table=wide[['agents','layout_id','episodes','success_rate','collision_rate','mean_sensitivity_xy','mean_command_change_rate','mean_obstacle_share']].copy()
    for c in ['success_rate','collision_rate','mean_obstacle_share']:table[c]=(table[c]*100).map(lambda x:f'{x:.1f}%')
    figures=[]
    for path in sorted(charts.glob('*.png')):
        figures.append(f'<details data-n="{100 if path.name.startswith("n100_") else 8}"><summary>{html.escape(path.stem)}</summary><img loading="lazy" src="data:image/png;base64,{base64.b64encode(path.read_bytes()).decode()}" alt="{html.escape(path.stem)}"></details>')
    page='''<!doctype html><html lang="ru"><meta charset="utf-8"><title>QuadSwarm: результаты</title><style>body{font:16px system-ui;margin:32px;color:#243447}table{border-collapse:collapse;font-size:13px}th,td{padding:8px;border-bottom:1px solid #ddd;text-align:right}th{background:#e9eef4}td:nth-child(2){text-align:left}img{max-width:100%;width:1000px}details{margin:12px 0;padding:8px;background:#f5f7fa}summary{cursor:pointer}button{margin:10px;padding:9px}</style><h1>QuadSwarm: контроль и тупики</h1><p>Один checkpoint на 1 000 005 120 шагов. Политика не дообучалась. Основная единица статистики - эпизод/seed.</p>'''
    page+=f'<p>Завершено эпизодов: {len(episodes)}. Агент-эпизодов: {int((episodes.agents).sum())}.</p>'
    page+='<p>obstacle_share - локальная чувствительность, не attention. Низкая чувствительность в тупике сама по себе не означает устойчивость.</p>'
    page+=table.to_html(index=False,escape=True)
    page+='<h2>Все графики</h2><button onclick="filterN(0)">Все</button><button onclick="filterN(8)">8 дронов</button><button onclick="filterN(100)">100 дронов</button>'
    page+=''.join(figures)+'<script>function filterN(n){document.querySelectorAll("details").forEach(x=>x.hidden=n&&+x.dataset.n!==n)}</script></html>'
    (out/'results.html').write_text(page,encoding='utf-8')
    build_pdf(out,episodes,wide,contrast,names)
    print('Analysis saved:',out,'episodes=',len(episodes),'charts=',len(list(charts.glob('*.png'))))


def build_pdf(out,episodes,wide,contrast,names):
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.pagesizes import A4,landscape
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak
    from matplotlib.font_manager import findfont,FontProperties
    pdfmetrics.registerFont(TTFont('DV',findfont(FontProperties(family='DejaVu Sans'))))
    pdfmetrics.registerFont(TTFont('DVB',findfont(FontProperties(family='DejaVu Sans',weight='bold'))))
    body=ParagraphStyle('body',fontName='DV',fontSize=10,leading=15,spaceAfter=9)
    head=ParagraphStyle('head',parent=body,fontName='DVB',fontSize=18,leading=23,spaceAfter=15)
    small=ParagraphStyle('small',parent=body,fontSize=8,leading=11)
    story=[Paragraph('QuadSwarm: результаты экспериментов',head),
           Paragraph(f'Завершено {len(episodes)} эпизодов на одной предобученной модели. '
                     f'Всего {int(episodes.agents.sum())} агент-эпизодов. Веса не обновлялись.',body),
           Paragraph('Использован checkpoint_000781254_1000005120.pth и config.json из ветки swarm100. '
                     'Комната 14 × 14 × 10 м, 2 наблюдаемых соседа, исходная физика и downwash, CPU + Numba. '
                     'Длительность эпизода 15 с. Сенсоры в этих экспериментах не отключались.',body),
           Paragraph('До: исходная случайная среда и восемь открытых контрольных проходов. '
                     'После: те же восемь проходов с дополнительными закрывающими стенами. '
                     'Старты и цели внутри каждой пары совпадают. Число уникальных seed в этом отчёте: '
                     + ', '.join(f'N={n}: {g.seed.nunique()}' for n,g in episodes.groupby('agents')) + '.',body),
           Paragraph('Строгий успех: удержание в радиусе 0,3 м от цели не менее 0,5 с и отсутствие контактов '
                     'с соседями, препятствиями и комнатой за весь эпизод. Исходная метрика проекта сохранена отдельно, '
                     'поскольку использует другое определение достижения и учёта столкновений. '
                     'В исходном случайном сценарии её порог цели равен 0,5 м, в фиксированных схемах 0,3 м. '
                     'Для основного сравнения применяется единая строгая метрика с порогом 0,3 м во всех средах.',body),
           Paragraph('Чувствительность и плавность рассчитаны формулами присланных drone_metrics.py и drone_influence.py. '
                     'Исправлено распознавание имени размера блока соседей. Насыщение и разности команд считаются на каждом шаге, '
                     'пробы чувствительности выполняются каждые 10 шагов, 8 проб, sigma=0,02 м, alpha=0,01.',body),
           Paragraph('Интервалы получены bootstrap по целым эпизодам, а не по зависимым дронам. '
                     'При одинаковом нулевом результате во всех seed bootstrap-интервал может выродиться в ноль. '
                     'Это не доказывает невозможность редкого успеха. Один обучающий seed не оценивает разброс между обучениями.',body),
           Paragraph('Туннели представлены стенами на всю высоту, как позволяет исходный плоский SDF. '
                     'Сложные схемы могут требовать больше 15 с, поэтому неудача включает ограничение по времени. '
                     'Корреляции диагностик описательные. obstacle_share не является attention или причинным вкладом. '
                     'Время вычислений измерено под параллельной нагрузкой и не является отдельным аппаратным бенчмарком.',body),PageBreak()]
    headline=[['N','Исходная среда','Открытые проходы','Тупики']]
    for n,g in wide.groupby('agents',sort=True):
        baseline=g[g.layout_id=='random_baseline'].success_rate.mean()
        controls=g[g.layout_id.str.endswith('_open')].success_rate.mean()
        traps=g[g.layout_id.str.endswith('_deadend')].success_rate.mean()
        headline.append([int(n),f'{baseline*100:.1f}%',f'{controls*100:.1f}%',f'{traps*100:.1f}%'])
    t=Table(headline,colWidths=[60,200,220,200])
    t.setStyle(TableStyle([('FONTNAME',(0,0),(-1,-1),'DV'),('FONTSIZE',(0,0),(-1,-1),12),
                          ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#243447')),('TEXTCOLOR',(0,0),(-1,0),colors.white),
                          ('TOPPADDING',(0,0),(-1,-1),10),('BOTTOMPADDING',(0,0),(-1,-1),10)]))
    story.extend([Paragraph('Главный результат',head),t,Spacer(1,18),
                  Paragraph('В таблице указан строгий успех. Для проходов показано равновзвешенное среднее по восьми '
                            'фиксированным семействам, включая сложные повороты. Открытая комната не включена в среднее проходов. '
                            'Плотность стартов отличается от исходной случайной среды, поэтому эффект закрывающей стены '
                            'нужно оценивать по парным сравнениям.',body),
                  Paragraph('Провал открытой и закрытой версии одной схемы означает, что политика испытывает сложности '
                            'уже с самим проходом. Нулевая разница для такой пары не означает, что тупики безопасны.',body),PageBreak()])
    for n in sorted(episodes.agents.unique(),reverse=True):
        data=wide[wide.agents==n].set_index('layout_id')
        story.append(Paragraph(f'Все схемы, N={n}',head))
        rows=[['Схема','Эпизоды','Успех, %','Контакт, %','Исходный\nуспех, %','Доля\nпрепятствий, %']]
        for entry in catalogue():
            if entry['id'] not in data.index:continue
            r=data.loc[entry['id']]
            rows.append([Paragraph(entry['title'],small),int(r.episodes),f'{r.success_rate*100:.1f}',f'{r.collision_rate*100:.1f}',
                         f'{r.official_agent_success*100:.1f}',f'{r.mean_obstacle_share*100:.1f}'])
        table=Table(rows,colWidths=[265,65,85,85,100,110],repeatRows=1)
        table.setStyle(TableStyle([('FONTNAME',(0,0),(-1,-1),'DV'),('FONTSIZE',(0,0),(-1,-1),9),
                                   ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#243447')),('TEXTCOLOR',(0,0),(-1,0),colors.white),
                                   ('VALIGN',(0,0),(-1,-1),'MIDDLE'),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f0f4f8')]),
                                   ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
        story.append(table);story.append(PageBreak())
        for metric in ['success_rate','collision_rate','mean_sensitivity_xy','mean_command_change_rate','mean_obstacle_share']:
            path=out/'charts'/f'n{n}_paired_{metric}.png'
            if path.exists():
                story.append(Paragraph(f'{METRICS[metric][0]}: разница после добавления тупика, N={n}',head))
                story.append(Image(str(path),width=690,height=345))
                story.append(Paragraph('Каждая точка - средняя разница между парными эпизодами. Интервал - 95% bootstrap по seed. '
                                       'Для долей изменение показано в процентных пунктах.',body));story.append(PageBreak())
    story.append(Paragraph('Динамика внутри эпизода',head))
    story.append(Paragraph('Далее для каждой пары показаны расстояние до цели, скорость, контакты, чувствительность к цели, '
                           'плавность и насыщение команд, а также локальная чувствительность к соседям и препятствиям. '
                           'Линии - средние по seed, полоса - одна стандартная ошибка. Это другая неопределённость, чем '
                           '95% bootstrap-интервал на диаграммах итогов.',body));story.append(PageBreak())
    for n in sorted(episodes.agents.unique(),reverse=True):
        for family in FAMILIES:
            path=out/'charts'/f'n{n}_timeseries_{family}.png'
            if path.exists():
                story.append(Paragraph(f'{FAMILIES[family][0]}, N={n}',head))
                story.append(Image(str(path),width=495,height=460));story.append(PageBreak())
    horizon=out/'horizon30/horizon_comparison.png'
    if horizon.exists():
        story.extend([Paragraph('Дополнительная проверка: 30 секунд',head),
                      Paragraph('16 эпизодов на 8 дронах: четыре семейства, открытая и закрытая версия, два seed. '
                                'Первые 15 секунд траекторий побайтово совпали с основной серией. '
                                'В проверенных тупиках увеличение горизонта не дало успешного достижения без контактов. '
                                'Некоторым открытым сложным схемам дополнительное время помогло. '
                                'Это диагностическая проверка, а не точная оценка редких событий.',body),
                      Image(str(horizon),width=700,height=300),PageBreak()])
    story.extend([Paragraph('Проверки и воспроизведение',head),
                  Paragraph('124 новых автоматических проверки прошли, включая настоящую модель и численные проверки диагностик. '
                            'CSV диагностик на контрольной линейной модели совпали с исходными скриптами побайтово. '
                            'Старые unittest из репозитория завершились с 9 ошибками интерфейсов и не объявляются прошедшими.',body),
                  Paragraph('Источник кода и конфигурации: github.com/Vikvins/quad-swarm-rl, ветка swarm100, '
                            'commit 85a0a2e1ff9993929b95d52646385a3b36fc2289. '
                            'Диагностики: ветка experiment-attention-katya, '
                            'commit 280800367d868d888424a1cab573f98a2d1e87d6.',body),
                  Paragraph('Команды воспроизведения находятся в SUITE_README_RU.md проекта. '
                            'Таблицы этого отчёта сохранены в tables/, графики в charts/. '
                            'Для аудита используются SHA256 весов и исходников из манифеста запуска. '
                            'Полный состав конкретной выгрузки указан в её README.',body)])
    size=landscape(A4)
    def footer(canvas,doc):
        canvas.setFont('DV',8);canvas.setFillColor(colors.HexColor('#66758a'))
        canvas.drawString(30,18,'QuadSwarm | 1B checkpoint | Парные эксперименты')
        canvas.drawRightString(size[0]-30,18,str(doc.page))
    doc=SimpleDocTemplate(str(out/'QuadSwarm_experiment_report.pdf'),pagesize=size,leftMargin=30,rightMargin=30,topMargin=25,bottomMargin=30)
    doc.build(story,onFirstPage=footer,onLaterPages=footer)


if __name__=='__main__':main()
