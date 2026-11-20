"""Build the exact-map PDF atlas, PNG/SVG images and machine-readable map catalogue."""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.font_manager import findfont,FontProperties
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4,landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Image,Table,TableStyle,PageBreak
from gym_art.quadrotor_multi.obstacles.layouts import LayoutSpec,build_layout,reference_route,validate_layout
from gym_art.quadrotor_multi.obstacles.suite_layouts import catalogue,FAMILIES


def draw(plan,ax,title,show_route=True):
    for x,y,_ in plan['obstacles']:
        ax.add_patch(Circle((x,y),plan['spec']['obstacle_size']/2,color='#626c76'))
    starts=np.asarray(plan['starts']);goals=np.asarray(plan['goals'])
    ax.scatter(*starts[:,:2].T,c='#147fbc',s=11,label='Старт')
    ax.scatter(*goals[:,:2].T,c='#218447',s=16,marker='x',label='Цель')
    if show_route and 'reference_route' in plan:
        route=np.asarray(plan['reference_route'])
        ax.plot(route[:,0],route[:,1],'--',c='#d8a41f',lw=1.5,label='Свободный путь')
    hx,hy=np.asarray(plan['spec']['room_dims'][:2])/2
    ax.set(xlim=(-hx,hx),ylim=(-hy,hy),aspect='equal',xlabel='x, м',ylabel='y, м',title=title)
    ax.grid(alpha=.12)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--baseline-json',type=Path)
    args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    data=[];rows=[]
    for entry in catalogue():
        if entry['id']=='random_baseline':
            if not args.baseline_json or not args.baseline_json.exists(): continue
            plan=json.loads(args.baseline_json.read_text());report={}
        else:
            obj=build_layout(LayoutSpec('suite',layout_id=entry['id'],num_agents=100,room_dims=(14.,14.,10.),seed=0))
            plan=obj.to_dict();report=validate_layout(obj)
            plan['reference_route']=reference_route(obj).tolist()
        plan['catalogue_entry']=entry
        (out/f"{entry['id']}.json").write_text(json.dumps(plan,indent=2,ensure_ascii=False))
        fig,ax=plt.subplots(figsize=(6.3,6.3));draw(plan,ax,entry['title']);ax.legend(loc='upper right',fontsize=7)
        fig.tight_layout();fig.savefig(out/f"{entry['id']}.png",dpi=160);fig.savefig(out/f"{entry['id']}.svg");plt.close(fig)
        data.append(plan)
        rows.append(dict(id=entry['id'],title=entry['title'],family=entry['family'],has_dead_end=entry['has_dead_end'],
                         paired_with=entry.get('paired_with'),width_m=entry.get('width'),num_obstacles=len(plan['obstacles']),
                         **{k:v for k,v in report.items() if k!='num_obstacles' and np.isfinite(v)}))
    (out/'all_layouts.json').write_text(json.dumps(data,indent=2,ensure_ascii=False))
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (out/'catalogue.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    fig,axes=plt.subplots(6,3,figsize=(14,25))
    for ax,plan in zip(axes.flat,data):draw(plan,ax,plan['catalogue_entry']['title'])
    fig.suptitle(f'{len(data)} схем: контрольные проходы и тупики',fontsize=18,y=.998)
    fig.tight_layout();fig.savefig(out/'all_layouts_overview.png',dpi=180);plt.close(fig)
    pdfmetrics.registerFont(TTFont('DV',findfont(FontProperties(family='DejaVu Sans'))))
    pdfmetrics.registerFont(TTFont('DVB',findfont(FontProperties(family='DejaVu Sans',weight='bold'))))
    body=ParagraphStyle('body',fontName='DV',fontSize=10,leading=15,textColor=colors.HexColor('#243447'))
    title=ParagraphStyle('title',parent=body,fontName='DVB',fontSize=24,leading=30,spaceAfter=20)
    head=ParagraphStyle('head',parent=body,fontName='DVB',fontSize=17,leading=23,spaceAfter=12)
    pagesize=landscape(A4)
    story=[Paragraph('QuadSwarm: атлас схем',title),Paragraph(f'{len(data)} карт для сравнения одной предобученной политики до и после добавления тупиков',head),
           Paragraph('Комната 14 × 14 × 10 м. Цилиндры диаметром 0,6 м. Примеры координат показаны для 100 дронов и seed=0. '
                     'Синие точки - старты, зелёные кресты - цели. Жёлтая линия подтверждает геометрическую достижимость '
                     'и не передаётся политике.',body),Spacer(1,15),
           Paragraph('Восемь пар отличаются только дополнительными закрывающими стенами. Старты, цели и seed внутри каждой пары совпадают. '
                     'Отдельно проверяются исходная случайная среда и открытая комната.',body),Spacer(1,15),
           Paragraph('Туннели здесь являются проходами в 2.5D: стены продолжаются на всю высоту комнаты. '
                     'Полноценная геометрия крыши и полёт над препятствиями не добавляются, поскольку опубликованная политика получает '
                     'девять значений плоского SDF. На виде сверху дроны на разных высотах могут выглядеть наложенными.',body),Spacer(1,15),
           Paragraph('Для каждой новой карты проверены безопасные стартовые позиции и связность свободного пространства. '
                     'Вместимость не достигается уменьшением заданного расстояния между дронами. '
                     'Это проверка геометрической достижимости, а не гарантия успешного пролёта всего роя за 15 секунд.',body),PageBreak()]
    byid={p['catalogue_entry']['id']:p for p in data}
    def pair_page(ids,heading,note):
        story.append(Paragraph(heading,head))
        images=[Image(str(out/f'{i}.png'),width=340,height=340) for i in ids if i in byid]
        story.append(Table([images],colWidths=[350]*len(images)))
        story.append(Spacer(1,10));story.append(Paragraph(note,body));story.append(PageBreak())
    pair_page(['random_baseline','open_room'],'Исходная среда и открытая комната',
              'Случайная среда использует исходный генератор препятствий и стартовых состояний из опубликованной конфигурации. '
              'Открытая комната является дополнительным контролем и не образует с ней парный эксперимент.')
    for family,(name,*_) in FAMILIES.items():
        pair_page([family+'_open',family+'_deadend'],name,
                  'Слева: открытая контрольная схема. Справа: добавлены закрывающие стены. '
                  'Для достижения цели в варианте с тупиком требуется отступление или обход через другой выход. '
                  'В обеих версиях совпадают координаты стартов и целей.')
    if isinstance(story[-1],PageBreak):story.pop()
    def footer(canvas,doc):
        canvas.setFont('DV',8);canvas.setFillColor(colors.HexColor('#66758a'))
        canvas.drawString(35,20,'QuadSwarm | Геометрия схем | seed=0')
        canvas.drawRightString(pagesize[0]-35,20,str(doc.page))
    doc=SimpleDocTemplate(str(out/'QuadSwarm_all_layouts.pdf'),pagesize=pagesize,rightMargin=35,leftMargin=35,topMargin=30,bottomMargin=35)
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print('Atlas saved:',out)


if __name__=='__main__': main()
