"""Primary target calendar and the three distinct experiment roles."""
from datetime import datetime, timedelta
from pathlib import Path
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch, Patch


def draw_protocol(summary, methods, output_dir):
    protocol = summary['protocol']
    blue, teal, orange = '#176A96', '#087D75', '#B5682F'
    ink, gray = '#183746', '#526874'
    fig = plt.figure(figsize=(9.4, 5.9), facecolor='white')
    ax = fig.add_axes([.18, .37, .79, .49])
    date = lambda s: mdates.date2num(datetime.fromisoformat(s))

    def period(y, start, end, color):
        left = date(start)
        ax.barh(y, date(end)+1-left, left=left, height=.57,
                color=color, edgecolor='white', linewidth=.6, zorder=3)
        # Context lies on a separate row for each evaluation period. Thus it
        # cannot paint over, or appear to remove, calibration target days.
        ax.barh(y, 180, left=left-180, height=.57, facecolor='#E7ECEF',
                edgecolor=gray, linewidth=.6, hatch='////', zorder=3)

    for y, year in zip([5,4,3,2], [15,5,3,1]):
        start,end = protocol['training_target_periods'][str(year)]
        period(y,start,end,blue)
    cs,ce = protocol['calibration_target_period']; ts,te = protocol['test_target_period']
    period(1,cs,ce,teal); period(0,ts,te,orange)
    ax.text((date(cs)+date(ce))/2,1,'5 years',color='white',
            ha='center',va='center',fontsize=10,weight='bold')
    ax.text(date(ce)+110,1,'Oct 1995 - Sep 2000',color=teal,
            ha='left',va='center',fontsize=9.5)
    ax.text((date(ts)+date(te))/2,0,'2000-10 to 2008-12',color='white',
            ha='center',va='center',fontsize=9.5,weight='bold')
    ax.axvline(date(cs),color=gray,ls='--',lw=1)
    ax.text(date(cs)-70,5.48,'Training ends 30 Sep 1995',fontsize=9.5,
            ha='right',va='bottom',color=gray)
    ax.set_yticks([5,4,3,2,1,0],['15-year training','5-year training','3-year training',
                                 '1-year training','Calibration targets','Test targets'])
    ax.set_xlim(date('1979-12-01'),date('2009-03-01'));ax.set_ylim(-.55,5.85)
    ax.xaxis.set_major_locator(mdates.YearLocator(5));ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(axis='y',length=0,labelsize=10);ax.tick_params(axis='x',labelsize=10)
    ax.spines[['top','right','left']].set_visible(False)
    ax.spines['bottom'].set_color('#B4C2C9');ax.grid(axis='x',color='#DFE7EB',lw=.65,zorder=0)
    ax.axhline(1.5,color='#DFE7EB',lw=.8)
    fig.text(.035,.96,'a   Target periods and antecedent context',fontsize=12.5,weight='bold',color=ink)
    fig.legend(handles=[Patch(fc=blue,label='Training targets'),Patch(fc=teal,label='Calibration'),
                        Patch(fc=orange,label='Test'),Patch(fc='#E7ECEF',ec=gray,hatch='////',label='180-day forcing context')],
               loc='upper left',bbox_to_anchor=(.17,.94),ncol=4,frameon=False,fontsize=9.5,
               columnspacing=1.4,handlelength=1.6)
    fig.text(.18,.305,r'Forecast day $t$: forcings through $t-1$; observed flow is never an input.',fontsize=10,color=gray)
    fig.text(.035,.255,'b   Experimental roles',fontsize=12.5,weight='bold',color=ink)
    expanded=json.loads((Path(__file__).resolve().parents[2]/'results/tables/method_comparison.json').read_text())
    cards=[(.035,blue,'Primary length gradient',
            f"{summary['n_basins']} basins | 4 lengths | 3 seeds\n12 fits; 180-day input\nSame 30 epochs; updates differ"),
           (.36,teal,'Disjoint endpoint replication',
            '4 additional subsets of 50 basins\n1 / 15 years; 3 seeds; 24 fits\n250 unique basins with primary'),
           (.685,orange,'Full-basin method benchmark',
            f"{methods['protocol']['n_basins']} basins; {len(expanded['methods'])} constructions\n365-day input; same test period\nIncludes all 250 sampled basins")]
    for x,color,title,body in cards:
        fig.add_artist(FancyBboxPatch((x,.035),.29,.18,transform=fig.transFigure,
                       boxstyle='round,pad=.008,rounding_size=.012',fc='#F3F7F9',ec='#D8E2E7',lw=.8))
        fig.text(x+.012,.183,title,fontsize=10.1,weight='bold',color=color)
        fig.text(x+.012,.15,body,fontsize=9.5,color=ink,va='top',linespacing=1.6)
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    for ext in ['pdf','svg','png']:
        fig.savefig(out/f'fig_protocol_timeline.{ext}',dpi=300,bbox_inches='tight',pad_inches=.08)
    plt.close(fig)
