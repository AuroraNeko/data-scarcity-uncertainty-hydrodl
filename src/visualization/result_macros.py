"""Result-derived prose values for the retained supplementary controls.

Each binding records its JSON pointer, aggregation and display precision. The
same JSON files drive supplement_tables; missing fields deliberately fail.
"""
import json


def control_macros(tables):
    values={};bindings={}
    def bind(name,file,pointer,kind='number',digits=3,scale=1):
        value=json.loads((tables/file).read_text(encoding='utf-8'))
        for key in pointer.split('/'):value=value[key]
        if kind=='ci':text='['+', '.join(f'{v*scale:.{digits}f}' for v in value)+']'
        elif kind=='integer':text=f'{value:,}'
        else:text=f'{value*scale:.{digits}f}'+(r'\%' if kind=='percent' else '')
        values[name]=text
        bindings[name]={'file':file,'json_pointer':'/'+pointer,'format':kind,'digits':digits,'scale':scale}
    equal='scarcity_equal_updates/equal_update_summary.json'
    arch='scarcity_standard_lstm/architecture_sensitivity_summary.json'
    for prefix,file,base in [('EqualUpdate',equal,'training_budgets/fixed_updates'),
                             ('StandardLSTM',arch,'architectures/standard_lstm')]:
        for suffix,path in [('OneYearNSE','by_year/1/median_nse/mean'),('FifteenYearNSE','by_year/15/median_nse/mean'),
            ('NSEDelta','effects_15_minus_1/nse/estimate'),('FifteenYearRawPICP','by_year/15/raw_picp/mean'),
            ('FifteenYearRawWidth','by_year/15/raw_mpiw_log/mean'),('FifteenYearRawWinkler','by_year/15/raw_winkler_log/mean')]:
            bind(prefix+suffix,file,base+'/'+path)
        bind(prefix+'NSEDeltaCI',file,base+'/effects_15_minus_1/nse/ci95','ci')
    bind('StandardLSTMParameters',arch,'architectures/standard_lstm/n_parameters','integer')
    bind('FixedEpochFifteenYearSteps',equal,'training_budgets/fixed_epochs/optimizer_budget/15/optimizer_steps','integer')
    bind('EqualUpdateSteps',equal,'training_budgets/fixed_updates/optimizer_budget/15/optimizer_steps','integer')
    rep='scarcity_replications/scarcity_replication_summary.json';base='combined_effects_15_minus_1/'
    for metric,prefix,scale,digits in [('nse','ReplicatedNSE',1,3),('raw_picp','ReplicatedRawPICP',100,1)]:
        bind(prefix+('Delta' if metric=='nse' else 'EndpointPP'),rep,base+metric+'/estimate',digits=digits,scale=scale)
        bind(prefix+('DeltaCI' if metric=='nse' else 'EndpointPPCI'),rep,base+metric+'/ci95','ci',digits,scale)
        bind(prefix+'PositiveBasins',rep,base+metric+'/fraction_basins_positive','percent',0,100)
    return values,bindings
