'''
Parameters used for linearization using approach #1.
'''  

inputs = [{'network'    : 'MV_EUR', # VOLTAGE INPUT
             'ElmFile'    : 'PCCVolt',
             'source'     : 'generate',
             'wave_specs' : {'type':'dip', 'sigma':0.01, 'mu':1, 'tstart':0, 'tstop':10,'step':0.001,
                             'y0':1, 't0' : 0.03, 'deltat' : 6, 'deltay':-0.01} 
            },
            {'network'    : 'MV_EUR', # FREQUENCY INPUT
             'ElmFile'    : 'PCCFreq',
             'source'     : 'generate',
             'wave_specs' : {'type':'step', 'sigma':0.001, 'mu':1, 'tstart':0, 'tstop':10,'step':0.001,
                             'y0':1, 't0' : 5, 'deltat' : 0.05, 'deltay':0.001} 
            }]

outputs = {
    'ElmRes' : 'ModalSim',
    'variables': {
        'SourcePCC.ElmVac': ['m:Psum:bus1','m:Qsum:bus1']},
    'outputs_idx' : {
        'SourcePCC.ElmVac\\m:Qsum:bus1':0,
        'SourcePCC.ElmVac\\m:Psum:bus1':1}
}

ref_angle = 'MV_EUR\\SourcePCC.ElmVac\\s:phiu'