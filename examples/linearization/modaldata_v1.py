'''
Parameters used for linearization using approach #1.
'''  


inputs_bd = [{'network'    : 'MV_EUR', # VOLTAGE INPUT
             'ElmFile'    : 'PCCVolt',
             'source'     : 'generate',
             'wave_specs' : {'type' : 'step', 'tstart' : 0, 'tstop' : 10,
                             'step' : 0.01, 't0' : 1,  'y0' : 1, 'deltay' : -0.05}
            },
            {'network'    : 'MV_EUR', # FREQUENCY INPUT
             'ElmFile'    : 'PCCFreq',
             'source'     : 'generate',
             'wave_specs' : {'type' : 'step', 'tstart' : 0, 'tstop' : 10,
                             'step' : 0.01, 't0' : 1, 'y0' : 1, 'deltay' : -0.05}
            }]

inputs_c = [{'network'    : 'MV_EUR', # VOLTAGE INPUT
             'ElmFile'    : 'PCCVolt',
             'source'     : 'generate',
             'wave_specs' : {'type' : 'dip', 'tstart' : 0, 'tstop' : 10,
                             'step' : 0.001, 't0' : 0.5,  'y0' : 1, 
                             'deltat' : 0.001, 'deltay' : 1}
            },
            {'network'    : 'MV_EUR', # FREQUENCY INPUT
             'ElmFile'    : 'PCCFreq',
             'source'     : 'generate',
             'wave_specs' : {'type' : 'dip', 'tstart' : 0, 'tstop' : 10,
                             'step' : 0.001, 't0' : 0.002, 'y0' : 1,
                             'deltat' : 0.001, 'deltay' : 0.05}
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

