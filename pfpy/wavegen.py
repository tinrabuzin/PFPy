import numpy as np

def dip(**wp):
    """
    Generates a step voltage dip
    """
    t = np.arange(wp['tstart'], wp['tstop'], wp['step'])
    y = wp['y0']*np.ones(t.shape)
    y[np.logical_and(t>=wp['t0'], t<=(wp['t0']+wp['deltat']))] = wp['y0']-wp['deltay']
    return {'time':t, 'y1' : y}

def step(**wp):
    """Generates a step signal"""
    t = np.arange(wp['tstart'], wp['tstop'], wp['step'])
    y = wp['y0']*np.ones(t.shape)
    y[t>=wp['t0']] = wp['y0']-wp['deltay']
    return {'time':t, 'y1' : y}

def const(**wp):
    """Generates a constant signal"""
    return {'time': np.arange(wp['tstart'], wp['tstop'], wp['step']),
            'y1' : wp['y0']*np.ones(int((wp['tstop']-wp['tstart'])/wp['step'])) 
        }

def impulse(**wp):
    """Generates an impulse signal"""
    wp['deltay'] = -1
    wp['deltat'] = wp['step']
    return dip(**wp)
    
wavetype = {
    'dip'     : dip,
    'const'   : const,
    'step'    : step,
    'impulse' : impulse
}