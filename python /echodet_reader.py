#!/bin/env python

import numpy as np
import csv


try: # see if we have Python 2.6+
    from collections import OrderedDict
except ImportError:
    OrderedDict = dict

def read_wbfile(filename):
    """
    Read b/w files produced by echodet
    
    Returns list of (metadata, data) pairs
    """
    result = []             # result
    data   = []             # data
    meta   = OrderedDict()  # meta data
    isdata = False
    iline  = 0
 
    with open(filename, 'rt') as fd:
        for line in fd.readlines():
            line = line.strip()
            if not line: continue
            iline += 1
            if iline <= 2:  # first two lines are info
                meta['info%d' % iline] = line
                continue
            if line  == 'values':  # values indicate start of data block
                isdata = True
                continue
            if line == '#eod' or line == '#nxt':
                result.append((meta,np.asarray(data).T))
                data = []
                meta = OrderedDict()
                isdata = False
                iline  = 0
                if line == '#eod': break
                continue
      
            xl = line.split()
            if isdata:
                try:
                    data.append([float(x) for x in xl])
                except ValueError:
                    pass
            else:
                key   = xl[0]
                value = " ".join(xl[1:])
                try:
                    meta[key] = eval(value)
                except (ValueError, SyntaxError, NameError):
                    meta[key] = value
    return result

def write_csv(filename, data, **kwargs):
    header = ["%s" % kwargs.pop('label'),]
    for key in kwargs:
        header.append("%s, %s" % (key, kwargs.get(key)))
    header.append(" ")
    header.append("%13.13s%13.13s%13.13s" % ("tau","S(Q,t)","dS(Q,t)"))
    np.savetxt(filename, data.T, fmt='%13.6f',header="\n".join(header), delimiter=',')





def test():
    import sys
    import matplotlib.pyplot as plt
  
    for filename in sys.argv[1:]:
        for m, d in read_wbfile(filename)[1:]:
            plt.errorbar(d[0],d[1], yerr=d[2], fmt='.-', label='q=%5.3f' % m['q'])
            
    ax = plt.gca()
    ax.set_xscale('log')
    ax.set_ylim(0.0,1.2)
    #plt.title(filename)
    plt.xlabel(r'$\tau$ [ns]')
    plt.ylabel(r'$S(q,\tau)/S(q,0)$')
    plt.legend(loc='best')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    test()
