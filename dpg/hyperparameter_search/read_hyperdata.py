import cPickle



def loaddata(dataname):
    pkl_file = open(dataname, 'rb')
    paramlist = cPickle.load(pkl_file)

    scores, params = zip(*paramlist)
    print( 'scores', scores)
    print('max score', max(scores))


loaddata('hyper_inv_pendulum')