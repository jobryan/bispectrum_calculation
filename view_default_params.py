# view_default_params.py
import pickle
s_fn = 'data/params.pkl'
d_params = pickle.load(open(s_fn, 'rb'))
for k in d_params.keys():
    print '%s: %s' % (k, d_params[k])