# local loading without fastai doesn't work

import pickle

pickle_file_name = './export.pkl'

pickle_file = open(pickle_file_name, 'rb')

learn = pickle.load(pickle_file)

print(learner)

# _pickle.UnpicklingError: A load persistent id instruction was encountered,
# but no persistent_load function was specified.