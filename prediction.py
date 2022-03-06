from keras.models import *
from util.data import *

model = load_model('./model/v2.hdf5')

testGene = testGenerator('data/membrane/test/image')
results = model.predict_generator(testGene, 30, verbose=1)
saveResult("data/membrane/test/prediction", results)
