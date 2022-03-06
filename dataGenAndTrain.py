from util.model import *
from util.data import *
from keras.callbacks import ModelCheckpoint

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 数据增强时用到的参数
data_gen_args = dict(rotation_range=0.2,
                     width_shift_range=0.05,
                     height_shift_range=0.05,
                     shear_range=0.05,
                     zoom_range=0.05,
                     horizontal_flip=True,
                     fill_mode='nearest')

myGene = trainGenerator(2, 'data/membrane/train', 'image', 'label', data_gen_args,
                        save_to_dir='data/membrane/train/aug')

model = unet()

# 保存模型，在每个epoch后，保存模型到filepath
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss', verbose=1, save_best_only=True)

# 训练
model.fit_generator(myGene, steps_per_epoch=300, epochs=1, callbacks=[model_checkpoint])

model.save('./model/v1')


