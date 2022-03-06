from util.model import *
from util.data import *
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 数据增强时用到的参数
data_gen_args = dict(rotation_range=0.2,
                     width_shift_range=0.05,
                     height_shift_range=0.05,
                     shear_range=0.05,
                     zoom_range=0.05,
                     horizontal_flip=True,
                     fill_mode='nearest')

myGene = trainGenerator(1, 'data/membrane/train', 'image', 'label', data_gen_args,
                        save_to_dir='data/membrane/train/aug')

model = unet()

model.summary()

# 保存模型，在每个epoch后，保存模型到filepath
model_checkpoint = ModelCheckpoint('models/v2.hdf5', monitor='loss', verbose=1, save_best_only=True)

TensorBoard(log_dir='logs', update_freq='batch')

# 训练
model.fit(myGene, steps_per_epoch=300, epochs=1, callbacks=[model_checkpoint])
