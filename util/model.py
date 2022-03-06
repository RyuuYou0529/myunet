from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *


def unet(pretrained_weights=None, input_size=(572, 572, 1)):
    # 572x527
    inputs = Input(input_size)

    # output: 570x570
    conv1 = Conv2D(64, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(inputs)
    # output: 568x568
    conv1 = Conv2D(64, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(conv1)

    # output: 284x284
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # output: 282x282
    conv2 = Conv2D(128, 3, activation='relu', padding="valid", kernel_initializer="he_normal")(pool1)
    # output: 280x280
    conv2 = Conv2D(128, 3, activation='relu', padding="valid", kernel_initializer="he_normal")(conv2)

    # output: 140x140
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # output: 138x138
    conv3 = Conv2D(256, 3, activation='relu', padding="valid", kernel_initializer="he_normal")(pool2)
    # output: 136x136
    conv3 = Conv2D(256, 3, activation='relu', padding="valid", kernel_initializer="he_normal")(conv3)

    # output: 68x68
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # output: 66x66
    conv4 = Conv2D(512, 3, activation='relu', padding="valid", kernel_initializer="he_normal")(pool3)
    # output: 64x64
    conv4 = Conv2D(512, 3, activation='relu', padding="valid", kernel_initializer="he_normal")(conv4)

    # output: 32x32
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # output: 30x30
    conv5 = Conv2D(1024, 3, activation='relu', padding="valid", kernel_initializer="he_normal")(pool4)
    # output: 28x28
    conv5 = Conv2D(1024, 3, activation='relu', padding="valid", kernel_initializer="he_normal")(conv5)

    # output: 56x56
    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv5))

    # output: 56x56
    crop4 = Cropping2D((4, 4))(conv4)
    merge6 = concatenate([crop4, up6], axis=3)
    # output: 54x54
    conv6 = Conv2D(512, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(merge6)
    # output: 52x52
    conv6 = Conv2D(512, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(conv6)

    # output: 104x104
    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))

    # output: 104x104
    crop3 = Cropping2D((16, 16))(conv3)
    merge7 = concatenate([crop3, up7], axis=3)
    # output: 102x102
    conv7 = Conv2D(256, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(merge7)
    # output: 100x100
    conv7 = Conv2D(256, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(conv7)

    # output: 200x200
    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))

    # output: 200x200
    crop2 = Cropping2D((40, 40))(conv2)
    merge8 = concatenate([crop2, up8], axis=3)
    # output: 198x198
    conv8 = Conv2D(128, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(merge8)
    # output: 196x196
    conv8 = Conv2D(128, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(conv8)

    # output: 392x392
    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))

    # output: 392x392
    crop1 = Cropping2D((88, 88))(conv1)
    merge9 = concatenate([crop1, up9], axis=3)
    # output: 390x390
    conv9 = Conv2D(64, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(merge9)
    # output: 388x388
    conv9 = Conv2D(64, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(conv9)
    # output: 388x388
    conv9 = Conv2D(2, 1, activation='relu', padding='valid', kernel_initializer='he_normal')(conv9)

    # output: 388x388
    conv10 = Conv2D(1, 1, activation='softmax')(conv9)

    model = Model(inputs, conv10)
    model.compile(optimizer=SGD(learning_rate=1e-4, momentum=0.99), loss='binary_crossentropy', metrics=['accuracy'])

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model
