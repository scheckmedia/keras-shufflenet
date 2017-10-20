import keras
from keras.preprocessing.image import ImageDataGenerator
from shufflenet import ShuffleNet
from keras.callbacks import CSVLogger, ModelCheckpoint


if __name__ == '__main__':
    groups = 3
    batch_size = 128
    ds = '/mnt/daten/Development/ILSVRC2012_256'

    model = ShuffleNet(groups=groups, input_shape=(224, 224, 3))

    csv_logger = CSVLogger('training_g%d.log' % groups)
    checkpoint = ModelCheckpoint(filepath='shufflenet_g%d.hdf5' % groups, verbose=1, save_best_only=True,
                                   monitor='val_acc', mode='max')

    train_datagen = ImageDataGenerator(rescale=1./255.0, samplewise_center=True,
                                       samplewise_std_normalization=True,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255.0,
                                      samplewise_center=True,
                                      samplewise_std_normalization=True)

    train_generator = train_datagen.flow_from_directory(
            '%s/train/' % ds,
            target_size=(224, 224),
            batch_size=batch_size)

    test_generator = test_datagen.flow_from_directory(
            '%s/val/' % ds,
            target_size=(224, 224),
            batch_size=batch_size)

    model.compile(
              optimizer=keras.optimizers.Adam(decay=4e-5),
              metrics=['accuracy'],
              loss='categorical_crossentropy')

    model.fit_generator(
            train_generator,
            steps_per_epoch=10000,
            epochs=10000,
            workers=6,
            use_multiprocessing=True,
            validation_data=test_generator,
            validation_steps=390,
            callbacks=[csv_logger, checkpoint])
