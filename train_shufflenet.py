import keras
from keras.preprocessing.image import ImageDataGenerator
from shufflenet import ShuffleNet
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau

if __name__ == '__main__':
    groups = 3
    batch_size = 128
    ds = '/mnt/daten/Development/ILSVRC2012_256'


    model = ShuffleNet(groups=groups, pooling='max')
    # model.load_weights("shufflenet_g3.hdf5")
    csv_logger = CSVLogger('training_g%d.log' % groups, append=True)
    checkpoint = ModelCheckpoint(filepath='shufflenet_g%d.hdf5' % groups, verbose=1, save_best_only=True,
                                   monitor='val_acc', mode='max')

    train_datagen = ImageDataGenerator(rescale=1./255,
                                       featurewise_center=True,
                                       shear_range=0.05,
                                       zoom_range=0.05,
                                       horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255,
                                      featurewise_center=True)

    train_datagen.mean = [0.406, 0.456, 0.485]
    test_datagen.mean = train_datagen.mean

    train_generator = train_datagen.flow_from_directory(
            '%s/train/' % ds,
            target_size=(224, 224),
            batch_size=batch_size)

    test_generator = test_datagen.flow_from_directory(
            '%s/val/' % ds,
            target_size=(224, 224),
            batch_size=batch_size)

    model.compile(
              optimizer=keras.optimizers.Adam(decay=1e-5),
              metrics=['accuracy'],
              loss='categorical_crossentropy')

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=4, cooldown=0)

    model.fit_generator(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            epochs=50,
            workers=6,
            use_multiprocessing=True,
            validation_data=test_generator,
            validation_steps=test_generator.samples // batch_size,
            callbacks=[csv_logger, checkpoint, reduce_lr])
