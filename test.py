from shufflenet import ShuffleNet
from keras.utils import plot_model

model = ShuffleNet()
plot_model(model, to_file='/var/www/model_groups_%d.svg' % 1, show_shapes=True, show_layer_names=True)