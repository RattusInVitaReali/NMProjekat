import keras
import visualkeras
from PIL import ImageFont
from keras.utils import plot_model

font = ImageFont.truetype("arial.ttf", 32)
model = keras.models.load_model('./models')
visualkeras.layered_view(model, legend=True, font=font, to_file="output.png")
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
