"""
MNIST opdracht B: "Conv Dense"      (by Marius Versteegen, 2021)

Bij deze opdracht gebruik je alleen nog als laatste layer een dense layer.

De opdracht bestaat uit drie delen: B1 tm B3.

Er is ook een impliciete opdracht die hier niet wordt getoetst
(maar mogelijk wel op het tentamen):
Zorg ervoor dat je de onderstaande code volledig begrijpt.

Tip: stap in de Debugger door de code, en bestudeer de tussenresultaten.
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from MavTools_NN import ViewTools_NN

#tf.random.set_seed(0) #for reproducability

# Model / data parameters
num_classes = 10

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

#print(x_test[0])
# print("show image\n")
# plt.figure()
# plt.imshow(x_test[0])
# plt.colorbar()
# plt.show()
# plt.grid(False)

inputShape = x_test[0].shape

# show the shape of the training set and the amount of samples
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# Conv layers expect images.
# Make sure the images have shape (28, 28, 1). 
# (for RGB images, the last parameter would have been 3)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# This time, we don't flatten the images, because that would destroy the 
# locality benefit of the convolution layers.

# convert class vectors to binary class matrices (one-hot encoding)
# for example 3 becomes (0,0,0,1,0,0,0,0,0,0)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

"""
Opdracht B1: 
    
Voeg als hidden layers ALLEEN Convolution en/of Dropout layers toe aan het onderstaande model.
Probeer een zo eenvoudig mogelijk model te vinden (dus met zo min mogelijk parameters)
dat een test accurracy oplevert van tenminste 0,97.

Voorbeelden van layers:
    layers.Dropout(getal)
    layers.Conv2D(getal, kernel_size=(getal, getal), padding="valid" of "same")
    layers.Flatten()
    layers.MaxPooling2D(pool_size=(getal, getal))

Beschrijf daarbij met welke stappen je bent gekomen tot je model,
je ervaringen daarbij en probeer die ervaringen te verklaren.

Ik ben begonnen met debuggen waarom tkinter niet werkte voor matplotlib, uiteindelijk duurde me het te lang
dus ben ik overgestapt van python3.10 naar 3.6 en toen werkte het wel. Helaas nog geen gpu mogelijkheden dus alles 
duurde erg lang. Na een half uur geprobeerd te hebben met 2 convolutional layers met verschillende hoeveelheden
filters en kernel groottes heb ik het opgegeven en nog een uur gespendeerd aan CUDA installeren op WSL. Nadat dit 
nogsteeds niet werkte heb ik de spoiler gekopieerd en ben ik hier mee verder gegaan. De configuratie in de spoiler is 
een goede keuze omdat de 2 convolutionele layers (dus inclusief de maxpooling anders is het technisch gezien geen 
convolutionele laag) de afbeelding pakken en hier 20 verschillende features in zoeken, de tweede convolutionele laag 
doet dat op een kleinere afbeelding dan de eerste waardoor ze dus allebij verschillende features vinden. Deze features worden
vervolgens plat geslagen en aan een dense layer gevoed (met een dropout om overfitting tegen te gaan) die deze features 
vervolgens classificeerd als cijfers.

Spoiler-mogelijkheid:
Mocht het je te veel tijd kosten (laten we zeggen meer dan een uur), dan
mag je de configuratie uit Spoiler_B.py gebruiken/kopieren.

Probeer in dat geval te beargumenteren waarom die configuratie een goede keuze is.
"""

def buildMyModel(inputShape):
    model = keras.Sequential(
        [
            keras.Input(shape=inputShape),
            
            layers.Conv2D(20, kernel_size=(3, 3), padding="same"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(20, kernel_size=(3, 3), padding="same"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(.1),
            layers.Dense(units=num_classes, activation='sigmoid')
        ]
    )
    return model

model = buildMyModel(x_train[0].shape)
model.summary()

"""
Opdracht B2 & B3:
conv2d (Conv2D)                 (None, 28, 28, 20) 
    (28x28 pixels met 20 filters) ((3 * 3 + 1) * 20 = 200)
max_pooling2d (MaxPooling2D)    (None, 14, 14, 20) 
    (14x14 door de maxpooling met kernel 2, deelt het aantal pixels door 2) (zelfde aantal params)
conv2d_1 (Conv2D)               (None, 14, 14, 20) 
    (De output shape van maxpooling veranderd niet bij ingang van conv2d, wel toevallig weer zelfde aantal filters) ((3 * 3 * 20 + 1) * 20 = 3620)
max_pooling2d_1 (MaxPooling2D)  (None, 7, 7, 20)   
    (Max pooling met kernel 2 halveert weer het aantal pixels) (zelfde aantal params)
flatten (Flatten)               (None, 980)         
    (7 * 7 * 20 = 980) (zelfde aantal params)
dropout (Dropout)               (None, 980)        
    (Dropout doet niks met de shape) (zelfde aantal params)
dense (Dense)                   (None, 10)         
    (Dense layer heeft 10 neuronen met allemaal 980 inputs) (10 * (980 + 1) = 9810)

Kopieer bovenstaande model summary en verklaar bij 
bovenstaande model summary bij elke laag met kleine berekeningen 
de Output Shape

"""

"""
Opdracht B3: 
    
Verklaar nu bij elke laag met kleine berekeningen het aantal parameters. Zie hierboven
"""

"""
# Train the model
"""
batch_size = 1024   # Larger often means faster training, but requires more system memory.
                  # if you get allocation accord
epochs = 250      # it's probably more then you like to wait for,
                  # but you can interrupt training anytime with CTRL+C

learningrate = 0.01
# loss_fn = "categorical_crossentropy" # can only be used, and is effictive for an output array of hot-ones (one dimensional array)
loss_fn = 'mean_squared_error'     # can be used for other output shapes as well. seems to work better for categorical as well..

optimizer = keras.optimizers.Adam(learning_rate=learningrate)
model.compile(loss=loss_fn,
              optimizer=optimizer,
              metrics=['categorical_accuracy'])

print("\nx_train_flat.shape:", x_train.shape)
print("y_train.shape", y_train.shape)

bInitialiseWeightsFromFile = True # Set this to false if you've changed your model.
learningrate = 0.0001 if bInitialiseWeightsFromFile else 0.01
if (bInitialiseWeightsFromFile):
    model.load_weights("Spoiler_B_weights.h5"); # let's continue from where we ended the previous time. That should be possible if we did not change the model.
                                        # if you use the model from the spoiler, you
                                        # can avoid training-time by using "Spoiler_B_weights.h5" here.

print ("\n");
print (ViewTools_NN.getColoredText(255,255,0,"Just type CTRL+C anytime if you feel that you've waited for enough episodes."))
print ("\n");
# # NB: validation split van 0.2 ipv 0.1 gaf ook een boost: van 99.49 naar 99.66
try:
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)
except KeyboardInterrupt:
    print("interrupted fit by keyboard")

"""
## Evaluate the trained model
"""

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])    
print("Test accuracy:", ViewTools_NN.getColoredText(255,255,0,score[1]))

model.summary()
model.save_weights('myWeights.h5')

prediction = model.predict(x_test)
print("\nFirst test sample: predicted output and desired output:")
print(prediction[0])
print(y_test[0])

# study the meaning of the filtered outputs by comparing them for
# a few samples
nLastLayer = len(model.layers)-1
nLayer=nLastLayer                 # this time, I select the last layer, such that the end-outputs are visualised.
print("lastLayer:",nLastLayer)

baseFilenameForSave=None
x_test_flat = None
# ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 0, baseFilenameForSave)
# ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 1, baseFilenameForSave)
# ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 2, baseFilenameForSave)
# ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 3, baseFilenameForSave)
# ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 4, baseFilenameForSave)
# ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 6, baseFilenameForSave)
# ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 5, baseFilenameForSave)
# ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 7, baseFilenameForSave)
