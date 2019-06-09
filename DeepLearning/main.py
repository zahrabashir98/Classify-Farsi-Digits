from HodaDatasetReader import read_hoda_dataset, read_hoda_cdb
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import matplotlib.pyplot as plt

batch_size = 128
epochs = 5

print('Reading train dataset (Train 60000.cdb)...')
X_train, Y_train = read_hoda_dataset(dataset_path='./DigitDB/Train 60000.cdb',
                                images_height=32,
                                images_width=32,
                                one_hot=False,
                                reshape=False)
                                
print('Reading test dataset (Test 20000.cdb)...')
X_test, Y_test = read_hoda_dataset(dataset_path='./DigitDB/Test 20000.cdb',
                              images_height=32,
                              images_width=32,
                              one_hot=False,
                              reshape=False)


from keras.utils import to_categorical
#one-hot encode target column
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)


print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten


model = Sequential()
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(32,32,1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# checkpointer = keras.callbacks.ModelCheckpoint(
#     'models/model.h5',
#     save_best_only = True,
#     monitor = 'val_acc',
#     mode = 'auto',
#     verbose = 1
# )

charting = True
#train the model
history_callback = model.fit(X_train, Y_train, batch_size=batch_size, validation_data=(X_test, Y_test),verbose=1, epochs=epochs)
if charting == True:
    plt.plot(history_callback.history['loss'])
    plt.plot(history_callback.history['val_loss'])
    plt.title('model error')
    plt.ylabel('error')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(history_callback.history['acc'])
    plt.plot(history_callback.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

test_score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', test_score[0])
print('Test accuracy:', test_score[1])
