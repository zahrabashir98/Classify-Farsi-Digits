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

# prediction
y_pre = model.predict(X_test)
INDEX = []
for data in y_pre:
    counter = 0
    for each in data:
        if each == 1:
            index = counter
            break
        counter += 1
    INDEX.append(index)

tn = 0
tp = 0
fp = 0
fn = 0

classes_info = []
for i in range(10):
    classes_info.append({'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0})

for i in range(20000):

    if Y_test[i][INDEX[i]] == 1:
        tp += 1
        tn += 1
        classes_info[INDEX[i]]['tp'] += 1
        classes_info[INDEX[i]]['tn'] -= 1
    else:
        fp += 1
        fn += 1
        true_class_index = -1
        for data in Y_test[i]:
            true_class_index += 1
            if data == 1:
                break
            
        classes_info[true_class_index]['fp'] += 1
        classes_info[INDEX[i]]['fn'] += 1
        classes_info[true_class_index]['tn'] -= 1
        classes_info[INDEX[i]]['tn'] -= 1
    
    for j in range(10):
            classes_info[j]['tn'] += 1

print("*******************\n")

for i in range(10):
    print("Class %s : %s\n"%(i,classes_info[i]))

print("*******************\n\n")

def calculate_pre_rec_f1(class_info):
    try:
        precision = class_info['tp'] / (class_info['tp'] + class_info['fp'])
        recall = class_info['tp'] / (class_info['tp'] + class_info['fn'])
        f1_score = 2*(precision * recall) / (precision + recall)
    except:
        print("devide by zero")
        precision =455
        recall =455
        f1_score = 500
    return precision, recall, f1_score

precision = []
recall = []
f1_score = []
for i in range(10):
    pre, rec, f1 = calculate_pre_rec_f1(classes_info[i])
    precision.append(pre)
    recall.append(rec)
    f1_score.append(f1)
    print('Class %s -> Precision: %s * Recall: %s * F1-score: %s' %(i, pre, rec, f1))
    print("==========================================================\n")
