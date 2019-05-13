from HodaDatasetReader import read_hoda_dataset, read_hoda_cdb
from matplotlib import pyplot as plt

print('Reading Train 60000.cdb ...')
train_images, train_labels = read_hoda_cdb('./DigitDB/Train 60000.cdb')

print('Reading Test 20000.cdb ...')
test_images, test_labels = read_hoda_cdb('./DigitDB/Test 20000.cdb')



plt.imshow(train_images[0], cmap='gray')
plt.title("Plot for %s (train data)" %train_labels[0])
plt.show()

plt.imshow(test_images[1], cmap='gray')
plt.title("Plot for %s (test data)" %test_labels[0])
plt.show()
