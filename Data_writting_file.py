import mnist_loader
import network
import pickle

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()


net = network.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)


with open('weights', 'wb') as fp:
    pickle.dump(net.weights, fp)
fp.close()

with open('biases', 'wb') as fp:
    pickle.dump(net.biases, fp)
fp.close()

##pickle_off = open('weights', 'rb')
##weights = pickle.load(pickle_off)
##print(sizes)

##pickle_off = open('biases', 'rb')
##biases = pickle.load(pickle_off)
##print(sizes)
