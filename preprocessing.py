''' '''

def preprocess_mnist(x_train, y_train, x_test, y_test):

    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    y_train = to_categorical(y_train, dtype='uint8')
    y_test = to_categorical(y_test, dtype='uint8')

    return (x_train, y_train), (x_test, y_test)
