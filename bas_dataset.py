import matplotlib.pyplot as plt
import numpy as np
import pydeep.base.numpyextension as npext

def random_sample_bas(length, num_samples):
    """ Creates a dataset containing random samples showing bars or stripes.
    :param length: Length of the bars/stripes.
    :type length: int
    :param num_samples: Number of samples
    :type num_samples: int
    :return: Samples.
    :rtype: numpy array [num_samples, length*length]
    """

    stripes = npext.generate_binary_code(length)
    stripes = np.repeat(stripes, length, 0)
    stripes = stripes.reshape(2 ** length, length * length)

    bars = npext.generate_binary_code(length)
    bars = bars.reshape(2 ** length * length, 1)
    bars = np.repeat(bars, length, 1)
    bars = bars.reshape(2 ** length, length * length)
    data = np.vstack((stripes[0:stripes.shape[0]-1],bars[1:bars.shape[0]]))

    distrib = np.zeros(2**(length*length))
    for sample in range(num_samples):
        i = np.random.randint(0,len(data))
        bin_list = [int(data[i,j] ) for j in range(length**2)]
        bin_string = ''
        for bit in bin_list:
            bin_string += str(bit)
        number = int(bin_string, 2)
        distrib[number] += 1/num_samples
    return distrib

def generate_bas_image(length, num_samples):
    """ Creates a dataset containing random samples showing bars or stripes.
    :param length: Length of the bars/stripes.
    :type length: int
    :param num_samples: Number of samples
    :type num_samples: int
    :return: Samples.
    :rtype: numpy array [num_samples, length*length]
    """
    data = np.zeros((num_samples, length * length))
    for i in range(num_samples):
        values = np.dot(np.random.randint(low=0, high=2, size=(length, 1)), np.ones((1, length)))
        if np.random.random() > 0.5:
            values = values.T
        data[i, :] = values.reshape(length * length)
    return data

def generate_bas_complete(length):
    """ Creates a dataset containing all possible samples showing bars or stripes and its distribution.
    :param length: Length of the bars/stripes.
    :type length: int
    :return: Samples.
    :rtype: numpy array [num_samples, length*length]
    """

    stripes = npext.generate_binary_code(length)
    stripes = np.repeat(stripes, length, 0)
    stripes = stripes.reshape(2 ** length, length * length)

    bars = npext.generate_binary_code(length)
    bars = bars.reshape(2 ** length * length, 1)
    bars = np.repeat(bars, length, 1)
    bars = bars.reshape(2 ** length, length * length)
    data = np.vstack((stripes[0:stripes.shape[0]-1],bars[1:bars.shape[0]]))
    #print(data)
    # generate distribution
    distrib = np.zeros(2**(length*length))
    for i in range(len(data)):
        bin_list = [int(data[i,j] ) for j in range(length**2)]
        bin_string = ''
        for bit in bin_list:
            bin_string += str(bit)
        number = int(bin_string, 2)
        distrib[number] = 1/len(data)
    return distrib

if __name__ == "__main__":

    #distrib = generate_bas_distribution(2,10)
    #print(distrib)
    #print(np.sum(distrib))
    #plt.plot(distrib)
    #plt.show()
    '''
    distrib = generate_bas_image(2)
    plt.plot(distrib, 'ro', label = 'True distribution')
    plt.xlabel('Data')
    plt.ylabel('Probability')
    plt.legend()    
    plt.show()
    '''
    
    num_samples = 10
    data = generate_bas_image(2,2)
    print(data)
    for i in range (num_samples):
        sample = data[i,:].reshape(2,2)
        print(sample)
        sample = np.array([[1. ,0.], [0., 0.]])
        plt.imshow(sample, cmap = 'flag')
        plt.xticks([])
        plt.yticks([])
        plt.show()



'''
def random_sample_bas(length, num_samples):
    """ Creates a dataset containing random samples showing bars or stripes.
    :param length: Length of the bars/stripes.
    :type length: int
    :param num_samples: Number of samples
    :type num_samples: int
    :return: Samples.
    :rtype: numpy array [num_samples, length*length]
    """
    data = np.zeros((num_samples, length * length))
    for i in range(num_samples):
        values = np.dot(np.random.randint(low=0, high=2, size=(length, 1)), np.ones((1, length)))
        if np.random.random() > 0.5:
            values = values.T
        data[i, :] = values.reshape(length * length)

    distrib = np.zeros(2**(length*length))
    for i in range(len(data)):
        bin_list = [int(data[i,j] ) for j in range(length**2)]
        bin_string = ''
        for bit in bin_list:
            bin_string += str(bit)
        number = int(bin_string, 2)
        distrib[number] += 1/len(data)
    return distrib
'''