from matplotlib import pyplot as plt

def valueHistogram(array):
    #this function sucks
    flattenArray = [x for x in array.flatten("C")]
    plt.hist(x=flattenArray, bins='auto')

def standardPlot(array):
    plt.imshow(array, cmap='gray')
    plt.show()

