from matplotlib import pyplot as plt

def valueHistogram(array):
    #this function sucks
    flattenArray = [x for x in array.flatten("C")]
    plt.hist(x=flattenArray, bins='auto')

def standardPlot(array):
    plt.imshow(array, cmap='gray')
    plt.show()

def dualPlot(array1, array2):
    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].plot(array1)
    axarr[1].scatter(array2)
    plt.show()

def tripplePlot(array1, array2, array3):
    plt.subplot(1,3,1)
    plt.imshow(array1, cmap='gray')
    plt.subplot(1,3,2)
    plt.imshow(array2, cmap='gray')
    plt.subplot(1,3,3)
    plt.imshow(array3, cmap='gray')
    plt.show()
