import numpy as np
import cv2

image_file = 'Sierpinski.png'

def read_image_rgb(image_file):
    #Return an height * width matrix of 3 elements (0-256) array for red green and blue proportions
    image_data = cv2.imread(image_file,1)
    #image_show(image_data,'Colour')
    height = image_data.shape[0]
    width = image_data.shape[1]
    return (image_data,height,width)

def read_image_grey(image_file):
    #Return an height * width matrix of elements (0-256) for gray scale
    image_data = cv2.imread(image_file,0)
    #image_show(image_data,'Greyscale')
    height = image_data.shape[0]
    width = image_data.shape[1]
    return (image_data,height,width)

def image_show(image_data,name):
    cv2.imshow(name,image_data)
    cv2.waitKey()
    cv2.destroyAllWindows()
    return 1

def image_grey_to_bool(image_data_grey,threshold):
    image_data_bool = cv2.inRange(image_data_grey,0,255*threshold)
    image_data_bool = image_data_bool / 255
    return image_data_bool

def fractal_dimension(image_data_grey, threshold=0.4):
    
    # From https://github.com/rougier/numpy-100 (#87)
    def boxcount(image_data_bool, box_size):
        no_of_pixels = np.add.reduceat(
            np.add.reduceat(image_data_bool, 
                        np.arange(0, image_data_bool.shape[0], box_size), axis=0),
                        np.arange(0, image_data_bool.shape[1], box_size), axis=1)
        # We count non-empty (0) and non-full boxes (k*k)
        return len(np.where((no_of_pixels > 0) & (no_of_pixels < box_size ** 2))[0])
        
    # Transform into a binary array

    image_data_bool = image_grey_to_bool(image_data_grey,threshold)
    
    #image_show(image_data_bool,"Boolean Image")

    print(image_data_bool)
    # Minimal dimension of image
    p = min(image_data_bool.shape)
    
    # Greatest power of 2 less than or equal to p
    n = 2 ** np.floor(np.log(p)/np.log(2))

    # Extract the exponent
    n = int(np.log(n)/np.log(2))

    # Build successive box sizes (from 2**n down to 2**1)
    sizes = 2 ** np.arange(n, 1, -1)

    # Actual box counting with decreasing size
    counts = []
    for size in sizes:
        counts.append(boxcount(image_data_bool, size))
    print(counts)
    # Fit the successive log(sizes) with log (counts)
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]
    
image_data,height,width = read_image_grey(image_file)

print("Minkowski–Bouligand dimension (computed): ", fractal_dimension(image_data))

