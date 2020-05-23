import numpy as np
import cv2
import math
import random
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


image_file = 'Boat.jpg'

image_processing_threshold = 0.65

identification_tolerance = 98 #percentage of object identification tollerance

patch_size = 8
step = 3

dimension_map_file = 'Dimension_Map.jpg'
irregularities_map_file = "Irregularitites_Map.jpg"

def read_image_rgb(image_file):
    #Return an height * width matrix of 3 elements (0-256) array for red green and blue proportions
    image_data = cv2.imread(image_file,1)
    #image_show(image_data,'Colour')
    return image_data


def read_image_grey(image_file):
    #Return an height * width matrix of elements (0-256) for gray scale
    image_data = cv2.imread(image_file,0)
    #image_show(image_data,'Greyscale')
    height = image_data.shape[0]
    width = image_data.shape[1]
    return image_data


def decompose_image_colours(image_rgb_data):
    print(image_rgb_data.shape)
    image_blue = image_rgb_data[:,:,0]
    image_green = image_rgb_data[:,:,1]
    image_red = image_rgb_data[:,:,2]
    return (image_red, image_green, image_blue)


def image_show(image_data,name):
    cv2.imshow(name,image_data)
    cv2.waitKey()
    cv2.destroyAllWindows()
    return 1


def image_save(image_data,file_name):
     cv2.imwrite(file_name,image_data)
     return 1


def image_grey_to_bool(image_data_grey,threshold):
    image_data_bool = cv2.inRange(image_data_grey,0,255*threshold)
    image_data_bool = image_data_bool / 255
    return image_data_bool


def boxcount(image_data_bool, box_size):
    # From https://github.com/rougier/numpy-100 (#87)
        no_of_pixels = np.add.reduceat(
            np.add.reduceat(image_data_bool, 
                        np.arange(0, image_data_bool.shape[0], box_size), axis=0),
                        np.arange(0, image_data_bool.shape[1], box_size), axis=1)
        # We count non-empty (0) and non-full boxes (k*k)
        #return len(np.where((no_of_pixels > 0) & (no_of_pixels < box_size ** 2))[0])
        return len(np.where((no_of_pixels > 0) & (no_of_pixels < box_size ** 2))[0])


def fractal_dimension(image_data_bool):
    
    
    #image_show(image_data_bool,"Boolean Image")

    # Minimal dimension of image
    p = min(image_data_bool.shape)
    
    # Greatest power of 2 less than or equal to p
    n = 2 ** np.floor(np.log(p) / np.log(2))

    # Extract the exponent
    n = int(np.log(n)/np.log(2))

    # Build successive box sizes (from 2**n down to 2**1)
    sizes = 2 ** np.arange(n, 1, -1)

    # Actual box counting with decreasing size
    counts = []
    for size in sizes:
        counts.append(boxcount(image_data_bool, size))

    # Fit the successive log(sizes) with log (counts)
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    
    return -coeffs[0]


def select_patch(image_data,center,patch_size):
    height = image_data.shape[0]
    width = image_data.shape[1]
    (i,j) = center
    patch_len = math.floor(patch_size / 2)

    start_i = max(0, i - patch_len)
    stop_i = min(height, i + patch_len)
    start_j = max(0, j - patch_len)
    stop_j = min(width, j + patch_len)

    return image_data[start_i : stop_i, start_j : stop_j]


def compute_dimension_map(image_grey,patch_size, step,image_processing_threshold):

    image_bool = image_grey_to_bool(image_grey,image_processing_threshold)
    
    #image_show(image_bool,'Bool Image')

    height = image_bool.shape[0]
    width = image_bool.shape[1]

    output = np.zeros((height,width))

    prog = 0
    for i in range(0,height,step):
        for j in range(0,width,step):
        
            patch = select_patch(image_bool,(i,j),patch_size)
            score = (fractal_dimension(patch)) / 2 
            output[i][j] = score 
            
            #Display Progress
            current_prog = int((i * width + j) * 100 / (height * width))
            if current_prog != prog:
                prog = current_prog
                print("Computing Dimension: ", prog, "%") 
    
            #Fill up cells between steps
            for dy in range(0,step):
                if dy + j < width:
                    for dx in range(0,step):
                        if dx + i < height:
                            output[i + dx][j + dy] = output[i][j]
    return output


def detect_dimension_irregularities(dimension_map):
    height = dimension_map.shape[0]
    width = dimension_map.shape[1]

    average_dimension = 0 
    samples = 0

    min_dimension = 0
    max_dimension = 2

    for i in range(height):
        for j in range(width):

            if dimension_map[i][j] == dimension_map[i][j]:

                #print("Dimension: ", dimension_map[i][j])

                average_dimension = (average_dimension * samples + dimension_map[i][j]) / (samples + 1)
                samples += 1

                if dimension_map[i][j] < min_dimension:
                    min_dimension = dimension_map[i][j]

                if dimension_map[i][j] > max_dimension:
                    max_dimension = dimension_map[i][j]

    irregularities_map = dimension_map

    max_dimension -= average_dimension
    min_dimension -= average_dimension
    domain = max_dimension - min_dimension

    print("Average: ", average_dimension)
    print("Minimum dimension: ",min_dimension)
    print("Maximum dimension: ",max_dimension)

    for i in range(0,height):
        for j in range(0,width):

            irregularities_map[i][j] = dimension_map[i][j] - average_dimension
            irregularities_map[i][j] = (irregularities_map[i][j] - min_dimension) / domain
            
            #print("Irregularities Map: ", irregularities_map[i][j])

    return irregularities_map


def add_selection_to_image(image_file,dimension_map_file,threshold):
    
    print("Adding Contour to original Image")
    
    image_data = read_image_rgb(image_file)
    dimension_map = read_image_grey(dimension_map_file)

    #image_show(dimension_map,'Dimension Map')

    height = dimension_map.shape[0]
    width = dimension_map.shape[1]

    for i in range (height):
        for j in range(width):
    
            if dimension_map[i][j] != 0 and dimension_map[i][j] <= threshold:
                dimension_map[i][j] = 255
            else:
                dimension_map[i][j] = 0

    dimension_map = cv2.medianBlur(dimension_map, 9)

    ret, thresh = cv2.threshold(dimension_map, 102 , 255, 0)

    kernel = np.ones((5,5), np.float32)/10
    dst = cv2.filter2D(thresh, -1, kernel)

    contour1, hierarchy = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image_data , contour1, -1, (0, 0, 255), 3 )

    return image_data


#__________________Program Running__________________




#image_data_grey = read_image_grey(image_file)

#box_counting_dimension_map = compute_dimension_map(image_data_grey,patch_size,step,image_processing_threshold)


#irregularities_map = detect_dimension_irregularities(box_counting_dimension_map)

#image_save(irregularities_map * 255, irregularities_map_file)


output = add_selection_to_image(image_file, irregularities_map_file, threshold = 255)

image_save(output,'Result of Box Counting Detection(' + str(253) + '.jpg')
