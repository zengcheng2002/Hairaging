import numpy as np
import cv2

# Constants
K = 8  # Number of clusters
normal_weight = 0.75
TEXSIZE = 2048
# Enumerations
NUM_REGIONS = 5
TEMPORAL = 0
FRONTAL = 1
PARIETAL = 2
VERTEX = 3
OCCIPITAL = 4
FACE = 5
NECK = 6
OTHER = 7

# Constants for male and female SD
MALE_TEMPORAL = 16
MALE_VERTEX = 14
MALE_OCCIPITAL = 11
MALE_OTHERS = 13

FEMALE_TEMPORAL = 13
FEMALE_VERTEX = 13
FEMALE_OCCIPITAL = 11
FEMALE_Parietal= 16

def create_parameter_texture(labels, map_cluster_to_region, b_is_male_character):
    param_img = np.zeros((TEXSIZE, TEXSIZE), dtype=np.uint8)

    b1 = np.zeros(NUM_REGIONS, dtype=np.float32) #store the SD of different regions in a array
    if b_is_male_character:
        b1[TEMPORAL] = MALE_TEMPORAL
        b1[FRONTAL] = MALE_OTHERS
        b1[PARIETAL] = MALE_OTHERS
        b1[VERTEX] = MALE_VERTEX
        b1[OCCIPITAL] = MALE_OCCIPITAL

    else:
        b1[TEMPORAL] = FEMALE_TEMPORAL
        b1[FRONTAL] = FEMALE_Parietal
        b1[PARIETAL] = FEMALE_Parietal
        b1[VERTEX] = FEMALE_VERTEX
        b1[OCCIPITAL] = FEMALE_OCCIPITAL

    labels = labels.astype(np.uint8)

    generator = np.random.default_rng()   #Insert parameter using random number
    b1_temporal = generator.normal(63, b1[TEMPORAL], size=(TEXSIZE, TEXSIZE))
    b1_vertex = generator.normal(63, b1[VERTEX], size=(TEXSIZE, TEXSIZE))
    b1_frontal = generator.normal(63, b1[FRONTAL], size=(TEXSIZE, TEXSIZE))
    b1_parietal = generator.normal(63, b1[PARIETAL], size=(TEXSIZE, TEXSIZE))
    b1_occipital = generator.normal(63, b1[OCCIPITAL], size=(TEXSIZE, TEXSIZE))

    for y in range(TEXSIZE):
        for x in range(TEXSIZE):  # insert parameter in every pixel
            cluster_idx = labels[y, x]
            head_region = map_cluster_to_region[cluster_idx]
            if head_region == TEMPORAL:
                param_img[y, x] = int(abs(np.floor(b1_temporal[y, x])))
            elif head_region == VERTEX:
                param_img[y, x] = int(abs(np.floor(b1_vertex[y, x])))
            elif head_region == OCCIPITAL:
                param_img[y, x] = int(abs(np.floor(b1_occipital[y, x])))
            elif head_region == PARIETAL:
                param_img[y, x] = int(abs(np.floor(b1_parietal[y, x])))
            elif head_region == FRONTAL:
                param_img[y, x] = int(abs(np.floor(b1_frontal[y, x])))
            else:
                param_img[y, x] = 0

    pckg_name = "D:\\Users\\dell\\Desktop\\HairAging13"  # you can modify this to the address that you want to store the result
    cv2.imwrite(pckg_name + ".png", param_img)

def modify_normal_map(img): # for modify the normal map
    img = img.astype(np.float32)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j, 2] < 128:  # mirror y axis
                img[i, j, 2] = 255 - img[i, j, 2]

            img[i, j, 1] = int(img[i, j, 1] * normal_weight)  # scales the normal value in this axis, allowing K-Means to be more precise
            img[i, j, 2] = int(img[i, j, 2] * normal_weight)

    img = img.astype(np.uint8)
    return img

def apply_clustering(img):  # for segmentation
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    img = img.astype(np.float32)
    points = img.reshape((-1, img.shape[2])).astype(np.float32)
    _, cluster_labels, centers = cv2.kmeans(points, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    cluster_labels = cluster_labels.reshape((img.shape[0], img.shape[1]))
    segmentation_img = centers[cluster_labels]
    centroids = centers.copy()  # save the new centroid values, which will be used to order the segmented regions
    segmentation_img = segmentation_img.astype(np.uint8)
    pckg_name = "D:/Users/dell/Desktop/HairAging/"
    tex_name = "Segmentation1_" # you can modify this to the address that you want to store the result of segmentation
    cv2.imwrite(pckg_name + tex_name + ".png", segmentation_img)  # creates a texture for the segmentation
    return cluster_labels, centroids

def segmentation(path):
    img = cv2.imread(path)
    img = modify_normal_map(img)
    labels, centroids = apply_clustering(img)
    return labels,centroids

def main():
    bIsMaleCharacter=False  # use this to control the sex, True for male, False for female
    path = 'D:\PycharmProjects\pythonProject\\sampledNormals.png' # The address you store your normal map
    map_cluster_to_region= [1,4,0,2,6,7,5,3]  # correspondence between labels and regions
    cluster_labels,new_centroids =segmentation(path)
    create_parameter_texture(cluster_labels, map_cluster_to_region, bIsMaleCharacter)
main()


