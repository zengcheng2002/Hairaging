import numpy as np
import cv2
TEXSIZE = 2048
FACIAL = 13
# Constants
K = 8  # Number of clusters
normal_weight = 0.75
def create_parameter_texture(labels, map_cluster_to_region):
    param_img = np.zeros((TEXSIZE, TEXSIZE), dtype=np.uint8)
    b1= np.zeros(1, dtype=np.float32)
    b1[0] = FACIAL
    generator = np.random.default_rng()
    b1_facial = generator.normal(68, b1[0], size=(TEXSIZE, TEXSIZE))
    for y in range(TEXSIZE):
        for x in range(TEXSIZE):
            cluster_idx = labels[y, x]
            head_region = map_cluster_to_region[cluster_idx]
            if head_region ==5 or head_region ==6:  # for Beard and eyebrows
                param_img[y, x] = int(abs(np.floor(b1_facial[y, x])))
            else:
                param_img[y, x] = 0
    pckg_name = "D:\\Users\\dell\\Desktop\\HairAging10" # you can modify this to the address that you want to store the result
    cv2.imwrite(pckg_name + ".png", param_img)

def modify_normal_map(img):
    img = img.astype(np.float32)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j, 2] < 128:  # mirror y axis
                img[i, j, 2] = 255 - img[i, j, 2]

            img[i, j, 1] = int(img[i, j, 1] * normal_weight)  # scales the normal value in this axis, allowing K-Means to be more precise
            img[i, j, 2] = int(img[i, j, 2] * normal_weight)

    img = img.astype(np.uint8)
    return img

def apply_clustering(img):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    img = img.astype(np.float32)
    points = img.reshape((-1, img.shape[2])).astype(np.float32)
    _, cluster_labels, centers = cv2.kmeans(points, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    cluster_labels = cluster_labels.reshape((img.shape[0], img.shape[1]))
    segmentation_img = centers[cluster_labels]
    centroids = centers.copy()  # save the new centroid values, which will be used to order the segmented regions
    print(centroids)
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
    map_cluster_to_region= [1,4,0,2,6,7,5,3]
    path = 'D:\PycharmProjects\pythonProject\\sampledNormals.png' # The address you store your normal map
    cluster_labels,new_centroids =segmentation(path)
    create_parameter_texture(cluster_labels, map_cluster_to_region)
main()