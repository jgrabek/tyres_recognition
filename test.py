import numpy as np
from skimage import exposure
from skimage import img_as_float
import os

def vector_median(image, arg):
    from skimage.filters import median
    img = 2 * median(image) + image
    vctr = np.reshape(img, (1, len(img) * len(img[0])))

    return vctr[0]


def vector_sobel(image, arg):
    from skimage.filters import sobel
    img = 2 * sobel(image) + image
    vctr = np.reshape(img, (1, len(img) * len(img[0])))
    return vctr[0]


def vector_canny(image, arg):
    from skimage.feature import canny
    from scipy import ndimage as ndi
    filtered = ndi.gaussian_filter(image, 1)
    edges = canny(filtered, sigma=1.4, low_threshold=0.05, high_threshold=0.2)
    img = edges + image
    vctr = np.reshape(img, (1, len(img) * len(img[0])))
    return vctr[0]


def vector_sharpened(image, arg):
    from scipy.ndimage import gaussian_filter
    alpha = 1
    filtered = gaussian_filter(image, 1)
    img = 2 * image + alpha * (image - filtered)
    vctr = np.reshape(img, (1, len(img) * len(img[0])))
    return vctr[0]


def vector_hist(image, nbins):
    vctr, b = exposure.histogram(img_as_float(image), nbins=nbins)
    out = vctr
    out = out / max(out)
    return out


def vector_hog(image, arg):
    filtered = image
    from skimage.feature import hog
    fd = hog(filtered, orientations=8, pixels_per_cell=(4, 4),
                        cells_per_block=(1, 1), visualise=False, feature_vector=True)

    mean = np.hstack((np.mean(image, axis=0), np.mean(image, axis=1)))
    mean = np.hstack((mean, np.mean(image)))

    tmp = np.hstack((fd, mean))
    return tmp

def grid_search(path, test_path, roi_size, vector_length):
    from ImageContainer import ImageContainer
    type_ = 'hog'
    container = ImageContainer(path,'')
    print(type_)
    container.set_type(type_)
    container.set_function(vector_hog, 0)

    c_min = 63
    c_max = 64
    c_step = 1
    gamma_min = 7
    gamma_max = 8
    gamma_step = 2
    index = 0
    index_max = 0
    for c_ in range(c_min, c_max, c_step):
        for gamma_ in range(gamma_min, gamma_max, gamma_step):
            index_max += 1


    container.cut_images_roi(size=roi_size, rotate=True, vector_length=vector_length)

    id, vector, m = container.get_vector()
    name = "sigmoid"#"rbf"
    from sklearn import svm
    import time
    file = open((name + 'data.txt'), 'a')
    start_time = time.time()
    file.write(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
    file.write("\n")
    file.close()

    for c_ in range(c_min, c_max, c_step):
        for gamma_ in range(gamma_min, gamma_max, gamma_step):
            print("progress %f%%" % (float(index / index_max) * 100.0))
            index += 1
            c = float(c_)
            gamma = float(gamma_) / 10000.0

            clf = svm.SVC(C=c, kernel=name, gamma=gamma, tol=1e-3, probability=True)
            clf.fit(vector, id)

            from time import gmtime, strftime

            directory = 'img/' + type_ + strftime("%H:%M:%S", gmtime()) + '/'
            os.makedirs(directory)

            for i in range(len(test_path)):
                label = type_ + ' ' + name + ' ' + str(i) + ' ' + 'C ' + str(int(c * 100)) + 'gamma ' + str(
                    int(gamma * 1000000))
                pic_time = time.time() * 1000.0
                tmp = test_path[i]
                container.get_check_image(clf, test_path[i], label=label, path=directory, id=-1,
                                          interesting_areas_status=False)
                print("pic %d: %dms" % (i, (time.time() * 1000.0 - pic_time)))
            end_time = time.time()
            print("exec time: %s s" % (end_time - start_time))
            start_time = end_time
            precision, recall, f1 = container.calculate_params()
            values = "c: " + str(c) + " gamma: " + str(gamma)
            params = "precision: " + str(precision) + " recall: " + str(recall) + " f1: " + str(f1)
            file = open((name + 'data.txt'), 'a')
            file.write((values + ' ' + params + "\n"))
            print((values + ' ' + params + "\n"))
            file.close()
    file = open((name + 'data.txt'), 'a')
    file.write(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
    file.write("\n")
    file.close()

def main():
    roi_size = (51, 51)
    vector_length = roi_size[0] * roi_size[1]
    directory = 'original/'
    files = ("00001.tiff",
             "00002.tiff",
             "00003.tiff",
             "00004.tiff",
             "00005.tiff",
             "00006.tiff",
             "00007.tiff",
             "00008.tiff",
             "00009.tiff",
             "00010.tiff",
             "00011.tiff",
             "00012.tiff"
             )
    path = []
    for item in files:
        path.append(directory + item)
    test_path = (
        "original/test_files/1.tiff",
        "original/test_files/2.tiff",
        "original/test_files/3.tiff",
        "original/test_files/4.tiff",
        "original/test_files/5.tiff"
    )
    grid_search(path, test_path, roi_size, vector_length)

    return


if __name__ == "__main__":
    main()