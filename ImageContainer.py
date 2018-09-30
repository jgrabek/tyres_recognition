import numpy as np

from skimage.color import rgb2gray
from skimage import exposure
from skimage import img_as_float
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.feature import hog


def filter(image, scale=0.4, const=1.35):

    filt = np.zeros((700, len(image[0])))
    half_length = int(len(image[0]) / 2)
    diff = 65
    max_value = half_length - diff
    x = np.array(range(0, max_value))
    left_gain = 0
    right_gain = 0
    y1 = scale * np.log((max_value - x) / max_value) + left_gain * (max_value - x) + const
    y2 = scale * np.log((x + 0.000001) / max_value) + right_gain * (max_value - x) + const
    z = np.concatenate((y1, np.zeros((diff * 2)), y2))
    tmp = z > 0
    z = np.multiply(z, tmp)
    for row in range(0, 700):
        filt[row] = z

    filtered = rgb2gray(np.multiply(rgb2gray(image), filt))

    return filtered


class ImageContainer:
    def __init__(self, file_list, exclude):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.images_prepared = []
        self.images_original = []
        self.images_roi = []
        self.roi_list = []
        self.roi_hit = []
        self.roi_type_cnt = [0, 0, 0, 0]
        self.iter = 0
        self.size = (0, 0)
        self.vector_length = 10
        self.type = 'hist'
        self.rotate = False
        self.add_images(file_list)
        self.fcn = None
        self.additional_arg = None
        self.exclude = exclude
        self.index = 0

    def calculate_params(self):
        if self.tp + self.fp > 0.0:
            precision = float(self.tp) / float(self.tp + self.fp)
        else:
            precision = 0
        if self.tp + self.fn > 0.0:
            recall = float(self.tp) / float(self.tp + self.fn)
        else:
            recall = 0
        if recall > 0 and precision > 0.0:
            f1 = 2.0 / ((1.0 / recall) + (1.0 / precision))
        else:
            f1 = 0
        self.tp = 0
        self.fn = 0
        self.fp = 0
        print("Precision: %f Recall: %f, F1 score: %f" % (precision, recall, f1))
        return precision, recall, f1

    def __prepare_image(self, imag):
        from skimage.filters import median, gaussian

        img_max, img_min = imag.max(), imag.min()
        if (img_max - img_min) > 0:
            imag = (imag - img_min) / (img_max - img_min)

        imag = median(imag)

        filtered = gaussian(imag, 2)
        imag = imag + 0.3 * (imag - filtered)

        img_max, img_min = imag.max(), imag.min()
        if (img_max - img_min) > 0:
            imag = (imag - img_min) / (img_max - img_min)

        return imag

    def __vector_from_histogram(self, image, nbins):

        vctr, b = exposure.histogram(img_as_float(image), nbins=nbins)
        out = vctr
        out = out / max(out)
        return out

    def __cut_coord_from_image(self, image, coord):
        is_coords_good = True

        for item in coord:
            if item < 0:
                is_coords_good = False
                break

        if not (max(coord[1], coord[3]) <= max(len(image), len(image[0])) and
                min(coord[1], coord[3]) <= min(len(image), len(image[0]))):
            is_coords_good = False

        if is_coords_good:
            x1 = coord[0]
            x2 = coord[1]
            y1 = coord[2]
            y2 = coord[3]
            img = image[y1:y2, x1:x2]

            if self.rotate and (np.max((x1, x2)) < len(image[0] / 2)):
                img = np.flip(img, 0)

            return img




    def __get_vector_for_image(self, image, coord):
        vctr = []
        img = self.__cut_coord_from_image(image, coord)

        if img is not None:
            filtered = img
            fd = hog(filtered, orientations=8, pixels_per_cell=(4, 4),
                     cells_per_block=(1, 1), visualise=False, feature_vector=True)

            mean = np.hstack((np.mean(img, axis=0), np.mean(img, axis=1)))
            mean = np.hstack((mean, np.mean(img)))
            vctr = np.hstack((fd, mean))

        return vctr

    def _get_point_coords(selfself, point, size):
        coord = []
        half_size_x = int(size[0] / 2)
        half_size_y = int(size[1] / 2)

        tmp = point

        tmp[1] = tmp[1] - 15

        coord.append(tmp[0] - half_size_x)
        coord.append(tmp[0] - half_size_x + size[0])
        coord.append(tmp[1] - half_size_y)
        coord.append(tmp[1] - half_size_y + size[1])

        coords = (coord[0], coord[1], coord[2], coord[3])
        return coords

    def __get_roi_coords(self, roi_item, size):
        length = len(roi_item[2])
        coord = []
        if length == 2:
            half_size_x = int(size[0] / 2)
            half_size_y = int(size[1] / 2)

            tmp = roi_item

            tmp[2][1] = tmp[2][1] - 15

            coord.append(tmp[2][0] - half_size_x)
            coord.append(tmp[2][0] - half_size_x + size[0])
            coord.append(tmp[2][1] - half_size_y)
            coord.append(tmp[2][1] - half_size_y + size[1])

        else:
            print("null")
            return 0

        coords = (coord[0], coord[1], coord[2], coord[3])

        return coords

    def __read_image(self, file_path, show_image=False):
        from skimage import io
        imag = io.imread(file_path)
        data_gray = rgb2gray(imag)
        imag = filter(data_gray)
        self.images_original.append(imag)
        imag = self.__prepare_image(imag)

        if show_image:
            io.imshow(imag)
            io.show()

        return imag

    def __read_image_roi(self, file_path, show_data=False):
        handler = open(file_path, "r")
        roi_list = []
        hit_list = []
        for line in handler:
            words = line.strip().split(",")
            if len(words) > 2:
                coords = []
                numbers = words[2:]
                for item in numbers:
                    coords.append(int(item))
                roi_list.append((words[0], words[1], coords))
                if int(words[1]) == 3:
                    hit_list.append(coords)
                if show_data:
                    print(line)

        handler.close()
        return roi_list

    def __cut_roi_image(self, image, roi_list):
        imageList = []
        plt.close()
        plt.gray()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(image)
        for it in roi_list:
            coord = self.__get_roi_coords(it, self.size)
            x1 = coord[0]
            y1 = coord[2]
            if int(it[1]) == 0:
                rect = mpatches.Rectangle((x1, y1), self.size[0], self.size[1], fill=False, edgecolor='yellow',
                                        linewidth=1)
                ax.add_patch(rect)
            elif int(it[1]) == 1:
                rect = mpatches.Rectangle((x1, y1), self.size[0], self.size[1], fill=False, edgecolor='green',
                                        linewidth=1)
                ax.add_patch(rect)
            elif int(it[1]) == 3:
                rect = mpatches.Rectangle((x1, y1), self.size[0], self.size[1], fill=False, edgecolor='blue',
                                        linewidth=1)
                ax.add_patch(rect)

            # data_rescale = self.__prepare_image(image)
            vctr = self.__get_vector_for_image(image, coord)
            if len(vctr) > 2:
                imageList.append((0, it[1], vctr))
                self.roi_type_cnt[int(it[1])] += 1
        ax.set_axis_off()
        plt.tight_layout()
        plt.savefig("original/" + str(self.index) + ".png")
        self.index += 1
        plt.close()

        return imageList

    def __find_interesting_areas(self, image):
        from skimage.feature import canny
        from scipy import ndimage as ndi
        filtered = ndi.gaussian_filter(image, 1)
        image = image - 0.6 * filtered
        image = (image - image.min())/(image.max() - image.min())
        diff = image.max() - image.min()
        img = canny(image, sigma=3, low_threshold=diff*0.04, high_threshold=diff*0.19)
        img = ndi.binary_dilation(img)
        img = ndi.binary_dilation(img)
        img = ndi.binary_closing(img)
        img = ndi.binary_fill_holes(img)
        return img

    def _comapreCoords(self, c1, c2, diff):
        value = 0
        for i in range(len(c1)):
            if c2[i] >= c1[i] - diff and c2[i] <= c1[i] + diff:
                value += 1
            else:
                break
        if value == 4:
            return True
        else:
            return False

    def checkHit(self, coord, hit_points, hit_points_cp, diff):
        pos_y = coord[0] + (coord[1] - coord[0]) / 2.0
        pos_x = coord[2] + (coord[3] - coord[2]) / 2.0
        for i in range(len(hit_points)):
            lower = hit_points[i][2][0] - diff
            upper = hit_points[i][2][0] + diff
            if  lower <= pos_y <= upper:
                lower = hit_points[i][2][1] - diff
                upper = hit_points[i][2][1] + diff
                if lower - diff <= pos_x <= upper + diff:
                    hit_points_cp.pop(i)
                    return 1
        return 0

    def get_vector(self):
        id = []
        vctr = []
        max_value = 0
        for item in self.images_roi:
            if int(item[1]) != 2:
                id.append(item[1])
                vctr.append(item[2])
        return id, vctr, max_value

    # def multiproc(self,match,y,ax, ):

    def get_check_image(self, classifier, img, label='', path='img/', id=-1, interesting_areas_status=True):
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        plt.ioff()
        hit_points = []
        hit_points_cp = []

        if type(img) is str:
            from skimage import io
            image = io.imread(img)
            tmp = img.strip().split(".")[0]
            hit_points = self.__read_image_roi(tmp + '.tiff.txt')
            hit_points_cp = hit_points
            fn = len(hit_points)
            data_gray = rgb2gray(image)
            image = filter(data_gray)
        else:
            image = self.images_original[img]
        image = self.__prepare_image(image)

        plt.close()
        plt.gray()

        if int(id) >= 0:
            fig, ax = plt.subplots(figsize=(10, 6))

            ax.imshow(image)

            match = [0, 0, 0, 0, 0]
            for y in range(0, len(image) - self.size[1], int(self.size[1] / 4)):
                for x in range(0, len(image[0]) - self.size[0], int(self.size[0] / 4)):
                    match[4] = match[4] + 1
                    coord = [x, x + self.size[0], y, y + self.size[1]]
                    vctr = self.__get_vector_for_image(image, coord)
                    vector = np.zeros((2, len(vctr)))
                    vector[0, :] = vctr

                    out = classifier.predict(vector)

                    if int(out[0]) == 0:
                        if int(out[0]) == 0:
                            rect = mpatches.Rectangle((x, y), self.size[0], self.size[1], fill=False, edgecolor='yellow',
                                                   linewidth=1)
                            ax.add_patch(rect)
                        match[0] = match[0] + 1
                    elif int(out[0]) == 1:
                        if int(id) == 1:
                            rect = mpatches.Rectangle((x, y), self.size[0], self.size[1], fill=False, edgecolor='green',
                                                linewidth=1)
                            ax.add_patch(rect)
                        match[1] = match[1] + 1
                    elif int(out[0]) == 2:
                        if int(id) == 2:
                            rect = mpatches.Rectangle((x, y), self.size[0], self.size[1], fill=False, edgecolor='red',
                                              linewidth=1)
                            ax.add_patch(rect)
                        match[2] = match[2] + 1
                    elif int(out[0]) == 3:
                        if int(id) == 3:
                            rect = mpatches.Rectangle((x, y), self.size[0], self.size[1], fill=False, edgecolor='blue',
                                                  linewidth=1)
                            ax.add_patch(rect)
                        match[3] = match[3] + 1

            ax.set_axis_off()
        elif int(id) == -1:
            fig, ax = plt.subplots(figsize=(20, 12))

            ax.imshow(image)

            match = [0, 0, 0, 0, 0]
            for y in range(0, len(image) - self.size[1], int(self.size[1] / 4)):
                for x in range(0, len(image[0]) - self.size[0], int(self.size[0] / 4)):
                    match[4] = match[4] + 1
                    coord = [x, x + self.size[0], y, y + self.size[1]]
                    vctr = self.__get_vector_for_image(image, coord)
                    vector = np.zeros((2, len(vctr)))
                    vector[0, :] = vctr

                    out = classifier.predict(vector)
                    if int(out[0]) == 0:
                        match[0] = match[0] + 1
                    elif int(out[0]) == 1:
                        match[1] = match[1] + 1
                    elif int(out[0]) == 2:
                        match[2] = match[2] + 1
                    elif int(out[0]) == 3:
                        rect = mpatches.Rectangle((x, y), self.size[0], self.size[1], fill=False, edgecolor='white',
                                                  linewidth=1)
                        ax.add_patch(rect)
                        match[3] = match[3] + 1
                        tmp = self.checkHit(coord, hit_points, hit_points_cp, 20)
                        if tmp == 1:
                            tp += 1
                        else:
                            fp += 1

        elif int(id) == -2:
            fig, (ax, ax1) = plt.subplots(2, 1, figsize=(10, 12))

            ax.imshow(image)
            ax1.imshow(image)

            match = [0, 0, 0, 0, 0]
            for y in range(0, len(image) - self.size[1], int(self.size[1] / 4)):
                for x in range(0, len(image[0]) - self.size[0], int(self.size[0] / 4)):
                    match[4] = match[4] + 1
                    coord = [x, x + self.size[0], y, y + self.size[1]]

                    vctr = self.__get_vector_for_image(image, coord)
                    vector = np.zeros((2, len(vctr)))
                    vector[0, :] = vctr

                    out = classifier.predict(vector)
                    if int(out[0]) == 0:
                        rect = mpatches.Rectangle((x, y), self.size[0], self.size[1], fill=False,
                                                      edgecolor='yellow',
                                                      linewidth=1)
                        ax.add_patch(rect)
                        match[0] = match[0] + 1
                    elif int(out[0]) == 1:
                        rect = mpatches.Rectangle((x, y), self.size[0], self.size[1], fill=False, edgecolor='green',
                                                      linewidth=1)
                        ax1.add_patch(rect)
                        match[1] = match[1] + 1

            ax.set_axis_off()
            ax1.set_axis_off()
        fn = len(hit_points_cp)
        self.tp += tp
        self.fn += fn
        self.fp += fp

        plt.tight_layout()
        plt.title(label)
        plt.savefig(path + label)
        plt.close()


        values = 'Match: ' + str(match[0]) + ' ' + str(match[1]) + ' ' + str(match[2]) + ' ' + str(match[3]) + ' ' + str(match[4])
        params = "TP: " + str(tp) + " FP: " + str(fp) + " FN: " + str(fn)
        file = open((path + 'data.txt'), 'a')
        file.write((values + ' ' + params + "\n"))
        file.close()
        print(values + ' ' + params)

        return 0

    def get_input_image_cnt(self):
        return len(self.images_prepared)

    def add_images(self, file_list):
        for item in file_list:
            self.images_prepared.append(self.__read_image(item))
            self.roi_list.append(self.__read_image_roi(item + '.txt'))
        print('Imported %d images' % len(self.images_prepared))

    def add_roi(self, roi_list, class_label):
        for item in roi_list:
            self.roi_list.append((self.__read_image_roi(item + '.txt'), item, class_label))

    def cut_images_roi(self, size=(50, 50), vector_length=10, rotate=False):
        self.images_roi.clear()
        self.size = size
        self.vector_length = vector_length
        self.rotate = rotate
        print('Cnt %d input images' % len(self.images_prepared))
        for image in range(0, len(self.images_prepared)):
            if len(self.exclude) > 0:
                for val in self.exclude:
                    if val != image:
                        self.images_roi.extend(self.__cut_roi_image(self.images_prepared[image],
                                                                    self.roi_list[image]))
            else:
                self.images_roi.extend(self.__cut_roi_image(self.images_prepared[image],
                                                            self.roi_list[image]))
        print('Cut %d ROI images' % len(self.images_roi))
        print("ROI types n:")
        for i in range(4):
            print("%d : %d" % (i, self.roi_type_cnt[i]))

    def show_images_original(self, index=-1):
        from skimage import io
        if index != -1:
            io.imshow(self.images_prepared[index])
            io.show()
        else:
            for item in self.images_prepared:
                io.imshow(item)
                io.show()

    def show_images_roi(self, index=-1):
        import matplotlib.pyplot as plt
        plt.close()
        plt.figure(figsize=(12, 8))
        plt.gray()
        plt.grid()
        if index != -1:
            plt.imshow(self.images_roi[index][0])
            plt.title(self.images_roi[index][1])
            plt.show()
            plt.close()

        else:
            for item in self.images_roi:
                plt.imshow(item[0])
                plt.title(item[1])
                plt.show()
                plt.close()

    def set_type(self, type):
        self.type = type

    def set_function(self, fcn, arg):
        self.fcn = fcn
        self.additional_arg = arg
