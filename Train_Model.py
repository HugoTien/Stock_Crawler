
# coding: utf-8

# In[1]:

# 此method用於優化、切割從網站下載的驗證碼
# 若執行有問題，可能與opencv support python版本有關，參照以下修正
# Download OpenCV package which support python 3's
# Link : https://www.lfd.uci.edu/~gohlke/pythonlibs/
# 下載opencv_python-3.4.0-cp36-cp36m-win_amd64.whl版本

# def split_stock_letter():
    
#     !pip install wheel
#     !pip install C:/Users/Hao-Ping/Desktop/stock_crawler/opencv_python-3.4.0-cp36-cp36m-win_amd64.whl
#     !pip install matplotlib
#     !pip install matplotlib
    
#     import matplotlib.pyplot as plt
#     import cv2
#     import numpy as np
    
#     verification_image = cv2.imread("C:/Users/Hao-Ping/Desktop/stock_crawler/CaptchaImage.jpg")
#     plt.axis("off")
#     plt.imshow(verification_image)
#     plt.show()

#     # 超過440外沒有其他東西，視為無顏色
#     kernel = np.ones((4,5),np.uint8)
#     erosion = cv2.erode(verification_image, kernel, iterations = 1)
#     plt.imshow(erosion)
#     plt.show()

#     # 模糊(淡化奇怪的點)
#     blurred = cv2.GaussianBlur(erosion, (5,5),0)
#     plt.imshow(blurred)
#     plt.show()

#     # 透過Canny演算法找出邊界
#     canny = cv2.Canny(blurred, 30,150)
#     plt.imshow(canny)
#     plt.show()

#     # 膨脹
#     dilation = cv2.dilate(canny, kernel, iterations = 1)
#     plt.imshow(dilation)
#     plt.show()

#     # 偵測輪廓
#     image, contours, hierarchy = cv2.findContours(dilation.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#     # BoundingRect，找出邊界x,y,寬,高
#     cnts = sorted([(c,cv2.boundingRect(c)[0])  for c in contours], key = lambda x:x[1])

#     array = []
#     for(c, _) in cnts:
#         (x,y,w,h)=cv2.boundingRect(c)
#         # 判斷寬高超過15才算一個字(不然會切太多)
#         if w > 15 and h > 15:
#             array.append((x,y,w,h))

#     # 產生個別圖案並存檔
#     fig = plt.figure()
#     for id, (x,y,w,h) in enumerate(array):
#         roi = dilation[y:y+h, x:x+w]
#         thresh = roi.copy()
#         a = fig.add_subplot(1, len(array), id+1)
#         # 切成同樣大小(50,50)
#         res = cv2.resize(thresh, (50,50))
#         plt.imshow(res)
#         plt.show()
#         # 存檔
#         cv2.imwrite("{}.png".format(id),res)


# In[2]:

# 此method用於切割經由wordpress外掛生成的驗證碼，並建立多個字母/數字資料夾，將其分別匯入，用於訓練model用
# 執行此method注意事項:
# 此script路徑需與 generated_captcha_images, extracted_letter_images 一致
# open cv問題，請參照前述debug
# 記得pip install imutils
# 此script主要來源於 https://medium.com/@ageitgey/how-to-break-a-captcha-system-in-15-minutes-with-machine-learning-dbebb035a710
# 圖片檔案及code請參考 https://s3-us-west-2.amazonaws.com/mlif-example-code/solving_captchas_code_examples.zip
# 此網站內附使用wordpress產生的大量驗證碼，無需自行生產

def general_single_letter():
    get_ipython().system('pip install imutils')
    import os
    import os.path
    import cv2
    import glob 
    # 文件搜索用
    import imutils


    CAPTCHA_IMAGE_FOLDER = "generated_captcha_images"
    OUTPUT_FOLDER = "extracted_letter_images"


    # Get a list of all the captcha images we need to process
    captcha_image_files = glob.glob(os.path.join(CAPTCHA_IMAGE_FOLDER, "*"))
    counts = {}

    # loop over the image paths
    for (i, captcha_image_file) in enumerate(captcha_image_files):
        print("[INFO] processing image {}/{}".format(i + 1, len(captcha_image_files)))

        # Since the filename contains the captcha text (i.e. "2A2X.png" has the text "2A2X"),
        # grab the base filename as the text
        filename = os.path.basename(captcha_image_file)
        captcha_correct_text = os.path.splitext(filename)[0]

        # Load the image and convert it to grayscale
        image = cv2.imread(captcha_image_file)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Add some extra padding around the image
        gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)

        # threshold the image (convert it to pure black and white)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        # find the contours (continuous blobs of pixels) the image
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Hack for compatibility with different OpenCV versions
        contours = contours[0] if imutils.is_cv2() else contours[1]

        letter_image_regions = []

        # Now we can loop through each of the four contours and extract the letter
        # inside of each one
        for contour in contours:
            # Get the rectangle that contains the contour
            (x, y, w, h) = cv2.boundingRect(contour)

            # Compare the width and height of the contour to detect letters that
            # are conjoined into one chunk
            if w / h > 1.25:
                # This contour is too wide to be a single letter!
                # Split it in half into two letter regions!
                half_width = int(w / 2)
                letter_image_regions.append((x, y, half_width, h))
                letter_image_regions.append((x + half_width, y, half_width, h))
            else:
                # This is a normal letter by itself
                letter_image_regions.append((x, y, w, h))

        # If we found more or less than 4 letters in the captcha, our letter extraction
        # didn't work correcly. Skip the image instead of saving bad training data!
        if len(letter_image_regions) != 4:
            continue

        # Sort the detected letter images based on the x coordinate to make sure
        # we are processing them from left-to-right so we match the right image
        # with the right letter
        letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

        # Save out each letter as a single image
        for letter_bounding_box, letter_text in zip(letter_image_regions, captcha_correct_text):
            # Grab the coordinates of the letter in the image
            x, y, w, h = letter_bounding_box

            # Extract the letter from the original image with a 2-pixel margin around the edge
            letter_image = gray[y - 2:y + h + 2, x - 2:x + w + 2]

            # Get the folder to save the image in
            save_path = os.path.join(OUTPUT_FOLDER, letter_text)

            # if the output directory does not exist, create it
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # write the letter image to a file
            count = counts.get(letter_text, 1)
            p = os.path.join(save_path, "{}.png".format(str(count).zfill(6)))
            cv2.imwrite(p, letter_image)

            # increment the count for the current key
            counts[letter_text] = count + 1


# In[3]:

# 用於resize
def resize_to_fit(image, width, height):
    """
    A helper function to resize an image to fit within a given size
    :param image: image to resize
    :param width: desired width in pixels
    :param height: desired height in pixels
    :return: the resized image
    """
    import imutils
    import cv2

    # grab the dimensions of the image, then initialize
    # the padding values
    (h, w) = image.shape[:2]

    # if the width is greater than the height then resize along
    # the width
    if w > h:
        image = imutils.resize(image, width=width)

    # otherwise, the height is greater than the width so resize
    # along the height
    else:
        image = imutils.resize(image, height=height)

    # determine the padding values for the width and height to
    # obtain the target dimensions
    padW = int((width - image.shape[1]) / 2.0)
    padH = int((height - image.shape[0]) / 2.0)

    # pad the image then apply one more resizing to handle any
    # rounding issues
    image = cv2.copyMakeBorder(image, padH, padH, padW, padW,
        cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (width, height))

    # return the pre-processed image
    return image


# In[31]:

# 建立類神經網路
def train_model():
    get_ipython().system('pip install keras')
    get_ipython().system('pip install tensorflow')
    get_ipython().system('pip install imutils')
    import imutils
    import cv2
    import pickle
    import os.path
    import numpy as np
    from imutils import paths
    from sklearn.preprocessing import LabelBinarizer
    from sklearn.model_selection import train_test_split
    from keras.models import Sequential
    from keras.layers.convolutional import Conv2D, MaxPooling2D
    from keras.layers.core import Flatten, Dense


    LETTER_IMAGES_FOLDER = "extracted_letter_images"
    MODEL_FILENAME = "captcha_model.hdf5"
    MODEL_LABELS_FILENAME = "model_labels.dat"


    # initialize the data and labels
    data = []
    labels = []

    # loop over the input images
    for image_file in paths.list_images(LETTER_IMAGES_FOLDER):
        # Load the image and convert it to grayscale
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize the letter so it fits in a 20x20 pixel box
        image = resize_to_fit(image, 20, 20)

        # Add a third channel dimension to the image to make Keras happy
        image = np.expand_dims(image, axis=2)

        # Grab the name of the letter based on the folder it was in
        label = image_file.split(os.path.sep)[-2]

        # Add the letter image and it's label to our training data
        data.append(image)
        labels.append(label)


    # scale the raw pixel intensities to the range [0, 1] (this improves training)
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    # Split the training data into separate train and test sets
    (X_train, X_test, Y_train, Y_test) = train_test_split(data, labels, test_size=0.25, random_state=0)

    # Convert the labels (letters) into one-hot encodings that Keras can work with
    lb = LabelBinarizer().fit(Y_train)
    Y_train = lb.transform(Y_train)
    Y_test = lb.transform(Y_test)

    # Save the mapping from labels to one-hot encodings.
    # We'll need this later when we use the model to decode what it's predictions mean
    with open(MODEL_LABELS_FILENAME, "wb") as f:
        pickle.dump(lb, f)

    # Build the neural network!
    model = Sequential()

    # 卷基層，15個5*5的filter
    model.add(Conv2D(15, (5, 5), padding="same", input_shape=(20, 20, 1), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Second convolutional layer with max pooling
    model.add(Conv2D(30, (5, 5), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # 隱藏層，256個node
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))

    # 32種可能
    model.add(Dense(32, activation="softmax"))

    # Ask Keras to build the TensorFlow model behind the scenes
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    # epoch2即可到達99%，每次訓練128張
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=128, epochs=2, verbose=1)

    # Save the trained model to disk
    model.save(MODEL_FILENAME)


# In[32]:

general_single_letter()
train_model()


# In[ ]:



