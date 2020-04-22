import cv2
import numpy as np
import pytesseract
import tempfile
from PIL import Image


def convertingPintToTheFourDimensionalArray(pts):
    rect = [[0, 0], [0, 0], [0, 0], [0, 0]]
    rect[0][0] = (pts[0][0][0])
    rect[0][1] = (pts[0][0][1])

    rect[1][0] = (pts[1][0][0])
    rect[1][1] = (pts[1][0][1])

    rect[2][0] = (pts[2][0][0])
    rect[2][1] = (pts[2][0][1])

    rect[3][0] = (pts[3][0][0])
    rect[3][1] = (pts[3][0][1])
    return rect


def order_points(pts):
    pts = convertingPintToTheFourDimensionalArray(pts)
    rect = np.zeros((4, 2), dtype="float32")
    for j in range(4):
        for i in range(4):
            if pts[i][0] >= pts[j][0]:
                pts[i], pts[j] = pts[j], pts[i]
    if pts[1][1] <= pts[0][1]:
        rect[0] = pts[1]
        rect[3] = pts[0]
    else:
        rect[0] = pts[0]
        rect[3] = pts[1]
    pts = np.delete(pts, 0, 0)
    pts = np.delete(pts, 0, 0)
    if pts[1][1] <= pts[0][1]:
        rect[1] = pts[1]
        rect[2] = pts[0]
    else:
        rect[1] = pts[0]
        rect[2] = pts[1]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def findingTotalCard(testingPath,modelPath):
    numberOfModelCard = 22
    testImage = cv2.imread(testingPath)
    testImage = cv2.resize(testImage,(800,600))
    cv2.imshow("original image", testImage)
    cv2.waitKey(0)
    testingImage = cv2.imread(testingPath)
    for i in range(numberOfModelCard):
        x , image = findingCardPosition(testingPath,modelPath+str(i)+".jpg")
        if x==1:
            cardImage = image
            return cardImage

def findingTotalCardNumber(testingImage,modelPath):
    find =0
    cardImage = testingImage
    numberOfCardNumberPicture=18
    imager=cv2.imread(modelPath+str(0)+".jpg")
    for i in range(numberOfCardNumberPicture):
        x,image = findingCardNumbersPosition(testingImage,modelPath+str(i)+".jpg")
        if x == 1:
            find=1
            cardImage = image
            break
    return cardImage,find
def findingCardPosition(testingPath,modelPath):
    testingImage = cv2.imread(testingPath)
    img = cv2.imread(modelPath, cv2.IMREAD_GRAYSCALE)  # queryiamge
    sift = cv2.xfeatures2d.SIFT_create()
    kp_image, desc_image = sift.detectAndCompute(img, None)

    index_params = dict(algorithm=0, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    frame = testingImage

    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # trainimage
    kp_grayframe, desc_grayframe = sift.detectAndCompute(grayframe, None)
    matches = flann.knnMatch(desc_image, desc_grayframe, k=2)
    good_points = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good_points.append(m)

    # Homography
    if len(good_points) > 12:
        query_pts = np.float32([kp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
        train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
        matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()
        h, w = img.shape
        pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, matrix)
        homography = cv2.polylines(frame, [np.int32(dst)], True, (0, 0, 255), 1)
        croped_image = four_point_transform(frame,np.int32(dst))
        cv2.destroyAllWindows()
        return 1,croped_image
    else:
        print("can't find card position")
        return 2,frame
    cv2.destroyAllWindows()
def findingCardNumbersPosition(testingImage,modelPath):
    img = cv2.imread(modelPath, cv2.IMREAD_GRAYSCALE)  # queryiamge
    # Features
    sift = cv2.xfeatures2d.SIFT_create()
    kp_image, desc_image = sift.detectAndCompute(img, None)
    # Feature matching
    index_params = dict(algorithm=0, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    frame = testingImage
    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # trainimage
    kp_grayframe, desc_grayframe = sift.detectAndCompute(grayframe, None)
    matches = flann.knnMatch(desc_image, desc_grayframe, k=2)
    good_points = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good_points.append(m)

    if len(good_points) > 10:
        query_pts = np.float32([kp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
        train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
        matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()

        h, w = img.shape
        pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, matrix)
        hq, wq, e = frame.shape
        croped_image = four_point_transform(frame,np.int32(dst))
        cv2.destroyAllWindows()
        return 1, croped_image
    else:
        print("can't find card number position")
        return 2, frame
    cv2.destroyAllWindows()

def process_image_for_ocr(file_path):
    # TODO : Implement using opencv
    temp_filename = set_image_dpi(file_path)
    im_new = remove_noise_and_smooth(temp_filename)
    im_new = clearNoise(im_new)
    return im_new

def set_image_dpi(file_path):
    IMAGE_SIZE = 1800
    BINARY_THREHOLD = 180
    im = Image.open(file_path)
    length_x, width_y = im.size
    factor = max(1, int(IMAGE_SIZE / length_x))
    size = factor * length_x, factor * width_y
    im_resized = im.resize(size, Image.ANTIALIAS)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    temp_filename = temp_file.name
    im_resized.save(temp_filename, dpi=(300, 300))
    return temp_filename

def image_smoothening(img):
    BINARY_THREHOLD = 180
    ret1, th1 = cv2.threshold(img, BINARY_THREHOLD, 255, cv2.THRESH_BINARY)
    ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blur = cv2.GaussianBlur(th2, (1, 1), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th3

def remove_noise_and_smooth(file_name):
    img = cv2.imread(file_name, 0)
    filtered = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41,
                                     3)
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    img = image_smoothening(img)
    or_image = cv2.bitwise_or(img, closing)
    return or_image
def clearNoise(image):
    kernel = np.ones((5,5),np.uint8)
    image = cv2.dilate(image,kernel,iterations = 1)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    for i in range(30):
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    image = cv2.erode(image,kernel,iterations = 1)
    image = cv2.dilate(image,kernel,iterations = 1)
    image = cv2.dilate(image,kernel,iterations = 1)
    for i in range(100):
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    image = cv2.erode(image,kernel,iterations = 1)
    image = cv2.erode(image,kernel,iterations = 1)
    return image

def ocr(image):
    image = image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Image", gray)

    gray = cv2.medianBlur(gray, 3)

    filename = "newImage.jpg"
    cv2.imwrite(filename, gray)

    text = pytesseract.image_to_string(Image.open(filename), lang='neng',config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
    print(text)
    cv2.imshow("Output", gray)
    cv2.waitKey(0)
    return text


font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
fontScale = 2
color = (0, 50, 255)
thickness = 5
modelCardPicturesFolderPath = "CreditCardOcrWithOpencv\\modelCardPicture\\"
modelCardNumberPicturesFolderPath = "CreditCardOcrWithOpencv\\modelCardNumberPicture\\"
testingImagePath = "type your path of image"
try:
    images = findingTotalCard(testingImagePath,modelCardPicturesFolderPath)
    try:
        image,find = findingTotalCardNumber(images,modelCardNumberPicturesFolderPath)
        if find==1:
            w,h,c = image.shape
            image = cv2.resize(image, (int(h/2), int(w/2)))
            path = "testing picture\\newImage.jpg"
            cardNumber = pytesseract.image_to_string(image, lang='neng',config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
            org = (20,300)
            print("cardNumber",cardNumber)
            images = cv2.resize(images, (800, 600))
            images = cv2.putText(images, cardNumber, org, font,fontScale, color, thickness, cv2.LINE_AA)

            cv2.imshow("croppedImage with cardNumber",images)
            cv2.waitKey(0)
            print("cardNumber",cardNumber)
            cv2.imshow("numberPartImage",image)
            cv2.waitKey(0)
        else:
            print("can't find the cardNumber")
    except cv2.error:
        print("can't find the position of cardNumber")
except cv2.error:
    print("can't find the card")

cv2.destroyAllWindows()
