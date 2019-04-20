import cv2
import numpy as np

np.warnings.filterwarnings('ignore')

def make_coordinates(image, line_parameters):
    try:
        slope, intercept = line_parameters
        height, width = image.shape[:-1]

        y1 = height
        y2 = int(y1*(4/7))
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)

        return np.array([x1, y1, x2, y2])
    except:
        return np.array([0, 0, 0, 0])

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []

    for x1, y1, x2, y2 in lines:
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope, intercept = parameters
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    left_line = make_coordinates(image, left_fit_average)
    right_fit_average = np.average(right_fit, axis=0)
    right_line = make_coordinates(image, right_fit_average)

    return np.array([left_line, right_line])

def remove_horizontal_lines(lines):
    result_lines = []
    min_slope = 0.4

    for line in lines:
        line = line.reshape(4)
        x1, y1, x2, y2 = line
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        if abs(slope) >= min_slope:
            result_lines.append(line)

    return result_lines

def display_lines(image, lines):
    line_image = np.zeros_like(image)

    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 255), 10)

    return line_image

def canny(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    image = cv2.Canny(image, 50, 150)

    return image

def ROI(image):
    offset = 20
    height, width = image.shape
    height -= offset
    width -= offset

    polygons = np.array([
    [(offset, 340),
     (300, 160),
     (width, 340),
     (width, height),
     (offset, height)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    mask = cv2.bitwise_and(image, mask)

    return mask

cap = cv2.VideoCapture('video.avi')
while(cap.isOpened()):
    _, frame = cap.read()
    original_image = np.copy(frame)
    frame = canny(frame)
    frame = ROI(frame)

    lines = cv2.HoughLinesP(frame, 2, np.pi/180, 100, np.array([]),
                           minLineLength=40, maxLineGap=15)
    if lines is not None:
        lines = remove_horizontal_lines(lines)
        lines = average_slope_intercept(original_image, lines)
        line_image = display_lines(original_image, lines)

        result_image = cv2.addWeighted(original_image, 0.6, line_image, 1, 0)
    else:
        result_image = original_image

    cv2.imshow('result', result_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
