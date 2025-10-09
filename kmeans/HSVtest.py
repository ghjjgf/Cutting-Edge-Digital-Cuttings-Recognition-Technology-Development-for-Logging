import cv2

img = "C:\\Users\\28162\\Desktop\\Purple-red-mudstone_37.webp"
img = cv2.imread(img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
window_size = 3
half_window = window_size // 2
height, width, _ = img.shape
hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

print(hsv_img[:,:,0]).flatten()