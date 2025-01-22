import cv2
import fast_image_processing as fip

img = cv2.imread('/home/pkarkeko/projects/sample_images/input_tiger.jpg')

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sobel_x = cv2.Sobel(src=img_gray, ddepth=cv2.CV_16S, dx=1, dy=0, ksize=5)
cv2.imwrite('sobelx_tiger_cv2.jpg', sobel_x)

sobel_y = cv2.Sobel(src=img_gray, ddepth=cv2.CV_16S, dx=0, dy=1, ksize=5)
cv2.imwrite('sobely_tiger_cv2.jpg', sobel_y)

sobel_xy = cv2.Sobel(src=img_gray, ddepth=cv2.CV_16S, dx=1, dy=1, ksize=5)
cv2.imwrite('sobelxy_tiger_cv2.jpg', sobel_xy)


img_fip = fip.Image('/home/pkarkeko/projects/sample_images/input_tiger.jpg')

edge_x = fip.edge_detect(img_fip, 1, 0, 5, fip.Hardware.CPU)
edge_x.save("sobelx_tiger_fip.jpg")

edge_y = fip.edge_detect(img_fip, 0, 1, 5, fip.Hardware.CPU)
edge_y.save("sobely_tiger_fip.jpg")

edge_xy = fip.edge_detect(img_fip, 1, 1, 5, fip.Hardware.CPU)
edge_xy.save("sobelxy_tiger_fip.jpg")







