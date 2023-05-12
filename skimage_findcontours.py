from PIL import ImageDraw
from skimage.measure import approximate_polygon, find_contours
import matplotlib.pyplot as plt
test_path = '/home/jia.chen/worshop/big_model/SAM/bk_cvat_upload_ann/ht/gtFine/default/148_livingroom_livingroom_00021_gtFine_labelIds.png'
test_img = cv2.imread(test_path, cv2.IMREAD_UNCHANGED)
test_img[test_img!=0]=255
contours = find_contours(test_img, 1.0)
rows,cols=test_img.shape
fig, axes = plt.subplots(1,2,figsize=(8,8))
ax0, ax1= axes.ravel()
ax1.axis([0,rows,cols,0])
for n, contour in enumerate(contours):
    ax1.plot(contour[:, 1], contour[:, 0], linewidth=2)
ax1.axis('image')
ax1.set_title('contours')
plt.savefig('test.png')