import pylab as pl
from TP2.roipoly import roipoly

# create image
img = pl.ones((100, 100)) * range(0, 100)

# show the image
pl.imshow(img, interpolation='nearest', cmap="Greys")
pl.colorbar()
pl.title("left click: line segment         right click: close region")

# let user draw first ROI
ROI1 = roipoly(roicolor='r') #let user draw first ROI

# show the image with the first ROI
pl.imshow(img, interpolation='nearest', cmap="Greys")
pl.colorbar()
ROI1.displayROI()



# show the image with both ROIs and their mean values
pl.imshow(img, interpolation='nearest', cmap="Greys")
pl.colorbar()
[x.displayROI() for x in [ROI1]]
[x.displayMean(img) for x in [ROI1]]
pl.title('The ROI')
pl.show()

# show ROI masks
pl.imshow(ROI1.getMask(img),
          interpolation='nearest', cmap="Greys")
pl.title('ROI mask')
pl.show()