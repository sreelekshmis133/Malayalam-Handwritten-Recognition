import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
font = cv2.FONT_HERSHEY_SIMPLEX

def read_transparent_png(file):
    """
    Change transparent bg to white
    """
#     print(filename)
#     file = r'C:\Users\Sreelekshmi\Desktop\mini\\'
   
    for i in data:
        
        print(filename)
        image_4channel = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        print(image_4channel)
        print(image_4channel.shape)
        alpha_channel = image_4channel[:,:,3]
        rgb_channels = image_4channel[:,:,:3]

    # White Background Image
        white_background_image = np.ones_like(rgb_channels, dtype=np.uint8) * 255

    # Alpha factor
        alpha_factor = alpha_channel[:,:, np.newaxis].astype(np.float32) / 255.0
        alpha_factor = np.concatenate((alpha_factor, alpha_factor, alpha_factor), axis=2)

    # Transparent Image Rendered on White Background
        base = rgb_channels.astype(np.float32) * alpha_factor
        white = white_background_image.astype(np.float32) * (1 - alpha_factor)
        final_image = base + white
#         print(final_image)
    cv2.imshow('final',final_image)
    cv2.waitKey(0)
#     return final_image.astype(np.uint8)

# read_transparent_png(r'C:\Users\Sreelekshmi\Desktop\mini\\')


def crop_to_sel(file):
    f = open(r'C:\Users\Sreelekshmi\Desktop\name.txt')
    data = f.readlines()
#     p = open(r'C:\Users\Sreelekshmi\Desktop\qwert\\')
    for i in data:
        filename = file + i[0:-1] + '.jpg'
        im = cv2.imread(filename, cv2.IMREAD_COLOR)
    

        img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
        ret, thresh = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY_INV)
#     print(thresh)
#     print(cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
   
        X = []
        Y = []
        Xl = []
        Yl = []
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            X.append(x)
            Y.append(y)
            Xl.append(x+w)
            Yl.append(y+h)
        try:
            smallx = sorted(X)[0]
            Yl.append(y+h)
            largex = sorted(Xl)[-1]
            smally = sorted(Y)[0]
            largey = sorted(Yl)[-1]
            print(im[smally:largey,smallx:largex])
            cv2.imwrite(r"C:\Users\Sreelekshmi\Desktop\qwert\\"+i[0:-1] +".jpg",im[smally:largey,smallx:largex])  
#             return im[smally:largey,smallx:largex]
        except:
            print('a')
            return im

a = r'C:\Users\Sreelekshmi\Desktop\mini\\'

crop_to_sel(a)
# cv2.imshow('test',image)
# cv2.waitKey(0)
    


# count=0.0
# for filename in glob.iglob(r'C:\Users\Sreelekshmi\Desktop\new\*.jpg'):
#     count+=1
# newc=0.0
# for filename in glob.iglob(r'C:\Users\Sreelekshmi\Desktop\new\*.jpg'):
    
# #     imgx = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
#     imgx2=read_transparent_png(filename)
#     print(imgx2.shape)
# # #     print(imgx2)
#     img= crop_to_sel(imgx2)
# #     print(img)
# #     cv2.imshow('sample',img)
# #     cv2.waitKey(0)

#     scale_percent = 50 # percent of original size	
#     width = int(img.shape[1])
#     height = int(img.shape[0])
#     print (width,height)
#     ratio = ((width*1.0)/(height*1.0))
#     width = int(30 * ratio)
#     print (width,height,ratio)
#     dim = (width, 30)
# #     # resize image
#     imout = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
#     cv2.imwrite("qwert\\"+filename[6:-4]+".jpg",imout)
# #     newc+=1
    
# #     print ((newc/count)*100)
#     #cv2.imshow('final', imout)
#     #cv2.waitKey(0)
#     #cv2.destroyAllWindows()


# In[ ]:

