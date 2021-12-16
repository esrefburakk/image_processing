from tkinter import *
import tkinter as tk
from PIL import ImageTk, Image ,ImageFilter
from tkinter import filedialog
import os
from tkinter import ttk
import cv2
import numpy as np
import matplotlib as plt
from matplotlib import pyplot as plt
from scipy.ndimage import zoom
import math

mainWindow = tk.Tk()
mainWindow.geometry("480x350")
mainWindow.resizable(width=True, height=True)

def getFileName():
    filename = filedialog.askopenfilename(title='open')
    return filename
def openImage():
    fname = getFileName()
    image = cv2.imread(filename=fname)
    cv2.imshow("IMAGE", image)
    def First_Process_Window():
        firstProcess = tk.Tk()
        firstProcess.geometry('480x350')
        firstProcess.title('First Process Window')

        fname = tk.StringVar()
        y = tk.StringVar()

        def imageToGray():
            h,w = image.shape[:2]
            gray_image = np.zeros((h,w), np.uint8)
            for i in range(h):
                for j in range(w):
                    gray_image[i,j] = np.clip(0.07 * image[i, j, 0] + 0.72 * image[i, j, 1] + 0.21 * image[i, j, 2], 0, 255)
            
            cv2.imshow("Gray Image", gray_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        def grayToblack():
            h,w = image.shape[:2]
            threshold = int(input("Enter the threshold count: "))
            wb_image = np.zeros((h,w),np.uint8)
            for i in range(h):
                for j in range(w):
                    count = image[i,j][0]+image[i,j][1]+image[i,j][2]
                    if (count/3) < threshold:
                        image[i,j] = 0
                    else:
                        image[i,j] = 255
                        
            cv2.imshow("Black White Image",image)
            cv2.waitKey(0)
            cv2.destroyAllWindows
                    
        def cropImage():
            
            #x,y,w,h = cv2.selectROI(image)
            
            x = int(input("Enter The X"))
            y = int(input("Enter The Y"))
            w = int(input("Enter The W"))
            h = int(input("Enter The H"))
            
            crop_image = image[y:y+h,x:x+w]
            cv2.imshow("Crop_Image",crop_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        def zoomIN_zoomOUT():
            w,h = image.shape[:2]
            
            zoomOut_w = w+50
            zoomOut_h = h+50
            
            zoomIn_w = w-50
            zoomIn_h = h-50
            
            outdim = (zoomOut_h,zoomOut_w)
            indim = (zoomIn_h,zoomIn_w)
            
            zoomOUT_image = cv2.resize(image,outdim,interpolation=cv2.INTER_NEAREST)
            zoomIN_image = cv2.resize(image,indim,interpolation=cv2.INTER_NEAREST)
            
            cv2.imshow("Zoom Out Image",zoomOUT_image)
            cv2.imshow("Zoom In Image",zoomIN_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        def Preprocess_Menu():

            def comboclick(event):
                # myLabel=Label(form, text = myCombo.get()).pack()
                if cbox.get() == 'Image to Gray':
                    imageToGray()
                elif cbox.get() == 'Gray to Black':
                    grayToblack()
                elif cbox.get() == 'ROI':
                    cropImage()
                elif cbox.get() == 'zoomIN-zoomOUT':
                    zoomIN_zoomOUT()
                
                else:
                    myLabel = Label(firstProcess, text=cbox.get()).pack()

            Feature = ['Preprocess Menu',
                       'Image to Gray',
                       'Gray to Black',
                       'ROI',
                       'zoomIN-zoomOUT']

            cbox = ttk.Combobox(firstProcess,value=Feature)
            cbox.current(0)
            cbox.bind("<<ComboboxSelected>>", comboclick)
            cbox.pack()


        radiobutton1 = tk.Radiobutton(firstProcess, text='I want to preprocess', activebackground='green', value='1',variable=fname, command = Preprocess_Menu)
        radiobutton1.pack()

        radiobutton2 = tk.Radiobutton(firstProcess, text='I dont want to preprocess', activebackground='green', value='2',variable=fname)
        radiobutton2.pack()

        button3 = tk.Button(firstProcess, text='Next',fg='black',bg='yellow', height=2, width=13, command=Second_Process_Window)
        button3.pack(side=tk.RIGHT)

        button4 = tk.Button(firstProcess, text='Back',fg='black',bg='yellow',height=2, width=13, command=getFileName)
        button4.pack(side = tk.LEFT)
        
        firstProcess.mainloop()
        
    def Second_Process_Window():
        secondProcess = tk.Tk()
        secondProcess.title("Second Process Menu")
        secondProcess.geometry("480x350")
        fname = tk.StringVar()
        y = tk.StringVar()
        
        def histogramCreate():
            color = ('b', 'g', 'r')
            for i, col in enumerate(color):
                histogram = cv2.calcHist([image], [i], None, [256], [0, 256])
                plt.plot(histogram, color=col)
                plt.xlim([0, 256])
            plt.show()
            
        def imageQuantization():
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    new_value = int(((2**4)*(image[i,j,0]))/(2**8))
                    new_value = int((255*new_value)/(2**4))
                    image[i,j,0] = new_value
                    image[i,j,1] = new_value
                    image[i,j,2] = new_value
            cv2.imshow("Quantization Image",image)
            cv2.waitKey(0)
        
        def Second_Process_Menu():
            Feature = ['Second Process Menu',
                       'Create Histogram',
                       'Image Quantization',]
            def comboclick(event):
                if cbox.get() == 'Create Histogram':
                    histogramCreate()
                elif cbox.get() == 'Image Quantization':
                    imageQuantization()
    
            cbox = ttk.Combobox(secondProcess,value = Feature)
            cbox.current(0)
            cbox.bind("<<ComboboxSelected>>", comboclick)
            cbox.pack()
        
        
        radiobutton = tk.Radiobutton(secondProcess,text='I want to preprocess', activebackground='green', value='1',variable=fname,command = Second_Process_Menu)
        radiobutton.pack()
        radiobutton2 = tk.Radiobutton(secondProcess,text='I dont want to preprocess',activebackground='green', value='2',variable=fname)
        radiobutton2.pack()
        
        button5 = tk.Button(secondProcess,text = 'Next',fg='black',bg='yellow',height=2,width=13,command=Third_Process_Window)
        button5.pack(side=tk.RIGHT)
        
        button6 = tk.Button(secondProcess,text = 'Cancel',fg='black',bg='yellow',height=2,width=13,command=First_Process_Window)
        button6.pack(side=tk.LEFT)
    
    def Third_Process_Window():
        thirdProcess = tk.Tk()
        thirdProcess.title('Filtering Menu')
        thirdProcess.geometry("480x350")
        fname = tk.StringVar()
        y = tk.StringVar()
        
        def gaussianBlurFilter():
            
            gauss = np.random.normal(0,1,image.size)
            gauss = gauss.reshape(image.shape[0],image.shape[1],image.shape[2]).astype('uint8')
            
            gaussian_noisy = cv2.add(image,gauss)
            
            """
            standart_sapma = input("Enter the count")
            mean = 0
            var = 2*math.pi*standart_sapma
            sigma = var**0.5
            gaussian = np.random.normal(mean,sigma,(224,224))
            gaussian_noisy = np.zeros(image.shape,np.float32)
            
            if len(image.shape) == 2:
                gaussian_noisy = image + gaussian
            else:
                gaussian_noisy[:,:,0] = image[:,:,0] + gaussian
                gaussian_noisy[:,:,1] = image[:,:,1] + gaussian
                gaussian_noisy[:,:,2] = image[:,:,2] + gaussian
                
                cv2.normalize(gaussian_noisy,gaussian_noisy,0,255,cv2.NORM_MINMAX,dtype=-1)
                gaussian_noisy = gaussian_noisy.astype(np.uint8)
                
            """
            cv2.imshow("Noisy Image",gaussian_noisy)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        def SharpeningFilter():
            sharpening_filter = np.array([[-1,-1,-1],
                                          [-1,9,-1],
                                          [-1,-1,-1]])
            sharpening_image = cv2.filter2D(image,-1,sharpening_filter)
            cv2.imshow("Sharpening Image",sharpening_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        def findEdgeFilter():
            gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            kernel = np.array([[-1,-1,-1],
                               [-1,8,-1],
                               [-1,-1,-1]])
            
            image_edge = cv2.filter2D(gray,-1,kernel)
            cv2.imshow("Image Edge",image_edge)
            cv2.waitKey(0)
        def meanFilter():
            kernel = np.ones((5,5),np.float32)/25
            meanImage = cv2.filter2D(image,-1,kernel)
            """
            h,w = image.shape
            mask = np.ones([3,3],dtype = int)
            mask = mask/9
            meanImage = np.zeros([h,w])
            
            for i in range (1,h-1):
                for j in range (1,w-1):
                    temp = image[i-1, j-1]*mask[0, 0]+image[i-1, j]*mask[0, 1]+image[i-1, j + 1]*mask[0, 2]+image[i, j-1]*mask[1, 0]+ image[i, j]*mask[1, 1]+image[i, j + 1]*mask[1, 2]+image[i + 1, j-1]*mask[2, 0]+image[i + 1, j]*mask[2, 1]+image[i + 1, j + 1]*mask[2, 2]
                    meanImage[i,j] = temp
            meanImage = meanImage.astype(np.uint8)
            """

            cv2.imshow("Mean Image",meanImage)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
                    
        def medianFilter():

            median_Image = cv2.medianBlur(image,3)
            
            cv2.imshow("Median Image",median_Image)
            cv2.waitKey(0)
            
        def contraHarmonic():
            Q = -1
            num = image ** (Q+1)
            denom = 1/image #image ** Q
            kernel = np.full((3,3),1.0)
            contraImage = cv2.filter2D(num,-1,kernel)/cv2.filter2D(denom,-1,kernel)
            
            cv2.imshow("Contra Harmonic Image",contraImage)
            cv2.waitKey(0)
            
        def filteringMenus():
            Feature = ['Filtering Menu',
                       'Gaussian Noisy',
                       'Image Sharpening',
                       'Image Find Edge',
                       'Median Filter',
                       'Mean Filter',
                       'Contra Harmonic Filter']
            def comboclick(event):
                if cbox.get() == 'Image Sharpening':
                    SharpeningFilter()
                elif cbox.get() == 'Gaussian Noisy':
                    gaussianBlurFilter()
                elif cbox.get() == 'Image Find Edge':
                    findEdgeFilter()
                elif cbox.get() == 'Median Filter':
                    medianFilter()
                elif cbox.get() == 'Mean Filter':
                    meanFilter()
                elif cbox.get() == 'Contra Harmonic Filter':
                    contraHarmonic()
                    
            cbox = ttk.Combobox(thirdProcess,value = Feature)
            cbox.current(0)
            cbox.bind("<<ComboboxSelected>>",comboclick)
            cbox.pack()
        
        radiobutton = tk.Radiobutton(thirdProcess,text='I want to preprocess', activebackground='green', value='1',variable = fname,command = filteringMenus) 
        radiobutton.pack()
        radiobutton2 = tk.Radiobutton(thirdProcess,text='I dont want to preprocess', activebackground='green', value='2',variable = fname)
        radiobutton2.pack()
        
        button7 = tk.Button(thirdProcess,text='Next',fg='black',bg='yellow',height=2,width=13,command=Fourth_Process_Window)
        button7.pack(side = tk.RIGHT)
        
        button8 = tk.Button(thirdProcess,text='Back',fg='black',bg='yellow',height=2,width=13,command=Second_Process_Window)
        button8.pack(side = tk.LEFT)
    
    
    
    def Fourth_Process_Window():
        fourthProcess = tk.Tk()
        fourthProcess.title("Morphological Process")
        fourthProcess.geometry("480x350")
        fname = tk.StringVar()
        y = tk.StringVar()
        
        def Morphological_Process_Menu():
            w,h,l = np.shape(image)
            newImage = np.ones((w,h,l))
            
            for i in range(w):
                for j in range(h):
                    if(image[i,j,0] > 125):
                        newImage[i,j] = 1
                    else:
                        newImage[i,j] = 0
            
            Feature = ['Morphological Process Menu',
                       'Black White Image Extension',
                       'Black White Image Erosion',
                       'Skeletonization']
            
            def ImageExtension():
                kernel = np.ones((5,5),np.uint8)
                image_dilation = cv2.dilate(newImage,kernel,iterations=1)
                cv2.imshow("Extension Image",image_dilation)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            def Erosion():
                kernel = np.ones((5,5),np.uint8)
                image_erosion = cv2.erode(newImage,kernel,iterations=1)
                cv2.imshow("Erosion Image",image_erosion)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            def Skeletonization():
                size = np.size(image)
                skel = np.zeros(image.shape,np.uint8)
                
                ret,img = cv2.threshold(image,127,255,0)
                element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
                while True:
                    open = cv2.morphologyEx(img,cv2.MORPH_OPEN,element)
                    temp = cv2.subtract(img,open)
                    eroded = cv2.erode(img, element)
                    skel = cv2.bitwise_or(skel,temp)
                    img = eroded.copy()
                    if cv2.countNonZero(img) == 0:
                        break
                cv2.imshow("Skeletonization",skel)
                cv2.waitKey(0)
            
            def comboclick(event):
                if cbox.get() == 'Black White Image Extension':
                    ImageExtension()
                elif cbox.get() == 'Black White Image Erosion':
                    Erosion()
                elif cbox.get() == 'Skeletonization':
                    Skeletonization()
            
            cbox = ttk.Combobox(fourthProcess,value = Feature)
            cbox.current(0)
            cbox.bind("<<ComboboxSelected>>",comboclick)
            cbox.pack()
        
        radiobutton = tk.Radiobutton(fourthProcess,text='I want to Morphological Process',activebackground = 'green',value = '1',command = Morphological_Process_Menu)
        radiobutton.pack()
        radiobutton2 = tk.Radiobutton(fourthProcess,text='I dont want to Morphological Process',activebackground = 'green',value = '2')
        radiobutton2.pack()
        
        button9 = tk.Button(fourthProcess,text='Next',fg='black',bg='yellow',height=2,width=13,command=Fifth_Process_Window)
        button9.pack(side = tk.RIGHT)
        button10 = tk.Button(fourthProcess,text='Back',fg='black',bg='yellow',height=2,width=13,command=Third_Process_Window)
        button10.pack(side = tk.LEFT)
        
    button2 = tk.Button(mainWindow, text='Next',fg='black',bg='yellow', height=2, width=13,command=First_Process_Window)
    button2.pack(side=tk.RIGHT)
    
    def Fifth_Process_Window():
        fifthProcess = tk.Tk()
        fifthProcess.title('Save The Image Different Format')
        fifthProcess.geometry("480x350")
        fname = tk.StringVar()
        y = tk.StringVar()
        
        Feature = ['Choice Format',
                   'JPEG',
                   'PNG',
                   'BMP']
        
        def jpeg_save():
            cv2.imwrite('C:/MyPythonWorks/Python/PyQT/HomeWork/testImage/jpeg_format_img.jpeg',image)
            cv2.waitKey(0) 
        def png_save():
            cv2.imwrite('C:/MyPythonWorks/Python/PyQT/HomeWork/testImage/jpeg_format_img.png',image)
            cv2.waitKey(0)
        def bmp_save():
            cv2.imwrite('C:/MyPythonWorks/Python/PyQT/HomeWork/testImage/jpeg_format_img.bmp',image)
            cv2.waitKey(0)
            
        def comboclick(event):
            if cbox.get() == 'JPEG':
                jpeg_save()
            elif cbox.get() == 'PNG':
                png_save()
            elif cbox.get() == 'BMP':
                bmp_save()
        
        cbox = ttk.Combobox(fifthProcess,value = Feature)
        cbox.current(0)
        cbox.bind("<<ComboboxSelected>>", comboclick)
        cbox.pack()
        
button1 = Button(mainWindow, text='Choice Image', fg='black', bg='yellow', height=2, width=13,command=openImage).pack()
mainWindow.mainloop()