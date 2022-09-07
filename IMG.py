
from tkinter import *
from PIL import ImageTk,Image
from tkinter import filedialog 
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

top = Tk()
top.title('IMAGE PROCESSING PROJECT')

f1=LabelFrame(top,text="Image Processing Project [by: Aya Islam]",fg="green",bg="white",pady=2)
f1.grid(row=0,column=0)
f21=LabelFrame(f1,text="Processes",fg="brown",bg="white",pady=90)
f21.grid(row=0,column=0)
f2=LabelFrame(f1,text="Images",fg="brown",bg="white")
f2.grid(row=0,column=1)
f3=LabelFrame(f21,text="",fg="brown",bg="white",padx=137)
f3.grid(row=0,column=0)
frame1 =LabelFrame(f3,width=180, height=100,text="Load image", fg="red",bg="white",padx=20,pady=20)
frame1.grid(row=0,column=0)
frame2 =LabelFrame(f3,width=180, height=100,text="Convert", fg="red",bg="white",padx=20,pady=20)
frame2.grid(row=0,column=1)

frame3 =LabelFrame(f21,text="Point Transform OP's", fg="red",bg="white",padx=120,pady=2)
frame3.grid(row=1,column=0)
frame33 =LabelFrame(frame3,text="Other Operations", fg="yellow",bg="white",padx=2,pady=14)
frame33.grid(row=0,column=1)
frame34 =LabelFrame(frame3,text="Brightness & Contrast", fg="yellow",bg="white",padx=2,pady=3)
frame34.grid(row=0,column=0)
frame4 =LabelFrame(f21,text="Local Transform OP's", fg="red",bg="white",padx=137,pady=10)
frame4.grid(row=2,column=0)
frame5 =LabelFrame(f21,text="Global Transform OP's", fg="red",bg="white",padx=100,pady=10)
frame5.grid(row=3,column=0)


frame6 =LabelFrame(f2,text="Original Image",fg="blue",padx=10,pady=10)
frame6.grid(row=0,column=2)
frame11 =LabelFrame(f2,text="Result",fg="blue",padx=10,pady=10)
frame11.grid(row=2,column=2)
frame12 =LabelFrame(f2,text="after noise adding",fg="blue",padx=10,pady=10)
frame12.grid(row=1,column=2)


canvas1= Canvas(frame6, width=200, height= 200,bg="white")
canvas1.pack()
canvas2= Canvas(frame11, width=200, height= 200,bg="white")
canvas2.pack()
canvas3= Canvas(frame12, width=200, height= 200,bg="white")
canvas3.pack()

def convert(image):
    new= image
    resized_image= new.resize((200,200), Image.ANTIALIAS)
    new_image= ImageTk.PhotoImage(resized_image)
    return new_image

def select_image():
    global im_path,out_img1
    top.filename= filedialog.askopenfilename(initialdir=" D:\ " , title="select an image", filetypes=(("jpg files", ".jpg"),("all files",".*")) )
    im_path=top.filename
    img1 = cv2.imread(im_path)
    cv2.imwrite('img.jpg',img1)
    in_img=(Image.open("img.jpg"))
    resized_image= in_img.resize((200,200), Image.ANTIALIAS)
    out_img1= ImageTk.PhotoImage(resized_image)
    c1=canvas1.create_image(100,100, anchor="center", image = out_img1)
myButton= Button(frame1, text="Open..", command= select_image )
myButton.grid(row=0,column=0)

def read_image():
    img1 = cv2.imread(im_path)
    img = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    return img 
 
def TO_RGB():
    global out_img,new_img
    img1 = cv2.imread(im_path)
    cv2.imwrite('C.jpg',img1)
    in_img=(Image.open("C.jpg"))
    out_img= convert(in_img)
    c2=canvas2.create_image(100,100, anchor="center", image = out_img) 

 
def TO_GRAY ():
    global out_img,new_img
    img_g=read_image()
    new_img = cv2.cvtColor(img_g, cv2.COLOR_RGB2GRAY)
    cv2.imwrite('C.jpg',new_img)
    in_img=(Image.open("C.jpg"))
    out_img= convert(in_img)
    c2=canvas2.create_image(100,100, anchor="center", image = out_img)
    
def converting(color):
    if (color ==1):
        TO_RGB()
    elif (color ==2):
        TO_GRAY()


color= IntVar()
color.set(1)

r1=Radiobutton(frame2,text="RGB", variable= color ,value= 1,command = lambda:converting(color.get()))
r1.grid(row=0,column=0)
r2=Radiobutton(frame2,text="Gray Scale", variable= color ,value= 2,command = lambda:converting(color.get()))
r2.grid(row=0,column=1)

def flip_ver():
    global out_img,new_img
    img1=read_image()
    img = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    new_img = cv2.flip(img, 0)
    cv2.imwrite('C.jpg',new_img)
    in_img=(Image.open("C.jpg"))
    out_img=convert(in_img)
    c2=canvas2.create_image(100,100, anchor="center", image = out_img)
myButton= Button(frame33, text=" Flip Vertically", command= flip_ver ,padx=30)
myButton.grid(row=0,column=0)
 
def flip_hor():
    global out_img,new_img
    img1=read_image()
    img = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    new_img = cv2.flip(img, 1)
    cv2.imwrite('C.jpg',new_img)
    in_img=(Image.open("C.jpg"))
    out_img=convert(in_img)
    c2=canvas2.create_image(100,100, anchor="center", image = out_img)
myButton= Button(frame33, text=" Flip Horizontally", command= flip_hor ,padx=22)
myButton.grid(row=1,column=0)
 
def crop():
    global out_img,new_img
    img1=read_image()
    img = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    upper = 50
    lower = 550
    left = 50
    right = 550
    new_img = img[upper: lower,left:right,:]
    cv2.imwrite('C.jpg',new_img)
    in_img=(Image.open("C.jpg"))
    out_img=convert(in_img)
    c2=canvas2.create_image(100,100, anchor="center", image = out_img)
myButton= Button(frame33, text=" Crop ", command= crop ,padx=51)
myButton.grid(row=2,column=0)
 
def plot_his():
    global out_img,new_img
    img1=read_image()
    img = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    hist = cv2.calcHist([img],[0], None, [256], [0,256])
    intensity_values = np.array([x for x in range(hist.shape[0])])
    plt.bar(intensity_values, hist[:,0], width = 5)
    plt.title("image histogram")
    plt.savefig('fig.jpg')
    in_img=(Image.open("fig.jpg"))
    out_img=convert(in_img)
    c2=canvas2.create_image(100,100, anchor="center", image = out_img)
myButton= Button(frame33, text=" Plot Histogram ", command= plot_his ,padx=24)
myButton.grid(row=3,column=0)
 
label44= LabelFrame(frame34,text="Brightness",fg="blue",padx=10,pady=10)
label44.grid(row=0, column=1)
def brightness_low():
    global out_img,new_img
    img1=read_image()
    img = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    alpha = 1 # Simple contrast control
    beta = -50   # Simple brightness control   
    new_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    cv2.imwrite('C.jpg',new_img)
    in_img=(Image.open("C.jpg"))
    out_img=convert(in_img)
    c2=canvas2.create_image(100,100, anchor="center", image = out_img)


 
def brightness_high():
    global out_img,new_img
    img1=read_image()
    img = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    alpha = 1 # Simple contrast control
    beta = 100   # Simple brightness control   
    new_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    cv2.imwrite('C.jpg',new_img)
    in_img=(Image.open("C.jpg"))
    out_img=convert(in_img)
    c2=canvas2.create_image(100,100, anchor="center", image = out_img)

 
def Bright(brightness):
    if (brightness ==1):
        brightness_low()
    elif (brightness ==2):
        brightness_high()
  
brightness= IntVar()
r11=Radiobutton(label44,text=" Low Brightness", variable= brightness ,value= 1,command = lambda:Bright(brightness.get()))
r11.grid(row=0,column=0)
r22=Radiobutton(label44,text=" High Brightness", variable= brightness ,value= 2,command = lambda:Bright(brightness.get()))
r22.grid(row=1,column=0)
      
def contrast_low():
    global out_img,new_img
    img1=read_image()
    img = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    alpha = .5 # Simple contrast control
    beta = 0   # Simple brightness control   
    new_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    cv2.imwrite('C.jpg',new_img)
    in_img=(Image.open("C.jpg"))
    out_img=convert(in_img)
    c2=canvas2.create_image(100,100, anchor="center", image = out_img)

 
def contrast_high():
    global out_img,new_img
    img1=read_image()
    img = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    alpha = 1.5 # Simple contrast control
    beta = 0   # Simple brightness control   
    new_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    cv2.imwrite('C.jpg',new_img)
    in_img=(Image.open("C.jpg"))
    out_img=convert(in_img)
    c2=canvas2.create_image(100,100, anchor="center", image = out_img)

 
label45= LabelFrame(frame34,text="Contrast",fg="blue",padx=15,pady=10)
label45.grid(row=1, column=1)
def Contrast(cont):
    if (cont ==1):
        contrast_low()
    elif (cont ==2):
        contrast_high()
  
cont= IntVar()
r12=Radiobutton(label45,text=" Low Contrast", variable= cont ,value= 1,command = lambda:Contrast(cont.get()))
r12.grid(row=0,column=0)
r23=Radiobutton(label45,text=" High Contrast", variable= cont ,value= 2,command = lambda:Contrast(cont.get()))
r23.grid(row=1,column=0)

def histogram_equal():
    global out_img,new_img
    img1=read_image()
    img_g = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    new_img = cv2.equalizeHist(img_g)
    cv2.imwrite('C.jpg',new_img)
    in_img=(Image.open("C.jpg"))
    out_img=convert(in_img)
    c2=canvas2.create_image(100,100, anchor="center", image = out_img)
myButton= Button(frame33, text=" Histogram Equalization", command= histogram_equal, padx=4)
myButton.grid(row=4,column=0)
 
def adaptive_thresh ():
    global out_img,new_img
    img1=read_image()
    img_g = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    new_img = cv2.adaptiveThreshold(img_g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    cv2.imwrite('C.jpg',new_img)
    in_img=(Image.open("C.jpg"))
    out_img=convert(in_img)
    c2=canvas2.create_image(100,100, anchor="center", image = out_img)
myButton= Button(frame33, text=" Adaptive Threshold", command= adaptive_thresh , padx=15)
myButton.grid(row=5,column=0)
 
def scaling():
    global out_img,new_img
    img1=read_image()
    img = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    new_img = cv2.resize(img,None,fx=3, fy=1.5, interpolation = cv2.INTER_CUBIC)
    cv2.imwrite('C.jpg',new_img)
    in_img=(Image.open("C.jpg"))
    out_img=convert(in_img)
    c2=canvas2.create_image(100,100, anchor="center", image = out_img)
myButton= Button(frame4, text="Scaling", command= scaling,padx=39 )
myButton.grid(row=0,column=0)
 
def transporting():
    global out_img,new_img
    img1=read_image()
    img = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    rows,cols = img.shape[:2]
    M = np.float32([[1,0,100],[0,1,50]])
    new_img = cv2.warpAffine(img,M,(cols,rows))
    cv2.imwrite('C.jpg',new_img)
    in_img=(Image.open("C.jpg"))
    out_img=convert(in_img)
    c2=canvas2.create_image(100,100, anchor="center", image = out_img)
myButton= Button(frame4, text="Translation", command= transporting ,padx=29)
myButton.grid(row=1,column=0)
 
def rotate():
    global out_img,new_img
    img1=read_image()
    img = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    height, width = img.shape[:2]
    center = (width/2, height/2)
    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=45, scale=1)
    new_img = cv2.warpAffine(src=img, M=rotate_matrix, dsize=(width, height))
    cv2.imwrite('C.jpg',new_img)
    in_img=(Image.open("C.jpg"))
    out_img=convert(in_img)
    c2=canvas2.create_image(100,100, anchor="center", image = out_img)
myButton= Button(frame4, text="Rotation", command= rotate ,padx=35)
myButton.grid(row=2,column=0)
 
def smooth():
    global out_img,new_img,out_img2
    img1=read_image()
    img = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    def add_noise(img):
        row , col = img.shape[:2]
        number_of_pixels = random.randint(300, 10000)
        for i in range(number_of_pixels):
            y_coord=random.randint(0, row - 1)
            x_coord=random.randint(0, col - 1)
            img[y_coord][x_coord] = 255
        number_of_pixels = random.randint(300 , 10000)
        for i in range(number_of_pixels):
            y_coord=random.randint(0, row - 1)
            x_coord=random.randint(0, col - 1)
            img[y_coord][x_coord] = 0 
        return img
    noise=add_noise(img)
    cv2.imwrite('noise.jpg',noise)
    in_img1=(Image.open("noise.jpg"))
    out_img2=convert(in_img1)
    new_img = cv2.blur(img,(10,10))
    cv2.imwrite('C.jpg',new_img)
    in_img=(Image.open("C.jpg"))
    out_img=convert(in_img)
    c2=canvas2.create_image(100,100, anchor="center", image = out_img)
    c3=canvas3.create_image(100,100, anchor="center", image = out_img2)
myButton= Button(frame4, text="Smothing by LPF", command= smooth,padx=13 )
myButton.grid(row=3,column=0)
 
def gaussian():
    global out_img,new_img,out_img2
    img1=read_image() 
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    def add_noise(img):
        row , col = img.shape
        number_of_pixels = random.randint(300, 10000)
        for i in range(number_of_pixels):
            y_coord=random.randint(0, row - 1)
            x_coord=random.randint(0, col - 1)
            img[y_coord][x_coord] = 255
        number_of_pixels = random.randint(300 , 10000)
        for i in range(number_of_pixels):
            y_coord=random.randint(0, row - 1)
            x_coord=random.randint(0, col - 1)
            img[y_coord][x_coord] = 0 
        return img
    noise=add_noise(gray)
    cv2.imwrite('noise.jpg',noise)
    in_img1=(Image.open("noise.jpg"))
    out_img2=convert(in_img1)
    new_img = cv2.GaussianBlur(gray,(3,3),0)
    cv2.imwrite('C.jpg',new_img)
    in_img=(Image.open("C.jpg"))
    out_img=convert(in_img)
    c2=canvas2.create_image(100,100, anchor="center", image = out_img)
    c3=canvas3.create_image(100,100, anchor="center", image = out_img2)
myButton= Button(frame4, text="Gaussian Filter", command= gaussian ,padx=28)
myButton.grid(row=0,column=1)
 
def average_f():
    global out_img,new_img,out_img2
    img1=read_image()
    img = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    def add_noise(img):
        row , col = img.shape[:2]
        number_of_pixels = random.randint(300, 10000)
        for i in range(number_of_pixels):
            y_coord=random.randint(0, row - 1)
            x_coord=random.randint(0, col - 1)
            img[y_coord][x_coord] = 255
        number_of_pixels = random.randint(300 , 10000)
        for i in range(number_of_pixels):
            y_coord=random.randint(0, row - 1)
            x_coord=random.randint(0, col - 1)
            img[y_coord][x_coord] = 0 
        return img
    noise=add_noise(img)
    cv2.imwrite('noise.jpg',noise)
    in_img1=(Image.open("noise.jpg"))
    out_img2=convert(in_img1)
    kernel = np.ones((5,5),np.float32)/20
    new_img = cv2.filter2D(img,-1,kernel)
    cv2.imwrite('C.jpg',new_img)
    in_img=(Image.open("C.jpg"))
    out_img=convert(in_img)
    c2=canvas2.create_image(100,100, anchor="center", image = out_img)
    c3=canvas3.create_image(100,100, anchor="center", image = out_img2)
myButton= Button(frame4, text="Averageing Filter ", command= average_f ,padx=20)
myButton.grid(row=1,column=1)
 
def sobel_f():
    global out_img,new_img
    img1=read_image()
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    def add_noise(img):
        row , col = img.shape[:2]
        number_of_pixels = random.randint(300, 10000)
        for i in range(number_of_pixels):
            y_coord=random.randint(0, row - 1)
            x_coord=random.randint(0, col - 1)
            img[y_coord][x_coord] = 255
        number_of_pixels = random.randint(300 , 10000)
        for i in range(number_of_pixels):
            y_coord=random.randint(0, row - 1)
            x_coord=random.randint(0, col - 1)
            img[y_coord][x_coord] = 0 
        return img
    noise=add_noise(gray)
    cv2.imwrite('noise.jpg',noise)
    in_img1=(Image.open("noise.jpg"))
    out_img2=convert(in_img1)
    img_gaussian = cv2.GaussianBlur(gray,(3,3),0)
    img_sobelx = cv2.Sobel(img_gaussian,cv2.CV_8U,1,0,ksize=5)
    img_sobely = cv2.Sobel(img_gaussian,cv2.CV_8U,0,1,ksize=5)
    new_img = img_sobelx + img_sobely
    cv2.imwrite('C.jpg',new_img)
    in_img=(Image.open("C.jpg"))
    out_img=convert(in_img)
    c2=canvas2.create_image(100,100, anchor="center", image = out_img)
    c3=canvas3.create_image(100,100, anchor="center", image = out_img2)
myButton= Button(frame4, text="Sobel Edge Detection", command= sobel_f,padx=10 )
myButton.grid(row=2,column=1)
 
def median():
    global out_img,new_img
    img1=read_image()
    img = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    def add_noise(img):
        row , col = img.shape[:2]
        number_of_pixels = random.randint(300, 10000)
        for i in range(number_of_pixels):
            y_coord=random.randint(0, row - 1)
            x_coord=random.randint(0, col - 1)
            img[y_coord][x_coord] = 255
        number_of_pixels = random.randint(300 , 10000)
        for i in range(number_of_pixels):
            y_coord=random.randint(0, row - 1)
            x_coord=random.randint(0, col - 1)
            img[y_coord][x_coord] = 0 
        return img
    noise=add_noise(img)
    cv2.imwrite('noise.jpg',noise)
    in_img1=(Image.open("noise.jpg"))
    out_img2=convert(in_img1)
    new_img = cv2.medianBlur(img,5)
    cv2.imwrite('C.jpg',new_img)
    in_img=(Image.open("C.jpg"))
    out_img=convert(in_img)
    c2=canvas2.create_image(100,100, anchor="center", image = out_img)
    c3=canvas3.create_image(100,100, anchor="center", image = out_img2)
myButton= Button(frame4, text="Median Filtering", command= median ,padx=23)
myButton.grid(row=3,column=1)
 
def erosion():
    global out_img,new_img
    img1=read_image()
    img = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    kernel = np.ones((5,5), np.uint8) 
    new_img = cv2.erode(img, kernel, iterations=1)
    cv2.imwrite('C.jpg',new_img)
    in_img=(Image.open("C.jpg"))
    out_img=convert(in_img)
    c2=canvas2.create_image(100,100, anchor="center", image = out_img)
myButton= Button(frame5, text="Erosion ", command= erosion,padx=60)
myButton.grid(row=0,column=0)
 
def dilation():
    global out_img ,new_img
    img1=read_image()
    img = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    kernel = np.ones((5,5), np.uint8) 
    new_img = cv2.dilate(img, kernel, iterations=1)
    cv2.imwrite('C.jpg',new_img)
    in_img=(Image.open("C.jpg"))
    out_img=convert(in_img)
    c2=canvas2.create_image(100,100, anchor="center", image = out_img)
myButton= Button(frame5, text="Dilation ", command= dilation,padx=60 )
myButton.grid(row=0,column=1)
 

def save_resault():
    cv2.imwrite("resault image.jpg",new_img)    
myButton= Button(f21, text=" Save" ,command= save_resault,padx=25)
myButton.grid(row=4,column=0)
 
 
top.mainloop()