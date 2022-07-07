from scipy.cluster.vq import kmeans
from scipy.cluster.vq import whiten
from matplotlib import pyplot as plt
from matplotlib import image as img
import argparse
import cv2
import numpy as np
import pandas as pd
import sys 

#Get the colour name -- KNN        
def GetColour(r,g,b):
    min_d=sys.maxsize
    for i in range(len(colours_set)):
        d =abs(r-int(colours_set.loc[i,"R"]))+abs(g-int(colours_set.loc[i,"G"]))+abs(b-int(colours_set.loc[i,"B"]))
        if min_d>d:
            min_d=d
            colour_name = colours_set.loc[i,"colour name"]
    return colour_name
    
#Show the colour on click        
def GetPoint(event,x,y,flags,params):
    img=cv2.imread(img_path,1)
    if event == cv2.EVENT_LBUTTONDOWN:
        xpos=x
        ypos=y
        rgb=img[y,x]
        b=int(rgb[0])
        g=int(rgb[1])
        r=int(rgb[2])
        cv2.rectangle(img,(20,20),(725,60),(b,g,r), -1)
        text = GetColour(r,g,b) +' R='+ str(r)+' G='+ str(g)+' B='+ str(b)
        text_colour=(255,255,255)
        if(r+b+g)>600:
            text_colour=(0,0,0)
        cv2.putText(img,text,(50,50),2,0.8,text_colour,2,cv2.LINE_AA)
        cv2.imshow('image',img)
    if event==cv2.EVENT_RBUTTONDOWN:
        cv2.imshow('image', img)


#construct to a dataframe for future data process
def GetDataframe(k=5):
    df = pd.DataFrame()
    df['r']=pd.Series(img[:,:,0].flatten())
    df['g']=pd.Series(img[:,:,1].flatten())
    df['b']=pd.Series(img[:,:,2].flatten())
    df['r_whiten'] = whiten(df['r'])
    df['g_whiten'] = whiten(df['g'])
    df['b_whiten'] = whiten(df['b'])
    cluster_centers,distortions = kmeans(df[['r_whiten', 'g_whiten', 'b_whiten']], k)
    r_std, g_std, b_std = df[['r', 'g', 'b']].std()
    colors=[]
    for color in cluster_centers:
        sr, sg, sb = color
        colors.append((int(sr*r_std), int(sg*g_std), int(sb*b_std)))
    #print(df[['r_whiten', 'g_whiten', 'b_whiten']])
    plt.imshow([colors])    
    plt.show()


##main
        
#get the image from the user in cmd and read 
ap=argparse.ArgumentParser()
ap.add_argument('-i','--image',required=True,help="Image Path")
args=vars(ap.parse_args())
img_path=args['image']
img=cv2.imread(img_path,1)


#load the dataset
colours_set=pd.read_csv('colours.csv')
num_cluster=int(input("Enter the number of colours required in palette: "))

#create a mouse event
cv2.imshow("image",img)
cv2.namedWindow('image')
cv2.setMouseCallback('image',GetPoint)
GetDataframe(num_cluster)

#close window        
cv2.waitKey(0)
cv2.destroyAllWindows()
