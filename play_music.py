def predict_image(img):
    pred=fishface.predict(img)
    print(pred)
    return pred
    
    
face_cascade=cv2.CascadeClassifier('/home/hana/haarcascade_frontalface_default.xml')
face_cascade2=cv2.CascadeClassifier('/home/hana/haarcascade_frontalface_alt.xml')
cap = cv2.VideoCapture(0)
count=0
 
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
 

while(cap.isOpened()&count<1):
      ret=cap.read()[0]
      img = cap.read()[1]
      if ret == True:
          im_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
          im_gray=cv2.resize(im_gray,(350,350))
          t=predict_image(im_gray)
          print(t)
          if t==0:
            emo="neutral"
          elif t==1:
            emo="angry"
          elif t==5:
            emo="surprised"
          elif t==6:
            emo="sad"
          elif t==7:
            emo="happy"
          faces=face_cascade.detectMultiScale(im_gray,1.1,10)
          for (x,y,w,h) in faces:
                cv2.putText(img,emo, (350,175), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
            
        # Display the resulting frame
          plt.imshow(img)
          plt.xticks([]),plt.yticks([])
          plt.show()
          count=count+1
         # cv2.imwrite("/home/hana/face_test.jpg",img)
 
   
 

cap.release()
import pandas
import subprocess,os
df=pandas.read_excel("/home/hana/song.xlsx")
emotion={}
emotion['neutral']=[x for x in df.neutral.dropna()]
emotion['angry']=[x for x in df.angry.dropna()]
emotion['happy']=[x for x in df.happy.dropna()]
emotion['sad']=[x for x in df.sad.dropna()]
emotion['surprised']=[x for x in df.surprised.dropna()]
def open_music(filename): 
    if sys.platform == "win32":
        os.startfile(filename)
    else:
        opener ="open" if sys.platform == "darwin" else "xdg-open"
        subprocess.call([opener, filename])
actions=[x for x in emotion[emo]]
open_music(actions[0])
