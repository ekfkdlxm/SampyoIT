import numpy as np
import cv2
import math
import pymssql
import time
import datetime
import tensorflow as tf
import multiprocessing

import Object_detection_image as objectDetection
#  데이터 베이스 연결

conn= pymssql.connect(host='192.168.4.31', user = 'sa' ,password= 'auto',database = 'PCS',  as_dict=True)

#gpu 메모리 해결책

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)


#  ip카메라 정보
ipAddress = "rtsp://admin:sampyo!1@10.62.67.226:554/profile2/media.smp"

cursor = conn.cursor()
cursor2 = conn.cursor()

#  쿼리 테스트
#cursor.execute("INSERT INTO [test1].[dbo].[PCS_PRODUCT_COUNT] (PLANT_CD,BPNO,ITEM_CD,PDATE,LINE_NO,PTIME,PMINUTE) VALUES ('I05','B','ABCD','2008',%d,'1254','1424')",(mortarCount+10))
#conn.commit()



#  영상 실행
cap = cv2.VideoCapture(ipAddress)
#cap = cv2.VideoCapture("7.mov")
i = 0

# RGB색상 상수
SCALAR_BLACK = (0,0,0)
SCALAR_WHITE = (255,255,255)
SCALAR_BLUE = (255,0,0)
SCALAR_GREEN = (0,200,0)
SCALAR_RED = (0,0,255)



global blobs
blobs= []

# 몰탈 count
mortarCount=np.zeros((6),np.int64)
# 몰탈 DB정보 업데이트용 TEMP
mortarTemp=np.zeros((6),np.int64)
#class DB정보 업데이트용 TEMP
classesTemp = '01'

#  첫프레임 read
_, imgInput = cap.read()
_, imgPre = cap.read()

#  첫 objectDetection Test
image_expanded = np.expand_dims(imgInput, axis=0)
objectDetection.mortarClassification(imgInput,image_expanded)

# 관심영역 Roi 지정


passLine = False

firstFrame = True

#  객체의 정보를 담고있는 blob class
class blob :

    def __init__(self, contour):

        global centerPosition #blob의 center x,y좌표
        global centerPositions
        global boundingRect #사격형 표현 x,y,w,h 의 멤버변수가짐
        global nextPosition
        global diagonal   #사각형의 기울기
        global aspectRatio # 사각형의 w,h 비율
        global blobTracked # blob이 Traking되고 있는지 판단하는 bool
        global matchNewBlob
        global numMatch
        global Contour
        global cx, cy
        global frameBlob

        self.centerPositions = []
        self.centerPosition = []
        self.boundingRect = []
        self.predictedNextPosition=[]
        self.Contour = []
        x,y,w,h = cv2.boundingRect(contour)
        self.boundingRect=[x,y,w,h]
        cx=(2*x+w)/2
        cy=(2*y+h)/2
        self.centerPosition = [cx, cy]
        self.diagonal = math.sqrt(w * w + h * h)
        self.aspectRatio = (w / (h * 1.0))
        self.blobTracked = True
        self.matchNewBlob = True
        self.numMatch = 0
        self.centerPositions.append(self.centerPosition)



# 함 수 부 분
##################################################################################
# 객체의 다음 포지션을 예측하는 함수
    def predictNextPosition(self):
            numPositions = len(self.centerPositions)

            if (numPositions == 1):
                self.predictedNextPosition = [self.centerPositions[-1][-2], self.centerPositions[-1][-1]]
            if (numPositions >= 2):
                deltaX = self.centerPositions[1][0] - self.centerPositions[0][0]
                deltaY = self.centerPositions[1][1] - self.centerPositions[0][1]
                self.predictedNextPosition = [self.centerPositions[-1][-2] + deltaX,
                                              self.centerPositions[-1][-1] + deltaY]
            if (numPositions == 3):
                sumOfXChanges= ((self.centerPositions[2][0] - self.centerPositions[1][0]) * 2) +((self.centerPositions[1][0] - self.centerPositions[0][0]) * 1)
                deltaX=(sumOfXChanges / 3)
                sumOfYChanges= ((self.centerPositions[2][1] - self.centerPositions[1][1]) * 2) +((self.centerPositions[1][1] - self.centerPositions[0][1]) * 1)
                deltaY=(sumOfYChanges / 3)
                self.predictedNextPosition=[self.centerPositions[-1][-2]+deltaX,self.centerPositions[-1][-1]+deltaY ]
            if (numPositions == 4):
                sumOfXChanges= ((self.centerPositions[3][0] - self.centerPositions[2][0]) * 3) +((self.centerPositions[2][0] - self.centerPositions[1][0]) * 2) +((self.centerPositions[1][0] - self.centerPositions[0][0]) * 1)
                deltaX=(sumOfXChanges / 6)
                sumOfYChanges= ((self.centerPositions[3][1] - self.centerPositions[2][1]) * 3) +((self.centerPositions[2][1] - self.centerPositions[1][1]) * 2) +((self.centerPositions[1][1] - self.centerPositions[0][1]) * 1)
                deltaY= (sumOfYChanges / 6)
                self.predictedNextPosition=[self.centerPositions[-1][-2]+deltaX,self.centerPositions[-1][-1]+deltaY ]
            if (numPositions >= 5):
                sumOfXChanges= ((self.centerPositions[numPositions-1][0] - self.centerPositions[numPositions-2][0]) * 4) +((self.centerPositions[numPositions-2][0] - self.centerPositions[numPositions-3][0]) * 3) +((self.centerPositions[numPositions-3][0] - self.centerPositions[numPositions-4][0]) * 2) +((self.centerPositions[numPositions-4][0] - self.centerPositions[numPositions-5][0]) * 1)
                sumOfYChanges= ((self.centerPositions[numPositions-1][1] - self.centerPositions[numPositions-2][1]) * 4) +((self.centerPositions[numPositions-2][1] - self.centerPositions[numPositions-3][1]) * 3) +((self.centerPositions[numPositions-3][1] - self.centerPositions[numPositions-4][1]) * 2) +((self.centerPositions[numPositions-4][1] - self.centerPositions[numPositions-5][1]) * 1)
                deltaX= (sumOfXChanges / 10)
                deltaY=(sumOfYChanges / 10)
                self.predictedNextPosition = [self.centerPositions[-1][-2] + deltaX,
                                              self.centerPositions[-1][-1] + deltaY]



###################################################################################
# 현재 프레임의 blob과 이전 존재했던 blob 간의 match 후 blob정보를 갱신,추가 하는 함수
def matchCurrentFrameBlobsToExistingBlobs(blobs,currentFrameBlobs):
    for existingBlob in blobs:
        existingBlob.matchNewBlob = False
        existingBlob.predictNextPosition()
    for currentFrameBlob in currentFrameBlobs:
        intIndexOfLeastDistance = 0
        dblLeastDistance = 1000000.0
        for i in range(len(blobs)):
            if (blobs[i].blobTracked == True):
                dblDistance=distanceBetweenPoints(currentFrameBlob.centerPositions[-1],blobs[i].predictedNextPosition) #현재 blob의 포지션과 예측된 blob의 다음 포지션의 거리 측정
                if (dblDistance < dblLeastDistance):
                    dblLeastDistance = dblDistance
                    intIndexOfLeastDistance = i
        if (dblLeastDistance < currentFrameBlob.diagonal * 0.5):
            blobs=addBlobToExistingBlobs(currentFrameBlob, blobs, intIndexOfLeastDistance)
        else:
            blobs,currentFrameBlob=addNewBlob(currentFrameBlob, blobs)
    for existingBlob in blobs:
        if (existingBlob.matchNewBlob == False): #blob이 이동한 거리와 blob의 BoundingRect diagonal 크기 비교 후
            existingBlob.numMatch = existingBlob.numMatch + 1
        if (existingBlob.numMatch >=5):
            existingBlob.blobTracked =False
    return blobs

###################################################################################
# 각 객체들 사이의 거리 계산을 위한 함수
def distanceBetweenPoints(pos1,pos2):
    if (pos2==[]):
        dblDistance=math.sqrt((pos1[0])**2+(pos1[1])**2)
    else:
        dblDistance=math.sqrt((pos2[0]-pos1[0])**2+(pos2[1]-pos1[1])**2)
    return dblDistance


###################################################################################
#  이미 존재했던 blob의 정보 프레임당 blob정보 추가
def addBlobToExistingBlobs(currentFrameBlob, blobs, intIndex):
    blobs[intIndex].Contour = currentFrameBlob.Contour
    blobs[intIndex].boundingRect = currentFrameBlob.boundingRect
    blobs[intIndex].centerPositions.append(currentFrameBlob.centerPositions[-1])
    blobs[intIndex].diagonal = currentFrameBlob.diagonal
    blobs[intIndex].aspectRatio = currentFrameBlob.aspectRatio
    blobs[intIndex].blobTracked = True
    blobs[intIndex].matchNewBlob = True
    return blobs

####################################################################################
# 새로운 blob frameblob -> blob으로 정보 갱신
def addNewBlob(currentFrameBlob,Blobs):
    currentFrameBlob.matchNewBlob = True
    blobs.append(currentFrameBlob)
    return blobs,currentFrameBlob


#####################################################################################
#  인식된 blob 영상에 그려줌
def drawBlobInfoOnImage(blobs,m1):
    for i in range(len(blobs)):
        if (blobs[i].blobTracked == True):

            x,y,w,h = blobs[i].boundingRect
            cv2.rectangle(m1,(x, y), (x + w, y + h), SCALAR_BLUE, 2)


 ####################################################################################
# 카운팅된 숫자를 실시간으로 영상 좌측상단에 표시한다.
def drawMortarCountOnImage(mortarCount,image):
    initText = "mortar Counter(1): " + str(mortarCount[0])
    initText2 ="mortar Counter(2): " + str(mortarCount[1])
    initText3 = "mortar Counter(3): " + str(mortarCount[2])

   # initText2 = "mortar Counter2"
    cv2.putText(image, " {}".format(initText), (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, SCALAR_RED, 2)
    cv2.putText(image, " {}".format(initText2), (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, SCALAR_BLUE, 2)
    cv2.putText(image, " {}".format(initText3), (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, SCALAR_BLACK, 2)

  #  cv2.putText(image, " {}".format(initText2), (20, 30),cv2.FONT_HERSHEY_SIMPLEX,0.5 (0,255,255),2)

####################################################################################
# 이전 프레임과 현재프레임 blob의 정보를 비교해 linePosition을 넘은 몰탈만 카운트
def checkIfBlobsCrossedTheLine(blobs,linePosition,mortarCount):
    passLinea= False
    for blob in blobs:
        if (blob.blobTracked == True and len(blob.centerPositions) >= 2):

            prevFrameIndex= len(blob.centerPositions) - 2
            currFrameIndex= len(blob.centerPositions) - 1

            if (blob.centerPositions[prevFrameIndex][-2] <= linePosition and blob.centerPositions[currFrameIndex][-2] > linePosition) :

                passLinea = True


    return passLinea

# 몰탈 종류별로 카운팅 해주고 DB용 class name으로 변경
# 현재상황 1.일반미장용 2.떠붙임용
def mortarClassificationAndCount(classes,mortarCount):

    global classesTemp

    if(classes == 1.0):

        mortarCount[0]+=1
        classesTemp = '01'

    elif(classes == 2.0):

        mortarCount[1]+=1
        classesTemp = '02'

    elif(classes == 3.0):

        mortarCount[2]+=1
        classesTemp = '03'

    elif (classes == 4.0):

        mortarCount[3]+=1
        classesTemp = ' '

    elif (classes == 5.0):

        mortarCount[4]+=1
        classesTemp = ' '

    elif (classes == 6.0):

        mortarCount[5]+=1
        classesTemp = ' '

    else:

        print("Error")


####################################################################################
# Main 부분
while True:


        # 현재 시간불러오기 시,분,초
    now = time.gmtime(time.time())
    # 현재 날짜와 시간 분을 9분늦게 DB 형식에 맞춰 문자열로 저장
    plcTemp = datetime.datetime.now() - datetime.timedelta(minutes=9)

    plcTime = plcTemp.strftime("%Y-%m-%d %H:%M")

    # 프레임 read
    _, imgPre=cap.read()
    _, imgInput =cap.read()




    if not _:
       print('end video')
       break

    #imgPre= cv2.flip(imgPre,1)
    #imgInput=cv2.flip(imgInput,1)
    imgPre=cv2.resize(imgPre, (680,480))
    imgInput=cv2.resize(imgInput, (680,480))

    rows, cols, channels = imgPre.shape

    Line = np.zeros((2, 2), np.float32)
    linePosition = round(cols * 0.75)

    Line[0][0] = linePosition
    Line[1][0] = linePosition
    Line[0][1] = 0
    Line[1][1] = cols

#cv2.imshow('imgPre', imgPre)

    preCopy=imgPre
    inputCopy=imgInput

## 이미지 그레이스케일로 변경
    if _ is True:
        inputCopy = cv2.cvtColor(inputCopy ,cv2.COLOR_BGR2GRAY)
        preCopy= cv2.cvtColor(preCopy, cv2.COLOR_BGR2GRAY)
    else:
        continue

## 가우시안 필터를 이용한 전처리
    inputCopy= cv2.GaussianBlur( inputCopy, (5, 5), 0)
    preCopy = cv2.GaussianBlur( preCopy, (5, 5), 0)

    imgDiff =cv2.absdiff(inputCopy,preCopy)

    #cv2.imshow('diff',imgDiff)

# 이진화
    _, imgBinary = cv2.threshold(imgDiff,30,200,cv2.THRESH_BINARY)

# 모폴로지 연산을 위한 mask 설정
    mask= cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

# 침식,팽창을 이용한 전처리
    for i in range(0, 1):
        imgBinary=cv2.dilate(imgBinary,mask,iterations = 1)
        imgBinary=cv2.dilate(imgBinary,mask,iterations = 1)
        imgBinary=cv2.erode(imgBinary,mask, iterations = 1)

#윤곽선 따기
    image, contours ,hierarchy=cv2.findContours(imgBinary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    imgContour =cv2.drawContours(imgBinary,contours,-1, SCALAR_WHITE, -1)

    #cv2.imshow("contours", imgContour)

#윤곽선 뭉뚱그리기? 아무튼 그런비슷한 개념이었음 윤곽선을 단순화하는 느낌
    global convexHulls
    convexHulls = []

    for i in range(len(contours)):
        convexHull = cv2.convexHull(contours[i])
        convexHulls.append(convexHull)

# 어떤 조건에 취합한 blob을 인식할 것인지 조건문
    frameBlob = []
    for i in range(len(convexHulls)):
        checkBlob = blob(convexHulls[i])

        if (checkBlob.diagonal > 60 and checkBlob.boundingRect[2] > 150 and
                    checkBlob.boundingRect[3] < 250 and checkBlob.aspectRatio < 1.5 and checkBlob.boundingRect[2] <300 ) :
            frameBlob.append(checkBlob)

 # 첫번째 프레임이면 blob정보 그대로 넣고 첫번째 프레임이 아니라면 match 함수로


    if (firstFrame == True):
        for i in frameBlob:
            blobs.append(i)
    else:
        blobs = matchCurrentFrameBlobsToExistingBlobs(blobs, frameBlob)

    inputCopy = imgInput

    drawBlobInfoOnImage(blobs, inputCopy)

    passLine = checkIfBlobsCrossedTheLine(blobs,linePosition,mortarCount)


    #score, classes = objectDetection.mortarClassification(imgInput, image_expanded)


    # blob이 관심영역으로 지정한 linePosition을 지나는 순간 해당영역 이미지 객체 분류
    if(passLine == True):
        cv2.line(inputCopy, (Line[0][0], Line[0][1]), (Line[1][0], Line[1][1]), SCALAR_GREEN, 3)

        # linePosition 영역만 저장
        imgTest = imgPre[30:800,linePosition-700:linePosition+700]
        i= i+10

        cv2.imwrite("hymTest"+str(i)+".jpg",imgTest)
        image_expanded = np.expand_dims(imgTest, axis=0)

        # 몰탈 포장 분류 object_detection API

        score,classes=objectDetection.mortarClassification(imgTest,image_expanded)

        ret = 2
        cv2.imshow("classfication",imgTest)

        #몰탈 카운트 함수
        mortarClassificationAndCount(classes,mortarCount)

        #result[0] = score result[1]= class
        print(score,classes)


    else:
        cv2.line(inputCopy, (Line[0][0], Line[0][1]), (Line[1][0], Line[1][1]), SCALAR_RED, 3)

    # 몰탈 카운트 정보 영상에 출력
    drawMortarCountOnImage(mortarCount,inputCopy)

    cv2.imshow('result',inputCopy)

  #  if ((cap.get(cv2.CAP_PROP_POS_FRAMES) + 1) < cap.get(cv2.CAP_PROP_FRAME_COUNT)):
  #      cap.read(imgInput)

  #  else :
  #      print("end")

  #      break

    # 9분마다 몰탈 포장 갯수 temp에 일시 저장
    if(now.tm_min%10-9 ==0 and now.tm_sec ==0):
        for i in range(0,6):
            mortarTemp[i] = mortarCount[i]

    # 8분마다 몰탈 포장 갯수 DB에 업데이트 단 8분에는 ptime -1 에 해당되는 영역 select 해야함.

    if(now.tm_min%10-8 ==0 and now.tm_sec ==30 ):

        cursor = conn.cursor()
        cursor.execute(
            "select * from PCS_PRODUCT_COUNT where PLC_TIME = %s and ITEM_CD = %s and BPNO= 'A' " ,(plcTime,classesTemp))

        data= cursor.fetchall()
        print (len(data))
        print (data)

        # select 된 갯수가 12개 존재 할 때 update 시작 (오류 방지)
        if(len(data)>= 1 ) :

            cursor2 = conn.cursor()

            cursor2.execute("update PCS_PRODUCT_COUNT set IMG_CNT= %s  where PLC_TIME= %s and BPNO = 'A' ",(int (mortarTemp[0]), plcTime))
            conn.commit()

            print(int(mortarTemp[0]))
            print(plcTime)

        # 8분마다 DB 업데이트 후 몰탈 TEMP 초기화
        for i in range(0, 6):
            mortarTemp[i] = 0

    # 00시 정각에 모든 몰탈 카운트 정보 초기화
    if(now.tm_hour==0 and now.tm_min == 0 and now.tm_sec == 0 ):

        for i in range(0,6):
          mortarTemp[i] = mortarCount[i]
          mortarCount[i]= 0

    # blob clear
    imgPre = imgInput
    frameBlob.clear()

    firstFrame = False
    cv2.waitKey(10)

    #  esc 누르면 종료
    if cv2.waitKey(1) ==27:
        break
'''
    cursor = conn.cursor()
    cursor.execute(
        "select * from PCS_PRODUCT_COUNT where PDATE=%s and PTIME = %s ", (ndate, now.tm_hour))

    data = cursor.fetchall()
    print(len(data))
'''

cap.release()
cv2.destroyAllWindows()

cursor.close()
cursor2.close()
conn.close()
