import math
import numpy as np
import random




def F1(x):
      R=sum(x**2)
      return R
def F2(x):
    R=sum(abs(x))+np.prod(abs(x))
    return R
def F3(x):
    dimension=np.size(x,2)
    R=0;
    for i in range(dimension):
        R=R+sum(x[1:i])**2
def F4(x):
    R=max(abs(x))
    return R

def F5(x):
    dimension=np.size(x,2)
    R=sum(100*(x[2:dimension]-(x[1:dimension-1]**2))**2+(x[1:dimension-1]-1)**2)
    return R
 
def F6(x):
    R=sum(abs((x+.5))**2)
    return R
def F7(x):
    dimension=np.size(x,2)
    R=sum(dimension*(x**4))+random.sample()
    return R
def F8(x):
    R=sum(-x*math.sin(math.sqrt(abs(x))))
    return R

def F9(x):
    dimension=np.size(x,2)
    R=sum(x**2-10*math.cos(2* math.pi*x))+10*dimension
    return R

def F10(x):
    dimension=np.size(x,2)
    R=-20*math.exp(-.2*math.sqrt(sum(x**2)/dimension))-math.exp(sum(math.cos(2*math.pi*x))/dimension)+20+math.exp(1);
    return R

def F11(x):
    dimension=np.size(x,2)
    R=sum(x**2)/4000-math.prod(math.cos(x/math.sqrt(dimension)))+1
    return R

def Ufun(x,a,k,m):
    R=k*((x-a)^m)*(x+a)+k*((x+a)^m)*(x(-a))
    return R

def F12(x):
    dimension=np.size(x,2)
    R=(math.pi/dimension)*(10*((math.sin(math.pi*(1+(x[1]+1)/4)))**2)+sum((((x[dimension-1]+1)/4)**2)*(1+10*((math.sin(math.pi*(1+(x[dimension]+1)/4))))**2))+((x[dimension]+1)/4)**2)+sum(Ufun(x,10,100,4))
    return R
def F13(x):
    dimension=np.size(x,2);
    R=.1*((math.sin(3*math.pi*x(1)))**2+sum((x[dimension-1]-1)**2*(1+(math.sin(3.*math.pi*x[dimension])))**2))+((x[dimension-1]**2)*(1+(math.sin(2*math.pi*x[dimension]))**2))+sum(Ufun(x,5,100,4))
    return R

def F14(x):
    bS=[]
    aS=[-32,-16,0,16,32,-32,-16,0,16,32,-32,-16,0,16,32,-32,-16,0,16,32,-32,-16,0,16,32,-32,-32,-32,-32,-32,-16,-16,-16,-16,-16,0,0,0,0,0,16,16,16,16,16,32,32,32,32,32]
    for j in range(1,25):
        bS[j]=sum((np.transpose(x)-aS[:,j])**6)
    R=(1/500+sum(1./(range(1,25)+bS)))**(-1)
    return R
   
def F15(x):
    aK=[.1957,.1947,.1735,.16,.0844,.0627,.0456,.0342,.0323,.0235,.0246]
    bK=[.25,.5,1,2,4,6,8,10,12,14,16]
    bK=1/bK
    R=sum((aK-((x[1]*(bK**2+x[2]*bK))/(bK**2+x[3]*bK+x[4])))^2)
    return R

def F16(x):
    R=4*(x(1)^2)-2.1*(x(1)^4)+(x(1)^6)/3+x(1)*x(2)-4*(x(2)^2)+4*(x(2)^4)
    return R
def F17(x):
    R=(x(2)-(x(1)^2)*5.1/(4*(math.pi^2))+5/math.pi*x(1)-6)^2+10*(1-1/(8*math.pi))*math.cos(x(1))+10
    return R
def F18(x):
    R=(1+(x(1)+x(2)+1)^2*(19-14*x(1)+3*(x(1)^2)-14*x(2)+6*x(1)*x(2)+3*x(2)^2))*...
    (30+(2*x(1)-3*x(2))^2*(18-32*x(1)+12*(x(1)^2)+48*x(2)-36*x(1)*x(2)+27*(x(2)^2)));
    return R
def F19(x):
    aH=[3,10,30,.1,10,35,3,10,30,.1,10,35]
    cH=[1,1.2,3,3.2];
    pH=[.3689,.117,.2673,.4699,.4387,.747,.1091,.8732,.5547,.03815,.5743,.8828]
    R=0
    for i in range(1,4):
        R=R-cH[i]*math.exp(-(sum(aH[i]*((x-pH[i])^2))))
    return R
   
def F20(x):
    aH=[10,3,17,3.5,1.7,8,.05,10,17,.1,8, 14,3,3.5,1.7,10,17,8,17,8,.05,10,.1,14]
    cH=[1,1.2,3,3.2]
    pH=[.1312,.1696,.5569,.0124,.8283,.5886,.2329,.4135,.8307,.3736,.1004,.9991,.2348,.1415,.3522,.2883,.3047,.6650,.4047,.8828,.8732,.5743,.1091,.0381]
    R=0
    for i in range(1,4):
        R=R-cH[i]*math.exp(-(sum(aH[i])*((x-pH[i]))^2))
    return R
   
def F21(x):
    aSH=[4,4,4,4,1,1,1,1,8,8,8,8,6,6,6,6,3,7,3,7,2,9,2,9,5,5,3,3,8,1,8,1,6,2,6,2,7,3.6,7,3.6]
    cSH=[.1,.2,.2,.4,.4,.6,.3,.7,.5,.5]
    R=0
    for i in range(1,5):
        R=R-((x-aSH[i])*np.transpose((x-aSH[i]))+cSH[i])^(-1)
    return R
def  F22(x):
    aSH=[4,4,4,4,1,1,1,1,8,8,8,8,6,6,6,6,3,7,3,7,2,9,2,9,5,5,3,3,8,1,8,1,6,2,6,2,7,3.6,7,3.6]
    cSH=[.1,.2,.2,.4,.4,.6,.3,.7,.5,.5]
    R=0
    for i in range(1,7):
        R=R-np.transpose((x-aSH[i])*(x-aSH[i]))+(cSH[i])^(-1)
    return R
def F23(x):
    aSH=[4,4,4,4,1,1,1,1,8,8,8,8,6,6,6,6,3,7,3,7,2,9,2,9,5,5,3,3,8,1,8,1,6,2,6,2,7,3.6,7, 3.6] 
    cSH=[.1,.2,.2,.4,.4,.6,.3,.7,.5,.5]
    R=0;
    for i in range(1,10):
        R=R-np.transpose((x-aSH[i])*(x-aSH[i]))+cSH[i]^(-1);
    return R







def fun_info(F):
    if F=='F1':
        fitness =F1
        lowerbound=-100
        upperbound=100
        dimension=30
    elif F== 'F2':
        fitness = F2;
        lowerbound=-1
        upperbound=1
        dimension=30
    elif F=='F3':
        fitness = F3
        lowerbound=-100
        upperbound=100
        dimension=30
    elif F== 'F4':
        fitness = F4
        lowerbound=-100
        upperbound=100
        dimension=30
    elif F=='F5':
        fitness = F5
        lowerbound=-30
        upperbound=30
        dimension=30
    elif F=='F6':
        fitness = F6
        lowerbound=-100
        upperbound=100
        dimension=30
    elif F=='F7':
        fitness = F7
        lowerbound=-1.28
        upperbound=1.28
        dimension=30
    elif F == 'F8':
        fitness = F8
        lowerbound=-500
        upperbound=500
        dimension=30
    elif F == 'F9':
        fitness = F9
        lowerbound=-5.12
        upperbound=5.12
        dimension=30
    elif F=='F10':
        fitness = F10
        lowerbound=-32
        upperbound=32
        dimension=30
    elif F=='F11':
        fitness = F11
        lowerbound=-600
        upperbound=600
        dimension=30
    elif F=='F12':
        fitness = F12
        lowerbound=-50
        upperbound=50
        dimension=30
    elif F=='F13':
        fitness = F13
        lowerbound=-50
        upperbound=50
        dimension=30
    elif F=='F14':
        fitness = F14
        lowerbound=-65.536
        upperbound=65.536
        dimension=2
    elif F=='F15':
        fitness = F15
        lowerbound=-5
        upperbound=5
        dimension=4
    elif F=='F16':
        fitness = F16
        lowerbound=-5
        upperbound=5
        dimension=2
    elif F=='F17':
        fitness = F17
        lowerbound=[-5,0]
        upperbound=[10,15]
        dimension=2
    elif F =='F18':
        fitness = F18
        lowerbound=-2
        upperbound=2
        dimension=2
    elif F == 'F19':
        fitness = F19
        lowerbound=0
        upperbound=1
        dimension=3
    elif F =='F20':
        fitness = F20
        lowerbound=0
        upperbound=1
        dimension=6
    elif F == 'F21':
        fitness = F21
        lowerbound=0
        upperbound=10
        dimension=4
    elif F =='F22':
        fitness = F22
        lowerbound=0
        upperbound=10
        dimension=4
    elif F=='F23':
        fitness = F23
        lowerbound=0
        upperbound=10
        dimension=4
    return lowerbound,upperbound,dimension,fitness
    




 
        

            

        


        


        


    

