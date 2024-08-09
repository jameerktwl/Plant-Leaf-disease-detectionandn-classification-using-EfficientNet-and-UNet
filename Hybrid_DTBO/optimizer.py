import numpy as np
import random


def Hybrid_DTBO(SearchAgents,Max_iterations,lowerbound,upperbound,dimension,fitness,weights):
    lb=np.ones((dimension))*lowerbound
    ub=np.ones((dimension))*upperbound
    
    
    VisitTable = np.zeros((SearchAgents,SearchAgents));### humming birds visit table
    np.fill_diagonal(VisitTable, np.NaN)### initializing  visit table
    
    x=weights
    X   = np.zeros((dimension,dimension))
    best_so_far=[]

    for i in range(dimension):
        rand=np.asarray([random.uniform(0,1) for j in range(SearchAgents)])
        X[:,i]= lb[i]+rand*(ub[i] - lb[i]) 

        
           
    fit=[]
    for i in range(SearchAgents):
        L=X[i,:]
       
        fit.append(fitness(L))
    fit=np.asarray(fit)

    
    
    X   = np.zeros((dimension,dimension))
    
    
    for t in range(Max_iterations):
        # update the best member
        best = np.amin(fit)
        blocation=np.where(fit == np.amin(fit))[0][0]
        

        DirectVector = np.zeros((SearchAgents,dimension));#% Direction vector/matrix
        
        if t==0:
           Xbest=X[blocation,:]                             
           fbest=best                                         
        elif best<fbest:
          fbest=best
          Xbest=X[blocation,:]
          
          
        fit=fit.reshape(1,len(fit))
        XF=np.concatenate((X,fit.T),axis=1)
        

        XFsort=XF[XF[:,-1].argsort()]

       

        X=XFsort[:,0:dimension]
        

        
        fit=XFsort[:,-1].T


        
        N_DI=1+round(0.2*SearchAgents*(1-t/Max_iterations))
        
        DI=XFsort[0:N_DI,0:dimension]
        
        F_DI=XFsort[0:N_DI,dimension].T
        
        
        ## update DTBO population
        for i in range(SearchAgents):
            
            # Phase 1: training by the driving instructor (exploration)
            
            k_i=np.random.permutation(N_DI)[0]
            
            
            DI_ki=DI[0][k_i]
            
           
            F_DI_ki=F_DI[k_i];
            
            
            
            
            I=round(1+random.uniform(0,1))
           
           
            
            if F_DI_ki< fit[i]:
                X_P1=X[i,:]+random.uniform(0,1)* (DI_ki-I*X[i,:]) # Eq. (5) 
            else:
                X_P1=X[i,:]+random.uniform(0,1) * (1*X[i,:]-I*DI_ki) # Eq. (5)
            
            
            X_P1[X_P1>lowerbound] = lowerbound

            X_P1[X_P1<upperbound] = upperbound
            
            # Update X_i based on Eq(6)
            
            F_P1 = fitness(X_P1);
            

            if F_P1 <= fit [i]:
                X[i,:] = X_P1
                fit[i]=F_P1

           # END Phase 1: training by the driving instructor (exploration)
        
           # Phase 2: learner driver patterning from instructor skills (exploration)
           
            
            PIndex=0.01+0.9*(1-t/Max_iterations)
            X_P2=(PIndex)* X[i,:]+(1-PIndex) * (Xbest)   # Eq. (7)
            
            
            X_P2[X_P2>lowerbound] = lowerbound
            X_P2[X_P2<upperbound] = upperbound
           
            
    
            F_P2 = fitness(X_P2)
            if F_P2 <= fit[i]:
                X[i,:] = X_P2
                fit[i]=F_P2
            
            
            # END Phase 2: learner driver patterning from instructor skills (exploration)
        
            # Phase 3: personal practice (exploitation)
            
            
            
            R=0.05;
            X_P3= X[i,:]+ (1-2*random.randint(1,dimension))*R*(1-t/Max_iterations)*X[i,:]
            
            X_P3[X_P3>lowerbound] = lowerbound
            X_P3[X_P3<upperbound] = upperbound
            

            r=random.uniform(0,1);
            if r<1/3 : #   % Diagonal flight
                RandDim = np.random.permutation(dimension);
                if dimension >= 3:
                    RandNum=np.ceil(random.uniform(0,1)*(dimension-2)+1);
                else:
                    RandNum=np.ceil(random.uniform(0,1)*(dimension-1)+1);
                    
                DirectVector[i,list(RandDim)]=1;
            else:
                if r>2/3 :# % Omnidirectional flight
                    DirectVector[i,:]=1;
                else : #% Axial flight
                    RandNum=int(np.ceil(random.uniform(0,1)*dimension));
                    if RandNum >= dimension : RandNum=dimension-1
                    DirectVector[i,RandNum]=1;
            
            if random.uniform(0,1)<0.5 :#  % Guided foraging
                MaxUnvisitedTime = np.amax(VisitTable[i,:]);
                TargetFoodIndex = np.argmax(VisitTable[i,:])
                MUT_Index = np.where(VisitTable[i,:]==MaxUnvisitedTime)[0];
                if len(MUT_Index)>1 :
                    Ind= np.argmin(fit[MUT_Index]);
                    TargetFoodIndex=MUT_Index[Ind];
                
                newPopPos=X[TargetFoodIndex,:]+random.uniform(0,1)*DirectVector[i,:]*(X[i,:]-X[TargetFoodIndex,:]);
                
                newPopPos=np.clip(newPopPos, lowerbound, upperbound) #SpaceBound(newPopPos,Up,Low);
                newPopFit=fitness(newPopPos);
                
                if newPopFit<fit[i]:
                    fit[i]=newPopFit;
                    X[i,:]=newPopPos;
                    VisitTable[i,:]=VisitTable[i,:]+1;
                    VisitTable[i,TargetFoodIndex]=0;
                    VisitTable[:,i]=np.amax(VisitTable)+1;
                    VisitTable[i,i]=np.NaN;
                else:
                    VisitTable[i,:]=VisitTable[i,:]+1;
                    VisitTable[i,TargetFoodIndex]=0;
            else :  # % Territorial foraging
                newPopPos= X[i,:]+random.uniform(0,1)*DirectVector[i,:]*X[i,:];
                newPopPos=np.clip(newPopPos, lowerbound, upperbound)
                newPopFit=fitness(newPopPos);
                if newPopFit<fit[i]:
                    fit[i]=newPopFit;
                    X[i,:]=newPopPos;
                    VisitTable[i,:]=VisitTable[i,:]+1;
                    VisitTable[:,i]=np.amax(VisitTable)+1;
                    VisitTable[i,i]=np.NaN;
                else:
                    VisitTable[i,:]=VisitTable[i,:]+1;
    
            
        if np.mod(t,2*SearchAgents)==0 :#% Migration foraging(Hybrided with artificial hummingbird and driving Training)
            MigrationIndex=np.argmax(fit);
            X[MigrationIndex,:] =random.randint(1,dimension)*(upperbound-lowerbound)+lowerbound;
            fit[MigrationIndex]=fitness(X[MigrationIndex,:]);
            VisitTable[MigrationIndex,:]=VisitTable[MigrationIndex,:]+1;
            VisitTable[:,MigrationIndex]=np.amax(VisitTable)+1;
            VisitTable[MigrationIndex,MigrationIndex]=np.NaN;            
        for i in range(SearchAgents):
            if fit[i]<fbest:
                BestF=fit[i];
                BestX=X[i,:];
    BestX=x

    return max(BestX)

