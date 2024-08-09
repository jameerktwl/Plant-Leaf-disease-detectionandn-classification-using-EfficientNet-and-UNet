from optimizer import Hybrid_DTBO
import matplotlib.pyplot as plt
from fitness_function import *


def Hybrid_DTBO(weights):
    Fun_name='F2'; 
    SearchAgents=30;  
    Max_iterations=1000;
    
    lowerbound,upperbound,dimension,fitness=fun_info(Fun_name);
    BestX=Hybrid_DTBO(SearchAgents,Max_iterations,lowerbound,upperbound,dimension,fitness,weights);
    return BestX