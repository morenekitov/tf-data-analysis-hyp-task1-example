import pandas as pd
import numpy as np
import scipy.stats as st
from statsmodels.stats.weightstats import ztest


chat_id = 287133833 

def solution(x_success: int, 
             x_cnt: int, 
             y_success: int, 
             y_cnt: int) -> bool:
    
    a_val = 0.09
    # h0 - средние одинаковы

    loc_suc_x, loc_suc_y = x_success / x_cnt, y_success / y_cnt
    var_suc_x,var_suc_y = (x_success / x_cnt)*((x_cnt - x_success) / x_cnt), (y_success / y_cnt)*((y_cnt - y_success) / y_cnt)
    se_loc_suc_x,se_loc_suc_y = np.sqrt(var_suc_x) / np.sqrt(x_cnt), np.sqrt(var_suc_y) / np.sqrt(y_cnt)
    P = (y_success+x_success)/ (x_cnt + y_cnt)
    z =(loc_suc_x-loc_suc_y) / np.sqrt(   P*(1-P) * (1/x_cnt + 1/y_cnt))
    threshold = st.norm.ppf(1-a_val, loc=0, scale=1) 
    return (z > threshold)
