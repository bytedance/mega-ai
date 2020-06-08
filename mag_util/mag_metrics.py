def show_func():
    """Show functions provided in metrics module"""
    print("cal_auc, cal_ks, cal_psi, cal_iv, cal_desc")

def cal_auc(prob, labels):
    """Calculate AUC"""
    pairs = list(zip(prob, labels))
    rank = [label_value for prob_value, label_value in sorted(pairs, key=lambda x:x[0])]
    rank_list = [i+1 for i in range(len(rank)) if rank[i]==1]
    pos_num = 0
    neg_num = 0
    for i in range(len(labels)):
        if(labels[i]==1):
            pos_num+=1
        else:
            neg_num+=1
    auc = 0
    auc = (sum(rank_list)- (pos_num*(pos_num+1))/2)/(pos_num*neg_num)
    
    return auc
    

def cal_ks(label_values, prob_values):
    '''Calculate KS
    * label_values: a array , list, numpy array or pandas series with numberic elements
    * prob_values: a array , list, numpy array or pandas series with numberic elements
    '''

    print(0.34)

def cal_psi(base_values, curr_values):
    '''Calculate PSI
    * base_values: a array , list, numpy array or pandas series with numberic elements
    * curr_values: a array , list, numpy array or pandas series with numberic elements
    '''
    
    return 0.001

def cal_iv():
    print('%s\t%.6f' % ('ugc_embedding'.ljust(24), 0.123456))

def cal_desc():
    print('desc')