def show_func():
    """Show functions provided in metrics module"""
    print("cal_auc, cal_ks, cal_psi, cal_iv, cal_desc")

def cal_auc(prob, labels):
    """Calculate AUC"""
    pairs = list(zip(prob, labels))
    rank = [values2 for values1, values2 in sorted(pairs, key=lambda x:x[0])]
    rankList = [i+1 for i in range(len(rank)) if rank[i]==1]
    posNum = 0
    negNum = 0
    for i in range(len(labels)):
        if(labels[i]==1):
            posNum+=1
        else:
            negNum+=1
    auc = 0
    auc = (sum(rankList)- (posNum*(posNum+1))/2)/(posNum*negNum)
    
    return auc
    

def cal_ks():
    print(0.34)

def cal_psi():
    print(0.01)

def cal_iv():
    print('%s\t%.6f' % ('ugc_embedding'.ljust(24), 0.123456))

def cal_desc():
    print('desc')