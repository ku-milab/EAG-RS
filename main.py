## Define libraries
import config
import setting as st
from LRP_sync import *
from Train_EAG_RS import *
"""
TMI revision source code
"""
# import matplotlib.pyplot as plt
def main(opt_):
    exp = st.setting(opt_, True) #TODO: experimental setting
    print('Good setting')
    ''' Step 1. Inter-regional relation learning '''
    First_step(exp, Flag_sw=False)
    Second_step(exp, Flag_sw=False)

    ''' Step 2. Connection-wise relevance score estimation module '''
    Evaluate_LRP(exp, Flag_sw=False)
    Extract_LRP_graph_feat(exp, Flag_sw=False)

    ''' Step 3. Diagnosis-informative ROI selection module and classifier '''
    Final_step(exp, Flag_sw=False)



if __name__ == "__main__":
    opt_ = {}
    # for mask_ in list(np.arange(0,1,0.1)):
    opt_['ROI_mask'] = 0.1
    site = "total"
    for fold in range(5):
    # for fold in range(1):

        opt_['site'] = site
        opt_['tmp_fold'] = fold

        config_ = config.get_args(opt_)
        dir = '../'
        # Text logging
        f = open(dir + 'log.txt', 'a')
        writelog(f, '---------------')
        writelog(f, str(config_))
        writelog(f, '---------------')
        main(config_)
