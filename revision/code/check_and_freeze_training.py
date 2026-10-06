from revision_common import *
import numpy as np
from train_revision import numeric_block,transform
from sklearn.preprocessing import StandardScaler,OneHotEncoder
def main():
    assert threshold(np.array([.9,.9,.8,.1]),.25)>.9
    assert threshold(np.array([.9,.9,.8,.1]),.5)==.9
    assert threshold(np.array([.1,.1]),0)>.1
    # Pure feature-processing test; no classifier is fitted.
    x=np.zeros((6,2480),np.float32);x[:,0]=[-100,-1,0,1,100,10000];x[:,2]=[0,1,0,1,2,3]
    cat=[2,3,4,5,6,701,702];num=np.setdiff1d(np.arange(2480),cat)
    scaler=StandardScaler().fit(numeric_block(x,np.arange(4),num))
    encoder=OneHotEncoder(handle_unknown='ignore',sparse_output=False,dtype=np.float32).fit(x[:4,cat])
    z=transform(x,np.arange(6),num,cat,scaler,encoder)
    assert np.isfinite(z).all() and z.dtype==np.float32
    assert np.allclose(z[:4,:len(num)].mean(axis=0),0,atol=1e-6)
    cfg=read(ROOT/'config/revision_analysis.json')
    for seed in SEEDS:
        p=ROOT/f'data/source_diagnostic_split_{seed}.npy'
        assert sha(p)==cfg['prepared_data_hashes'][p.name]
        assert set(np.unique(np.load(p)))=={0,1,2}
    save(ROOT/'logs/software_preflight.json',{'status':'PASS','checks':['threshold ties','zero FP budget','signed transformation','training-only scaling','unknown category handling','five prepared split identities'],'classifier_fits':0})
    p=ROOT/'config/training_code_hashes.json';assert not p.exists()
    save(p,{n:sha(ROOT/'code'/n) for n in ['revision_common.py','train_revision.py']})
    save(ROOT/'config/author_confirmation.json',{'exclusive_submission':'CONFIRMED_NOT_SUBMITTED_TO_ANY_JOURNAL','full_postal_addresses':'OMITTED_AT_AUTHOR_REQUEST_FOR_CURRENT_SUBMISSION_STAGE','author_message':'1. chưa gửi tạp chí nào. 2. Địa chỉ bưu chính theo tôi là không bắt buộc ở giai đoạn submit đến tạp chí nên không cần sử dụng lúc này','remaining_author_confirmation_blockers':[],'publication_guarantee':False})
    print('PASS: training preflight; configuration and executable hashes recorded')
if __name__=='__main__':main()
