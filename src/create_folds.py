# import pandas and model selection module of scikit-learn
import pandas as pd
from sklearn import model_selection

if __name__=="__main__":
    df=pd.read_csv('/Users/muratkahraman/Downloads/project/input/train.csv')

    #we create a new column called kfold and fill it with -1
    df['kfold']=-1

    #next step is to randomize the rows of data
    df=df.sample(frac=1).reset_index(drop=True)

    #iniate the kfolds class from model_selection module
    kf=model_selection.KFold(n_splits=5)

    #fill the new kfold column
    for f, (t_,v_) in enumerate(kf.split(X=df)):
        df.loc[v_,'kfold']=f

    #save the new csv with kfold column  
    df.to_csv("/Users/muratkahraman/Downloads/project/input/train_folds.csv", index=False) 