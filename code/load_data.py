import pandas as pd

def main():
    list_samples,files,names = load_votes()
    import IPython
    IPython.embed()

def load_votes():
    list_samples = []; list_file = []
    for i in range(702):
        df = pd.read_csv('outputs/file'+str(i)+'.csv',header=1)
        df.sort_values(['party'],inplace=True)
        if df.shape == (100,6):
            df['vote'] = df['vote'].replace(['Nay','Yea','Not Voting','Present'],[0,1,0,0])
            list_samples.append(df['vote'])
            list_file.append(i)
            names = list(df['name'])
    return list_samples,list_file,names

if __name__ == '__main__':
    main()
