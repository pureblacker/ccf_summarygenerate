import os,json
def preprocess(data_name):
    pathraw=os.path.join('data_set',data_name)
    filenames=os.listdir(pathraw)

    for i in filenames:
        path=os.path.join(pathraw,'{}'.format(i))

        with open(path,'r',encoding='utf8') as f:
            data=json.load(f)
            data['source']=''.join(data['source']).replace('`','')\
                            .replace('.','').replace('|','').replace(':','').replace('-','')\
                            .replace(',','').replace('!','').replace('?','').replace('\'','')
            data['summary']=''.join(data['summary']).replace('`','')\
                            .replace('.','').replace('|','').replace(':','').replace('-','')\
                            .replace(',','').replace('!','').replace('?','').replace('\'','')
        f.close()
        with open(path,'w') as f:
            json.dump(data,f)
        f.close()

    
preprocess('train')
preprocess('test')
preprocess('dev')