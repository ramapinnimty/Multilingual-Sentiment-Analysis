from googletrans import Translator
import json
import os

def read_input_files(source_folder,save_path):
    file_list=os.listdir(source_folder)
    var=1
    for file in file_list:
        englishFile=open(os.path.join(source_folder,file),encoding='utf-8')
        x=englishFile.readlines()[0]
        x1=x[0:5000]
        x1=x1.replace('<br /><br />','')
        x2=x[5000:]
        x2=x2.replace('<br /><br />','')
        translator=Translator()
        try:
            a=translator.translate(str(x1),dest='hi')   
            b=translator.translate(str(x2),dest='hi')
        except json.decoder.JSONDecodeError as err:
            print(err)
        else:
            englishFile.close()
            print(str(var)+"\n")
            var=var+1
            completeName=os.path.join(save_path,file)
            hindiFile=open(completeName,'w',encoding="utf-8")
            hindiFile.writelines(str(a.text))
            hindiFile.writelines(str(b.text))
            hindiFile.close() 
        
print(os.listdir(os.curdir))
unsup=read_input_files('./unsup', './hindi')        
