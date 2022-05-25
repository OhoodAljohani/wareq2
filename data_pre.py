
# Importing : 
import pandas as pd
import numpy as np
import re
import enchant
# Scrapping plants name from the file : 
# 1 : Reading the file : 
filename = r'.\data\Riyadh-Plants-Manual-Ar4.txt' 
# 2 : Defining a list ( data ) to store all lines : 
data = [] 
#d = enchant.Dict("en_US")
# 3 : Opening the file :
with open(filename,encoding="utf8") as fn:  
# 4 : Read each line : 
   line = fn.readline()
# 5 : Keep count of lines in case needed later : 
   line_count = 0
   while line:
       
       #print("Line {}: {}".format(lncount, ln.strip('/n')))
       s = re.sub('\W+',' ',line).strip()
       type(s)
       #s = ln.strip('/n')
       if s:
           #if not d.check(s):
           data.append(s)
       line = fn.readline()
       line_count += 1
data
# Final result info : 
print(" data contains : ",len(data)," Lines . ") 
# Define names to hold the plants names  and info to hold information 
# about each plant : 
names = []
info = []
s = ""
for n in data:
    if re.match("^[a-zA-Z]+.*", n):
        index = data.index(n)
        new = data[index+2]
        names.append(n)
        names.append(new)
        for i in range(6):
            index= index+3
            s = s+data[index+1]
        info.append(s)
        #index = 0
        #s = ''
names
len(info)
for i in names:
    if i=="معلومات عامة":
        index = data.index(i)
        del names[index+1]
        names.remove(i)
names
data
############
cname = []
cinfo = []
for n in names:
    if re.match("^[a-zA-Z]+.*", n):
        index = data.index(n)
        new = data[index+2]
        cname.append(new)
        '''if new !="معلومات عامة":
            cname.append(new)
            for i in range(4):
                index= data.index(new)
                s = s+data[index+i]
            cinfo.append(s)
        index = 0
        s = '''''
cname
cinfo
for c in cname:
    '''if re.match("^[a-zA-Z]+.*", c):
        cname.remove(c)
    elif len(c)>=30:
        cname.remove(c)'''
    if c =="معلومات عامة":
        cname.remove(c)
len(cname)
cname
k =[]
for t in cname:
    if t not in k:
        k.append(t)
for f in cinfo:
    if re.match("^[a-zA-Z]+.*", f):
        cinfo.remove(f)
m = list(dict.fromkeys(cname))
df= pd.DataFrame()
df['name'] = cname
df['info'] = cinfo
df.head()
color = []
time = []
index = 0
for n in data:
    if n == "معلومات عامة":
        index = data.index(n)
        new = data[index+36]
        color.append(new)
        new2 = data[index+4]
        time.append(new2)
data
color

time


###############################
import re
filename = r'.\data\Riyadh-Plants-Manual-Ar5.txt' 
data2 = [] 
import enchant
d = enchant.Dict("en_US")
with open(filename,encoding="utf8") as fn:  
# Read each line
   line = fn.readline()
# Keep count of lines
   lncount = 0
   while line:
       
       #print("Line {}: {}".format(lncount, ln.strip('/n')))
       s = re.sub('\W+',' ',line).strip()
       type(s)
       #s = ln.strip('/n')
       if s:
           if not d.check(s):
               data2.append(s)
       line = fn.readline()
       lncount += 1
data2
del data2[231]
del data2[230]
del data2[229]
del data2[71]
del data2[0]
del data2[148]
len(data2)
a = data2[0]
a[0]
num = []
for d in data2 :
    a = d 
    u = ""
    for i in a : 
        if i.isdigit():
            u = u+i
    num.append(u)
len(num)

import pandas as pd 
frame = pd.DataFrame()
frame["num"] = num
frame["name"] = data2
'''frame.sort_values(by='num',inplace=True)
frame.set_index("index")
frame.sort_index(ascending=True)'''
frame.name = frame.name.str.replace('\d+', '')
frame["num"] = pd.to_numeric(frame["num"])
frame.info()
frame = frame.sort_values(by='num')
frame.to_csv("data.csv")
data = pd.read_csv("data.csv")
frame = data.sort_values(by='num')
import pandas as pd 
frame = pd.read_csv("data.csv")
frame = frame.sort_values(by='num')
frame.head()
###############################################
# Scrapping plants name from the file : 
# 1 : Reading the file : 
filename = r'.\data\Riyadh-Plants-Manual-Ar4.txt' 
# 2 : Defining a list ( data ) to store all lines : 
text = [] 
#d = enchant.Dict("en_US")
# 3 : Opening the file :
with open(filename,encoding="utf8") as fn:  
# 4 : Read each line : 
   line = fn.readline()
# 5 : Keep count of lines in case needed later : 
   line_count = 0
   while line:
       
       #print("Line {}: {}".format(lncount, ln.strip('/n')))
       s = re.sub('\W+',' ',line).strip()
       type(s)
       #s = ln.strip('/n')
       if s:
           #if not d.check(s):
           text.append(s)
       line = fn.readline()
       line_count += 1
##########################################
growth = []
color = []
time = []
len(text)
text[0]
for i,t in enumerate(text): 
    if t == "طبيعة النمو":
        growth.append(text[i+1])
    if t == "اللون":
        color.append(text[i+1])
    if t == "موعد اإلزهار":
        time.append(str(text[i+1])+" "+str(text[i+2]))
        #time.append(text[i+2])
len(growth)
len(color)
len(time)
time 

'''colors = ["أبيض أصفر","وردي","وردي باهت","أصفر المع","أحمر المع","أصفر لامع","أصفر","أبيض","أصفر خفيف","أخضر","أبيضأصفر باهت","أخضر خفيف","وردي غامق","بنفسجي","أحمر برتقالي داكن","أصفر برتقالي","أبيض أخضر خفيف","أصفر وردي أحمر, ","وردي أحمر","ليلكي وردي باهت","وردي الحلق","أصفر فاتح","أبيض ثانوي","أخضر باهت","أرجواني","أحمر أرجواني, ","أزرق","أصفر خفيف أحمر"]
c =[]
for color in text:
    if color in colors:
        c.append(color)
c
'''
frame.info()
color.append("none")
time.append("none")
import pandas as pd 
frame2 = pd.DataFrame()
frame2["Growth"] = growth
frame2["Color"] = color
frame2["Time"] = time
frame2.info()
frame2.to_csv("data2.csv")
