import datetime 

d = datetime.datetime.now()
d_formated = d.strftime("%Y年%m月%d日%H時%M分")

print(d_formated)