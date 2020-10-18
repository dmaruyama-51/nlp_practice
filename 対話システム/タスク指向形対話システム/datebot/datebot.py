import datetime 

user_uttr = input("user> ")

if "何時" in user_uttr:
    d = datetime.datetime.now() 
    d_formatted = d.strftime("%Y年%m月%d日%H時%M分")

    system_uttr = "{}です。".format(d_formatted)
    print(system_uttr)
else:
    print("すみません、わかりません。")