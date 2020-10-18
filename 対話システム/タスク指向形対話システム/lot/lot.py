import random 

utterance = input("user> ")
if utterance == "くじびき":
    while True:
        results = ["あたり", "はずれ"]
        result = random.choice(results)
        if result == "あたり":
            comment = "おめでとうございます。"
        else:
            comment = "残念でした。"
        print("くじ引きの結果は...{}です！{}".format(result, comment))

        retry_check = input("もう一度くじを引きますか？> ")
        if retry_check != "はい":
            print("くじびきを終了します。")
            break
    
    else:
        print("すみません、わかりません。")