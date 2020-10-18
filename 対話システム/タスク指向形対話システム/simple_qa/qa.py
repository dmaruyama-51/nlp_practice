import re 

qa_list = [
    ["富士山.*高さ", "富士山の高さは3,776mです。"],
    ["東京.*区.*いくつ", "東京都には23の区があります。"]
]

while True:
    text = input("> ")

    if re.search("ありがとう", text):
        print("ありがとうございました。また質問してくださいね！")
        break
    else:
        found = False 

        for qa in qa_list:
            pattern = qa[0] # 質問パターン
            answer = qa[1] # 解答
            if re.search(pattern, text):
                print(answer)
                found = True 
                break

        if not found:
            print("すみません、わかりません。")