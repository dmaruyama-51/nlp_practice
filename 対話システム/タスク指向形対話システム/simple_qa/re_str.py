import re 

# . は任意の1文字を表現
# * は直前の文字の0回以上の繰り返し
pattern = "対話.*システム"
texts = ["対話のシステム", "対話システム", "対話できるシステム"]

for text in texts:
    if re.search(pattern, text):
        print(pattern, text, "マッチしました")
    else:
        print(pattern, text, "マッチしませんでいた")