import json
import re

# 欲しい形式第一段階: 各テキストのリストにする
with open('coca-samples-text/text_acad.txt', 'r') as f:
    acad = f.readlines()

with open('coca-samples-text/text_blog.txt', 'r') as f:
    blog = f.readlines()

with open('coca-samples-text/text_fic.txt', 'r') as f:
    fic = f.readlines()

with open('coca-samples-text/text_mag.txt', 'r') as f:
    mag = f.readlines()

with open('coca-samples-text/text_news.txt', 'r') as f:
    news = f.readlines()

with open('coca-samples-text/text_spok.txt', 'r') as f:
    spok = f.readlines()

with open('coca-samples-text/text_tvm.txt', 'r') as f:
    tvm = f.readlines()

with open('coca-samples-text/text_web.txt', 'r') as f:
    web = f.readlines()

texts_genres = {'acad':acad, 'blog':blog, 'fic':fic, 'mag':mag, 'news':news, 'spok':spok, 'tvm':tvm, 'web':web}
girl_genres = {}
for genre, texts in texts_genres.items():
    for text in texts: #readlinesで読み込んだlineの一つひとつに対して、lineは一つのテキストとみなせる
        text = re.sub('^@@\d+', '', text)
        text = re.sub('<.+?>', '', text)
        text = re.sub('@+', '', text)
        text = re.sub('\n', '', text)
        sentences = re.split("(?<=\.)", text)
        for sentence in sentences:
            # 後ほど発覚, Skullgirlsみたいな部分文字列も含んでしまっていた
            # なので一旦単語レベルで分割してからチェック
            words = list(sentence.split())
            if 'girl' in words or 'girls' in words:
                target_sentence = sentence
                print('target_sentence:', target_sentence)
                girl_genres[target_sentence] = genre

with open('girl_genres.json', 'w') as f:
    json.dump(girl_genres, f)

print(girl_genres)