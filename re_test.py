import re
a = '<p> aiueo </p>'
b = '@@1192 To be or not to be. Is that your question?'
c = '@@1192 To be or not to be. Is that your question?<p> aiueo </p> '

print(re.sub('<.+?>', '', a))
print(re.sub('^@@\d+', '', b))
c = re.sub('^@@\d+', '', c)
print('c is:', c)
c = re.sub('<.+?>', '', c)
print('c is:', '', c)
