from transformers import pipeline

clf = pipeline("sentiment-analysis", model="uer/roberta-base-finetuned-jd-binary-chinese")
print(clf("我今天心情很好"))
print(clf("今天心情麻麻地"))