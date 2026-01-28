from transformers import GPT2TokenizerFast

tokenizer = GPT2TokenizerFast.from_pretrained('Xenova/gpt-4')
enc = tokenizer.encode('hello,world 3.10 and 3.9')
print(enc)
# 把上面的词ID输出 解码出来
for token_id in enc:
    print(tokenizer.decode([token_id]))


# 不同模型的分词器 在线体验
# https://huggingface.co/spaces/Xenova/the-tokenizer-playground