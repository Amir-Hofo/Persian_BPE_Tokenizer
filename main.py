from packages import *
from preprocess import *
from tokenizer_training import *

'''
Datasets:
https://huggingface.co/datasets/wikimedia/wikipedia
https://huggingface.co/datasets/RohanAiLab/persian_blog
https://huggingface.co/datasets/MahtaFetrat/HomoRich-G2P-Persian
'''

os.system('cls' if os.name == 'nt' else 'clear')

# dataset= preprocess_pipeline_fn()
dataset= pd.read_csv("merged_dataset.csv")

tokenizer= tokenizer_training_fn(dataset)
# tokenizer= Tokenizer.from_file("Persian_BPE_Tokenizer_10K.json")




# # vocab
# vocab= tokenizer.get_vocab()
# sorted_vocab= dict(sorted(vocab.items(), key= lambda item: item[1]))
# print(10 * "--", " vocab ", 10 * "--", "\n",
# *(f"{token}: {index}" for i, (token, index) in enumerate(sorted_vocab.items()) if i< 10),
# "...", *(f"{token}: {index}" for token, index in list(sorted_vocab.items())[-5:]), sep=", ")

# # test
# test_texts= ["این یک متن آزمایشی برای بررسی عملکرد توکنایزر است.",
#              "دیروز به کتابخانه رفتم و کتابی درباره تاریخ باستان خواندم.",
#              "سلام! چطور می‌توانم به سرعت زبان فارسی را یاد بگیرم؟",
#              "هوای تهران امروز خیلی گرم و آفتابی است، ولی شب خنک می‌شود."]

# encoded_text= tokenizer.encode(text)
# print(10 * "--", " test ", 10 * "--")
# print("text: ", text)
# print("tokens: ", encoded_text.tokens)
# print("ids: ", encoded_text.ids)
# print("decoded: ", tokenizer.decode(encoded_text.ids))