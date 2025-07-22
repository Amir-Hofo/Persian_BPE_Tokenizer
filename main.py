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

if os.path.exists("merged_dataset.csv"):
    dataset= pd.read_csv("merged_dataset.csv")
else: dataset= preprocess_pipeline_fn()

if os.path.exists("Persian_BPE_Tokenizer_30K.json"):
    tokenizer= Tokenizer.from_file("Persian_BPE_Tokenizer_30K.json")
else: tokenizer= tokenizer_training_fn(dataset)


# test
test_texts= ["این یک متن آزمایشی برای بررسی عملکرد توکنایزر است.",
             "دیروز به کتابخانه رفتم و کتابی درباره تاریخ ایران باستان خواندم.",
             "سلام! چطور می‌توانم به سرعت زبان فارسی را یاد بگیرم؟",
             "هوای تهران امروز خیلی گرم و آفتابی است، ولی شب خنک می‌شود."]

for i, text in enumerate(test_texts):
    encoded_text= tokenizer.encode(text)
    print(10 * "--", f" test {i+1}", 10 * "--")
    print("text: ", text)
    print("tokens: ", encoded_text.tokens)
    print("ids: ", encoded_text.ids)
    # print("decoded: ", tokenizer.decode(encoded_text.ids))
    tokens = tokenizer.decode_batch([[id] for id in encoded_text.ids])
    print("decoded:", ' '.join(tokens))

# evaluation criteria
print(70*"-"), print("evaluation"), print(70*"-")
unk_rate, compression_ratio= evaluation_fn(tokenizer, dataset, 100_000)
print(f"unk rate: {unk_rate}%")
print(f"compression ratio: {compression_ratio}")