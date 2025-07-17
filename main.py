from datasets import load_dataset
import pandas as pd
import blog_dataset_downloader
import os
os.system('cls' if os.name == 'nt' else 'clear')

'''
Datasets:
https://huggingface.co/datasets/wikimedia/wikipedia
https://huggingface.co/datasets/RohanAiLab/persian_blog
https://huggingface.co/datasets/MahtaFetrat/HomoRich-G2P-Persian
--------------------------------------------------------------------
Future datasets:
https://huggingface.co/datasets/cis-lmu/GlotCC-V1
https://huggingface.co/datasets/RohanAiLab/persian_blog_V2
https://huggingface.co/datasets/tspersian/Persian-Dataset
https://huggingface.co/datasets/Depositair/Oscar_Persian_Cleaned
--------------------------------------------------------------------
Processed datasets of more than 10M:
https://huggingface.co/datasets/mshojaei77/PersianCorpus_merged
https://huggingface.co/datasets/yeganehmohammadi98/persian-multi-source-corpus
'''

## wikipedia
wiki_dataset= load_dataset("wikimedia/wikipedia", "20231101.fa")
wiki_dataset= pd.DataFrame(wiki_dataset['train']['text'])

## persian blogs
blog_dataset= pd.read_csv("./blogs/blogs.csv")

## homo rich
homorich_dataset= load_dataset("MahtaFetrat/HomoRich-G2P-Persian", verification_mode= "no_checks")
homorich_dataset= pd.DataFrame(homorich_dataset["train"]["Grapheme"])

## merging datasets
merging_datasets= pd.concat([wiki_dataset, blog_dataset["text"], homorich_dataset], ignore_index= True)
del wiki_dataset, blog_dataset, homorich_dataset
merging_datasets= merging_datasets.sample(frac= 1).reset_index(drop= True)
merging_datasets.to_csv('merging_datasets.csv', index= False, encoding= 'utf-8-sig')
print(merging_datasets.shape)