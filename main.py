from packages import *
from utils import *
from preprocess import *

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

## merging datasets
wiki_dataset, blog_dataset, homorich_dataset= preprocess_pipeline_fn()
merging_datasets= pd.concat([wiki_dataset, blog_dataset, homorich_dataset], ignore_index= True)
del wiki_dataset, blog_dataset, homorich_dataset
print(70*"-"), print("preprocessing is done."), print(70*"-")

# merging_datasets= merging_datasets.sample(frac= 1).reset_index(drop= True)
merging_datasets.to_csv('merging_datasets.csv', index= False, encoding= 'utf-8-sig')
print(merging_datasets.shape)