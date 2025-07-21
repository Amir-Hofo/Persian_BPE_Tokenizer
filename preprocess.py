from packages import *
from utils import *

def preprocess_fn(dataset: pd.DataFrame, normaliztion= False, show_sample= False) -> pd.DataFrame:
    # Remove duplicate texts
    dataset= dataset.drop_duplicates(subset='text')
    dataset= dataset.copy()
    # Remove URLs
    dataset['text']= dataset['text'].apply(lambda x: re.sub(r'http\S+|www\S+', '', x) if isinstance(x, str) else x)
    # Remove HTML tags
    dataset['text']= dataset['text'].apply(lambda x: re.sub(r'<[^>]+>', '', x) if isinstance(x, str) else x)
    # Remove emojis and unnecessary characters
    dataset['text']= dataset['text'].apply(lambda x: re.sub(r'[^\u0600-\u06FF\s.,!?]', '', x) if isinstance(x, str) else x)
    # Remove English texts
    dataset['text']= dataset['text'].apply(lambda x: re.sub(r'[a-zA-Z]+', '', x) if isinstance(x, str) else x)
    # Remove empty or invalid texts
    dataset= dataset[dataset['text'].notna() & (dataset['text'].str.strip() != '')]
    # Remove extra spaces
    dataset['text']= dataset['text'].apply(lambda x: re.sub(r'\s+', ' ', x).strip() if isinstance(x, str) else x)
    # Normalization
    if normaliztion:
        dataset['text']= dataset['text'].apply(lambda x: Normalizer(persian_numbers= True).normalize(x) if isinstance(x, str) else x)
    if show_sample:
        print(dataset['text'].sample(n= 3).to_list())
    return dataset


def uniform_length(dataset: pd.DataFrame, target_length= 1000) -> pd.DataFrame:
    full_text= ' '.join(dataset['text'].dropna().astype(str))
    chunks, start= [], 0
    while start < len(full_text):
        end= min(start+ target_length, len(full_text))
        while end < len(full_text) and full_text[end] not in [' ', '\n']:
            end +=1
        chunks.append(full_text[start: end].strip())
        start= end +1
    return pd.DataFrame(chunks, columns= ['text'])


def preprocess_pipeline_fn():
    wiki_dataset= load_dataset("wikimedia/wikipedia", "20231101.fa")
    wiki_dataset= pd.DataFrame(wiki_dataset['train']['text'], columns= ['text'])

    blog_dataset= pd.read_csv("./blogs/blogs.csv")

    homorich_dataset= load_dataset("MahtaFetrat/HomoRich-G2P-Persian", verification_mode= "no_checks")
    homorich_dataset= pd.DataFrame(homorich_dataset["train"]["Grapheme"], columns= ['text'])
    print(70*"-"), print("download & load datasets is done."), print(70*"-")

    dataset_names= ["Wikipedia", "Persian Blog", "HomoRich"]
    datasets= [wiki_dataset, blog_dataset, homorich_dataset]
    for i in range(len(datasets)):
        datasets[i]= uniform_length(preprocess_fn(datasets[i]))
        eda_dataset(datasets[i], dataset_names[i])
        show_short_samples(datasets[i], dataset_names[i])
    
    return datasets[0], datasets[1], datasets[2]