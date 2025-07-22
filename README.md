# Persian BPE Tokenizer (30K)

A Byte-Pair Encoding (BPE) tokenizer with a vocabulary size of 30,000, trained on ~2M Persian texts with an average length of 10,000 characters for NLP tasks.

## Usage

### Encoding
```python
from tokenizers import Tokenizer
tokenizer= Tokenizer.from_file("Persian_BPE_Tokenizer_30K.json")
encoded_text= tokenizer.encode("این یک متن آزمایشی است.")
print("Tokens:", encoded_text.tokens)
print("IDs:", encoded_text.ids)
```

### Decoding
```python
decoded_text= tokenizer.decode_batch([[id] for id in encoded_text.ids])
print("Decoded:", decoded_text)
```

## Training Data
This tokenizer was trained on the following datasets:
- Wikipedia (20231101.fa): https://huggingface.co/datasets/wikimedia/wikipedia
- Persian Blog: https://huggingface.co/datasets/RohanAiLab/persian_blog
- HomoRich: https://huggingface.co/datasets/MahtaFetrat/HomoRich-G2P-Persian

## License
Code and tokenizer: MIT License

## Evaluation Metrics
- UNK Rate: 0.0% (on 100,000 samples)
- Compression Ratio: 4.56 (on 100,000 samples)
  
## Requirements
- **For using the tokenizer**:
  - Python >= 3.9
  - tokenizers
- **For training the tokenizer**:
  - pandas
  - datasets
  - requests
  - hazm