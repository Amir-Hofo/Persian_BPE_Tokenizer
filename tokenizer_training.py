from packages import *
    
def tokenizer_training_fn(dataset: pd.DataFrame, vocab_size= 30_000, min_frequency= 5, pre_tokenizer= True)-> Tokenizer:
    unk, eos= "[UNK]", "<|endoftext|>"
    tokenizer= Tokenizer(models.BPE(unk_token= unk))
    if pre_tokenizer: tokenizer.pre_tokenizer= pre_tokenizers.Whitespace()
    trainer= trainers.BpeTrainer(vocab_size= vocab_size, min_frequency= min_frequency,
                                special_tokens= [unk, eos] )

    tokenizer.train_from_iterator(dataset["text"], trainer)
    print(10 * "--", " vocab size ", 10 * "--")
    print(tokenizer.get_vocab_size())

    tokenizer.post_processor= processors.TemplateProcessing(
        single= f"{eos} $A {eos}",
        special_tokens= [(eos, tokenizer.token_to_id(eos))]
        )
    tokenizer.decoder= decoders.BPEDecoder()


    tokenizer.save(f"Persian_BPE_Tokenizer_{vocab_size//1000}K.json")
    print(70*"-"), print("tokenizer training is complete and saved."), print(70*"-")

    return tokenizer