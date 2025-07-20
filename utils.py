def download_blog_dataset():
    url= "https://huggingface.co/datasets/RohanAiLab/persian_blog/resolve/main/blogs.zip"
    output_path= "blogs.zip"
    extract_folder= "blogs"

    if not os.path.exists(output_path):
        print("Downloading blogs.zip...")
        r= requests.get(url, stream= True)
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("Download completed.")

    if not os.path.exists(extract_folder):
        print("Extracting blogs.zip...")
        with ZipFile(output_path, 'r') as zip_ref:
            zip_ref.extractall(extract_folder)
        print("Extraction completed.")


def eda_dataset(df, name):
    text_column= "text"
    print(f"--- EDA for {name} ---")
    print(f"تعداد نمونه‌ها: {len(df)}")
    print(f"تعداد نمونه‌های خالی: {df[text_column].isna().sum() + (df[text_column].str.strip() == '').sum()}")
    print(f"میانگین طول متن: {df[text_column].str.len().mean():.2f}")
    print(f"حداکثر طول متن: {df[text_column].str.len().max()}")
    print(f"حداقل طول متن: {df[text_column].str.len().min()}")
    # print(f"تعداد کلمات منحصربه‌فرد: {len(set(' '.join(df[text_column].dropna().astype(str)).split()))}")
    # print(f"نمونه متن: {df[text_column].iloc[0][:100] if isinstance(df[text_column].iloc[0], str) else 'غیرمعتبر'}")
    print("\n")


def show_short_samples(df, dataset_name, max_length= 1000, num_samples= 5):
    print(f"--- نمونه‌های با طول کمتر از {max_length} کاراکتر برای {dataset_name} ---")
    short_samples = df[df['text'].str.len() < max_length]
    print(f"تعداد نمونه‌های با طول کمتر از {max_length} کاراکتر: {len(short_samples)}")
    if len(short_samples) > 0:
        for i, text in enumerate(short_samples['text'].head(num_samples)):
            print(f"نمونه {i+1}: {text[:100]} (طول: {len(text)} کاراکتر)")
    else:
        print("هیچ نمونه‌ای با طول کمتر از {max_length} کاراکتر یافت نشد.")
    print("\n")