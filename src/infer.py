# infer.py
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import timm
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import joblib
import re
import requests
from time import sleep
from concurrent.futures import ThreadPoolExecutor
import scipy.sparse

def download_images(image_links, out_dir, concurrency=5):
    def download_single_image(url, path):
        try:
            response = requests.get(url, stream=True, timeout=10)
            response.raise_for_status()
            with open(path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        except Exception:
            return False

    os.makedirs(out_dir, exist_ok=True)
    paths = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []
        for i, url in enumerate(image_links):
            path = os.path.join(out_dir, f'{i}.jpg')
            paths.append(path)
            futures.append(executor.submit(download_single_image, url, path))
            sleep(0.1)
        for future in tqdm(futures):
            future.result()
    return paths

def extract_text_features(df):
    df['text'] = df['catalog_content'].fillna('')
    df['text_len'] = df['text'].str.len()
    df['word_count'] = df['text'].str.split().str.len()
    df['digit_count'] = df['text'].str.findall(r'\d').str.len()
    df['punct_count'] = df['text'].str.findall(r'[\.,;:!?]').str.len()
    df['avg_token_len'] = df['text'].apply(lambda x: np.mean([len(w) for w in x.split()]) if len(x.split()) > 0 else 0)

    pack_patterns = r'(pack of \d+|\d+ pack|\d+ count|\d+ ct|\d+pk|\d+ pk)'
    df['pack_count'] = df['text'].str.findall(pack_patterns, flags=re.I).str.len() + df['text'].str.count(r'\d+') // 2
    df['pack_count'] = df['pack_count'].clip(0, 50)

    unit_patterns = r'(\d+\.?\d*)\s*(oz|ounce|fl oz|g|gram|kg|lb|pound|ml|l|liter|ct|count|pack)'
    df['unit_qty'] = df['text'].str.extract(unit_patterns, flags=re.I)[0].astype(float).fillna(0)
    df['unit'] = df['text'].str.extract(unit_patterns, flags=re.I)[1].fillna('unknown')

    def extract_brand(text):
        tokens = re.findall(r'\b[A-Z][a-zA-Z]{2,}\b', text)
        if tokens:
            return tokens[0]
        return 'unknown'
    df['brand'] = df['text'].apply(extract_brand)
    brand_freq = df['brand'].value_counts()
    top_brands = brand_freq.head(100).index
    df['brand'] = df['brand'].apply(lambda x: x if x in top_brands else 'other')

    return df

def tfidf_ridge_meta(X_text, vectorizers):
    tfidf_word, tfidf_char, ridge = vectorizers
    X_tfidf = scipy.sparse.hstack([tfidf_word.transform(X_text), tfidf_char.transform(X_text)])
    return ridge.predict(X_tfidf)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = timm.create_model('resnet18.a1_in1k', pretrained=True, num_classes=0).to(device)
model.eval()
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class ImageDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert('RGB')
            img = transform(img)
            return img, 0
        except:
            return torch.zeros(3, 224, 224), 1

def extract_image_embeddings(paths, batch_size=32):
    dataset = ImageDataset(paths)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=False)
    embeddings = []
    missing = []
    with torch.no_grad():
        for imgs, flags in tqdm(loader):
            imgs = imgs.to(device)
            emb = model(imgs)
            embeddings.append(emb.cpu().numpy())
            missing.extend(flags.numpy())
    embeddings = np.vstack(embeddings)
    embeddings[np.array(missing) == 1] = 0
    return embeddings, np.array(missing)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', required=True)
    parser.add_argument('--images', default='./images/test')
    parser.add_argument('--model', default='model/')
    parser.add_argument('--out', default='test_out.csv')
    args = parser.parse_args()

    df = pd.read_csv(args.test)
    paths = download_images(df['image_link'], args.images)

    model_a, model_b, model_c, weights, vectorizers, le_unit, le_brand, iso, num_feats, cat_feats = joblib.load(os.path.join(args.model, 'models.pkl'))

    df = extract_text_features(df)
    df['unit'] = le_unit.transform(df['unit'].fillna('unknown'))
    df['brand'] = le_brand.transform(df['brand'].fillna('other'))

    df['ridge_meta'] = tfidf_ridge_meta(df['text'], vectorizers)

    emb, missing = extract_image_embeddings(paths)
    df['image_missing'] = missing

    X_num = df[num_feats].values
    X_cat = df[cat_feats].values
    X = pd.DataFrame(np.hstack([X_num, X_cat]), columns=num_feats + cat_feats)

    pred_a = model_a.predict(X)
    pred_b = model_b.predict(X)
    pred_c = model_c.predict(emb)

    test_ens = weights[0] * pred_a + weights[1] * pred_b + weights[2] * pred_c
    test_pred = np.expm1(test_ens)
    test_pred = np.clip(test_pred, 0.01, None)
    # Assume train p1 p99 ~0.5,100 for clip
    test_pred = np.clip(test_pred, 0.5, 100)
    test_cal = iso.predict(test_pred)

    out_df = pd.DataFrame({'sample_id': df['sample_id'], 'price': test_cal})
    out_df.to_csv(args.out, index=False)