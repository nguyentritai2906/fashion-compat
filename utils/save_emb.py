import csv
import argparse
import json
import os
import numpy as np
import torch
from torchvision import transforms as T
from polyvore_outfits import TripletImageLoader
from tqdm import tqdm

transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(256),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load('./extra_dense_encoder.pt')
model.eval()
model.to(device)

parser = argparse.ArgumentParser()
parser.add_argument('--datadir', type=str, default='./data/')
parser.add_argument('--polyvore_split', type=str, default='nondisjoint')
parser.add_argument('--rand_typespaces', type=bool, default=False)
parser.add_argument('--num_rand_embed', type=int, default=4)
args = parser.parse_args()

fn = os.path.join('data', 'polyvore_outfits', 'polyvore_item_metadata.json')
meta_data = json.load(open(fn, 'r'))

image_loader = TripletImageLoader(args, 'test', meta_data, transform=transform)
test_loader = torch.utils.data.DataLoader(image_loader, batch_size=4, shuffle=False)

embeddings = []
item_types = []
item_ids = []
item_paths = []

print('Loading embeddings')
for input_images, input_item_types, input_item_ids, input_paths in tqdm(test_loader):
    input_images = input_images.to(device)
    # input_images = input_images.unsqueeze(0).to(device)
    input_images = torch.permute(input_images, (0, 2, 3, 1))
    embedding = model(input_images).data
    embeddings.append(embedding)
    item_types.extend(input_item_types)
    item_ids.extend(input_item_ids)
    item_paths.extend(input_paths)
embeddings = torch.cat(embeddings)

os.makedirs('./embeddings', exist_ok=True)
np.save('./embeddings/embeddings.npy', embeddings.cpu().numpy())
print('Embeddings saved')

tsv_handler = open('./embeddings/embeddings_metadata.tsv', 'w')
tsv_writer = csv.writer(tsv_handler, delimiter='\t')
tsv_writer.writerow(['id', 'type', 'path'])

print('Writing tsv')
for item_id, item_type, item_path in tqdm(zip(item_ids, item_types, item_paths)):
    tsv_writer.writerow([item_id, item_type, item_path])
tsv_handler.close()

for type_pair, index in image_loader.typespaces.items():
    specific_type_embeddings = embeddings[:, index]
    # Save as npy
    np.save(f'./embeddings/embeddings_{type_pair[0]}_{type_pair[1]}_{index}.npy',
            specific_type_embeddings.cpu().numpy())
    print(f'Saved embeddings for {type_pair[0]}_{type_pair[1]}_{index}')
