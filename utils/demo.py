from PIL import Image, ImageOps
import shutil
import pandas as pd
import csv
import argparse
import json
import os
import numpy as np
import torch
from torchvision import transforms as T
from polyvore_outfits import TripletImageLoader
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default='./data/')
    parser.add_argument('--polyvore_split', type=str, default='nondisjoint')
    parser.add_argument('--rand_typespaces', type=bool, default=False)
    parser.add_argument('--num_rand_embed', type=int, default=4)
    args = parser.parse_args()
    return args

def get_typespace(anchor, pair, typespaces):
    """ Returns the index of the type specific embedding
        for the pair of item types provided as input
    """
    query = (anchor, pair)
    if str(query) not in typespaces:
        query = (pair, anchor)

    return typespaces[str(query)]

def load_embedding_for_typespace(root_path, q_type, cond_type, typespace):
    fname = f'embeddings_{q_type}_{cond_type}_{typespace}.npy'
    if not os.path.exists(os.path.join(root_path, fname)):
        fname = f'embeddings_{cond_type}_{q_type}_{typespace}.npy'
    fpath = os.path.join(root_path, fname)
    typespace_embeddings = np.load(fpath)
    return typespace_embeddings

def find_k_nearest_neighbors(metadata, typespace_embeddings, cond_type, anchor_index, k=5):
    # Get index of candidate items for each question
    candidates = metadata[metadata['type'] == cond_type]
    candidate_indexes = candidates.index

    # Get embeddings for anchor
    anchor_embedding = typespace_embeddings[anchor_index]
    candidate_embeddings = typespace_embeddings[candidate_indexes]

    # Calculate Euclidean distance between anchor and candidate items
    distances = np.linalg.norm(candidate_embeddings - anchor_embedding, axis=1)
    top_k_min_distance_indexes = np.argsort(distances)[:k]
    real_index_in_typespace = candidate_indexes[top_k_min_distance_indexes]
    return real_index_in_typespace

def images_to_sprite(images, size):
    one_square_size = int(np.ceil(np.sqrt(len(images))))
    master_width = size * one_square_size
    master_height = size * one_square_size
    spriteimage = Image.new(
        mode='RGBA',
        size=(master_width, master_height),
        color=(0, 0, 0, 0)
    )

    for count, image in enumerate(images):
        div, mod = divmod(count, one_square_size)
        h_loc = size * div
        w_loc = size * mod
        image = image.resize((size, size))
        spriteimage.paste(image, (w_loc, h_loc))
    return spriteimage.convert('RGB')

def save_sprite(output_path, metadata, image_indexes, answer_index=None):
    image_paths = metadata.iloc[image_indexes]['path']
    images = [Image.open(path) for path in image_paths]

    # Add green border to last image
    if answer_index is not None:
        answer_path = metadata.iloc[answer_index]['path'].values
        answer_image = Image.open(answer_path[0])
        img_with_border = ImageOps.expand(answer_image,border=30,fill='green')
        images[-1] = img_with_border

    sprite = images_to_sprite(images, 112)
    sprite.save(output_path)


def main(args):
    K = 5

    fn = os.path.join('data', 'polyvore_outfits', 'polyvore_item_metadata.json')
    meta_data = json.load(open(fn, 'r')) # Metadata for the dataset
    transform = T.Compose([
        T.Resize(112),
        T.CenterCrop(112),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    question_loader = TripletImageLoader(args, 'test', meta_data, transform=transform)
    test_loader = torch.utils.data.DataLoader(question_loader, batch_size=4, shuffle=False)

    embeddings_metadata = pd.read_csv('./embeddings/embeddings_metadata.tsv', sep='\t')
    typespaces = json.load(open('typespaces.json', 'r'))
    output_path = './output/'
    data_path = './data/polyvore_outfits/images/'

    for q_index, (questions, answers, is_correct) in tqdm(enumerate(question_loader), total=len(question_loader)):
        # Get types and indexes of questions
        question_rows = [embeddings_metadata[embeddings_metadata['id'] ==
                                             int(qid[1])]['type'] for qid in
                         questions]
        question_types = [qrow.values[0] for qrow in question_rows]
        question_indexes = [qrow.index[0] for qrow in question_rows]

        # Get type and index of answer
        answer_rows = [embeddings_metadata[embeddings_metadata['id'] ==
                                           int(aid[1])]['type'] for aid in
                       answers]
        answer_type = answer_rows[0].values[0]
        answer_index = answer_rows[0].index

        # Condition is the type of candidate items we want
        # This could be input as an argument
        # Only answers of the this type are considered
        # Set this to answer_type for testing purposes
        condition_type = answer_type

        # For each question, find the nearest neighbors in the typespace
        # has the same type as the condition type
        candidate_indexes = []
        for q_type, q_index in zip(question_types, question_indexes):
            typespace = get_typespace(q_type, condition_type, typespaces)
            typespace_embeddings = load_embedding_for_typespace('./embeddings',
                                                                q_type,
                                                                condition_type,
                                                                typespace)

            closest_indexes = find_k_nearest_neighbors(embeddings_metadata,
                                                       typespace_embeddings,
                                                       condition_type,
                                                       q_index,
                                                       k=K*2)
            candidate_indexes.extend(closest_indexes.values)

        # For earch candidate calculate the distance to the question images
        distances = np.zeros((len(candidate_indexes), ), dtype=np.float32)
        for q_type, q_index in zip(question_types, question_indexes):
            typespace = get_typespace(q_type, condition_type, typespaces)
            typespace_embeddings = load_embedding_for_typespace('./embeddings',
                                                                q_type,
                                                                condition_type,
                                                                typespace)
            anchor_embedding = typespace_embeddings[q_index]
            candidate_embeddings = typespace_embeddings[candidate_indexes]
            distances += np.linalg.norm(candidate_embeddings - anchor_embedding, axis=1)

        # Take the top K closest candidates
        distances = np.array(distances)
        min_distances_indexes = np.argsort(distances)[:K]
        candidate_indexes = list(set(candidate_indexes[i] for i in
                                     min_distances_indexes if i not in
                                     question_indexes))

        # Save the nearest neighbors to a sprite
        q_output_path = os.path.join(output_path, f'{q_index}')
        os.makedirs(q_output_path, exist_ok=True)
        save_sprite(os.path.join(q_output_path, 'i_sprite.jpg'), embeddings_metadata, question_indexes)

        c_output_path = os.path.join(output_path, f'{q_index}')
        os.makedirs(c_output_path, exist_ok=True)
        save_sprite(os.path.join(c_output_path, 'o_sprite.jpg'), embeddings_metadata, candidate_indexes, answer_index=answer_index)



if __name__ == "__main__":
    main(parse_args())