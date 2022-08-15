from PIL import Image
import pandas as pd
import shutil
import argparse
import csv
import argparse
import json
import os
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorboard.plugins import projector

def images_to_sprite(images, size):
    one_square_size = int(np.ceil(np.sqrt(len(images))))
    master_width = size * one_square_size
    master_height = size * one_square_size
    spriteimage = Image.new(
        mode='RGBA',
        size=(master_width, master_height),
        color=(0, 0, 0, 0)
    )

    for count, image in tqdm(enumerate(images), total=len(images)):
        div, mod = divmod(count, one_square_size)
        h_loc = size * div
        w_loc = size * mod
        image = image.resize((size, size))
        spriteimage.paste(image, (w_loc, h_loc))
    return spriteimage.convert('RGB')

def main():
    THUMBNAIL_SIZE = 32
    typespaces = json.load(open('typespaces.json', 'r'))

    for typename in range(len(typespaces)):
        embedding_space = [k for k, v in typespaces.items() if v == typename]
        embedding_space = embedding_space[0]
        embedding_space_1 = embedding_space.split(',')[0][2:-1]
        embedding_space_2 = embedding_space.split(',')[1][2:-2]
        embeddings = np.load(f'embeddings/embeddings_{embedding_space_1}_{embedding_space_2}_{typename}.npy')
        assert embeddings.shape == (47854, 64), f'{embeddings.shape}'

        projectors_dir = f'projectors/{embedding_space_1}_{embedding_space_2}_{typename}'
        os.makedirs(projectors_dir, exist_ok=True)

        embeddings = tf.Variable(embeddings, name='embeddings')
        checkpoint = tf.train.Checkpoint(embedding=embeddings)
        checkpoint.save(os.path.join(projectors_dir, "embedding.ckpt"))

        shutil.copyfile('embeddings/embeddings_metadata.tsv',
                        os.path.join(projectors_dir, 'metadata.tsv'))
        metadata = pd.read_csv(os.path.join(projectors_dir, 'metadata.tsv'), sep='\t')
        images = []
        for index, row in metadata.iterrows():
            image_path = row['path']
            image = Image.open(image_path)
            image = image.resize((THUMBNAIL_SIZE, THUMBNAIL_SIZE))
            images.append(image)
        sprite_image = images_to_sprite(images, THUMBNAIL_SIZE)
        sprite_image.save(f'{projectors_dir}/sprite.jpg', transparency=0)

        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        # NOTE: Hardcode this line
        embedding.tensor_name = 'embedding/.ATTRIBUTES/VARIABLE_VALUE'
        embedding.metadata_path = 'metadata.tsv'
        embedding.sprite.image_path = 'sprite.jpg'
        embedding.sprite.single_image_dim.extend([THUMBNAIL_SIZE, THUMBNAIL_SIZE])
        projector.visualize_embeddings(projectors_dir, config)

        print("Done processing for {}".format(embedding_space))

if __name__ == "__main__":
    main()
