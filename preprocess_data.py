import pandas as pd
import os
from PIL import Image
import open_clip
import torch
import faiss
import numpy as np
from typing import Dict


# Load CLIP model and preprocess function
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', device=device)
model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
tokenizer = open_clip.get_tokenizer('ViT-B-32')


def image_to_embedding(image_path, model, preprocess, device):
    """
    Function to convert an image to a CLIP embedding
    """
    # Load and preprocess the image
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    # Encode the image to get the embedding
    with torch.no_grad():
        image_embedding = model.encode_image(image).cpu().numpy()

    return image_embedding


def load_images_by_barcode(csv_path: str, images_path: str) -> Dict:
    """
    Load all images for which 
    """
    # Load the CSV file
    df = pd.read_csv(csv_path)

    image_dict = {}

    # Extract image embeddings for each barcode
    for index, row in df.iterrows():
        barcode = row['GTIN_ID']
        image_folder = os.path.join(images_path, str(barcode))
        if os.path.exists(image_folder):
            for image_file in os.listdir(image_folder):
                image_path = os.path.join(image_folder, image_file)
                try:
                    # Preprocess the image and create an embedding
                    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
                    with torch.no_grad():
                        image_features = model.encode_image(image).cpu().numpy()

                    # If the barcode already exists in the image_dict, append the image features to the list
                    if barcode in image_dict:
                        image_dict[barcode].append(image_features)
                    else:
                        # Otherwise, initialize a new list for this barcode
                        image_dict[barcode] = [image_features]
            
                except Exception as e:
                    print(f"Error processing image {image_path}: {e}")

    return image_dict, df


def build_index(image_dict: Dict):
    """
    Build a FAISS index from the image embeddings.
    Map the multiple embeddings to their associated barcode.

    all_embeddings is a matrix with shape (num_embeddings, embedding_dimension)
    barcode_mapping is a list that maps each row in all_embeddings to a specific barcode
    """
    # Initialize a list to store all embeddings and corresponding barcodes
    all_embeddings = []
    barcode_mapping = []  # This will map each embedding to its corresponding barcode

    # Loop through the image_dict to extract all image embeddings and store them
    for barcode, embeddings_list in image_dict.items():
        for embedding in embeddings_list:
            all_embeddings.append(embedding)  # Add the embedding to the list
            barcode_mapping.append(barcode)   # Add the corresponding barcode

    # Convert the list of embeddings to a numpy array for FAISS
    all_embeddings = np.vstack(all_embeddings)  # Convert list of arrays to a 2D array

    # Build the FAISS index
    dimension = all_embeddings.shape[1]  # Embedding dimension size from CLIP
    index = faiss.IndexFlatL2(dimension)  # Use L2 distance for similarity search
    index.add(all_embeddings)  # Add the embeddings to the index

    # Assuming index is already built
    gpu_resources = faiss.StandardGpuResources()  # Create GPU resources
    index_gpu = faiss.index_cpu_to_gpu(gpu_resources, 0, index)  # Transfer index to GPU (0 is the GPU ID)
    
    return index_gpu, barcode_mapping, dimension


def query_index(query_embedding, index, barcode_mapping, k=5):
    """
    Query the FAISS index with a given query embedding and return the top k matching barcodes.
    
    Args:
    - query_embedding: The query embedding (numpy array).
    - index: The FAISS index.
    - barcode_mapping: The list mapping embeddings to barcodes.
    - k: The number of top results to retrieve.
    
    Returns:
    - List of top k matching barcodes.
    """
    # Perform the search (retrieve the k nearest neighbors)
    distances, indices = index.search(query_embedding, k)

    # Retrieve the corresponding barcodes using the indices
    matching_barcodes = [barcode_mapping[i] for i in indices[0]]

    return matching_barcodes, distances


if __name__ == "__main__":
    # load the images based on csv and available files
    data, df = load_images_by_barcode(csv_path='data/bg_master_data.csv', images_path='data/images')

    # load the index, index to barcode mapping and get the dimension for embeddings
    index, mapping, dimension = build_index(image_dict=data)

    # Example usage
    #query_embedding = np.random.rand(1, dimension).astype('float32')  # Replace with actual query embedding

    # "\\wsl.localhost\Ubuntu\home\ubuntu\repos\schwarzit.scpoc-api\detection-api\src\app\images\small_\21.jpg"

    # Load and process your image from file (replace 'path_to_image.jpg' with the actual file path)
    image_path = 'data/21.jpg'
    query_embedding = image_to_embedding(image_path, model, preprocess, device)

    top_barcodes, distances = query_index(query_embedding, index, mapping, k=5)

    print("Top matching barcodes:", top_barcodes)
    print("Distances:", distances)
    #print(data)