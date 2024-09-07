import open_clip
import torch

from generator import generate_response_with_image_and_text, encode_text
from preprocess_data import image_to_embedding, query_index, load_images_by_barcode, build_index


# Load CLIP model and preprocess function
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', device=device)
clip_model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active


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
    query_embedding = image_to_embedding(image_path, clip_model, preprocess, device)

    top_barcodes, distances = query_index(query_embedding, index, mapping, k=5)

    print("Top matching barcodes:", top_barcodes)
    print("Distances:", distances)

    text_query = "What is this product?"
    metadata = "This is a state-of-the-art device with advanced features."

    # Step 3: Retrieve similar images (e.g., using a vector database like FAISS)
    # Here we simply assume image_features are already encoded and available.
    text_features = encode_text(text_query, clip_model)

    # You can implement a similarity search here using FAISS or other retrieval mechanisms

    # Example of image description generated from the CLIP model
    image_description = "The image shows a sleek, modern device with a minimalistic design."

    # Generate the conversational response
    response = generate_response_with_image_and_text(text_query, metadata, image_description)

    print(response)