from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import open_clip
import torch
from PIL import Image
from transformers import LlamaForCausalLM, AutoTokenizer


# Load CLIP model and preprocess function
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', device=device)
clip_model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active

# Load LLaMA model and tokenizer
llama_model_name = "akjindal53244/Llama-3.1-Storm-8B"  # Replace with the correct LLaMA model path
llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_name)
llama_model = LlamaForCausalLM.from_pretrained(llama_model_name, device_map="auto")

# Step 1: Encode Image with CLIP
def encode_image(image_path, model, preprocess):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image).cpu().numpy()
    return image_features

# Step 2: Encode Text with CLIP
def encode_text(text, model):
    text_tokens = open_clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens).cpu().numpy()
    return text_features



# Step 4: Integrate with LLaMA for a conversational response
def generate_response_with_image_and_text(query, metadata, image_description):
    prompt = f"""
    You are a shopping assistant. A customer asked: "{query}"

    Based on the information, here is the product description:
    
    - Metadata: {metadata}
    - Image Description: {image_description}
    
    Provide a useful response based on this information:
    """
    
    inputs = llama_tokenizer(prompt, return_tensors="pt").to(device)
    outputs = llama_model.generate(inputs['input_ids'], max_length=200, num_beams=5, early_stopping=True)
    response = llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


