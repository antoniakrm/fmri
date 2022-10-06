from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation, SegformerModel
from PIL import Image
import torch
import requests
import os
import numpy as np

model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
# model_name = "nvidia/mit-b0"
# model_name = 'nvidia/segformer-b5-finetuned-ade-640-640'

feature_extractor = SegformerFeatureExtractor.from_pretrained(model_name, \
        cache_dir=os.path.expanduser(f"~/.cache/huggingface/transformers/models/{model_name}"), output_hidden_states=True)

model = SegformerModel.from_pretrained(model_name, \
    cache_dir=os.path.expanduser(f"~/.cache/huggingface/transformers/models/{model_name}"), \
    output_hidden_states=True, return_dict=True)

model.eval()

p_nums = sum(param.numel() for param in model.parameters())
print("Number of parameter: %.2fM" % (p_nums/1e6))
print()
# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image1 = Image.open(requests.get(url, stream=True).raw)
# image1 = Image.open(os.path.expanduser('~/Dir/datasets/imagenet_21k_small/n12598027/n12598027_1315.JPEG')).convert('RGB')
# image2 = Image.open(os.path.expanduser('~/Dir/datasets/imagenet_21k_small/n12598027/n12598027_1620.JPEG')).convert('RGB')

# inputs = feature_extractor(images=[image1, image2, image1, image2, image2], return_tensors="pt")
# # print(inputs.shape)
# with torch.no_grad():
#     outputs = model(**inputs)
# # # logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)
# # hidden = outputs.hidden_states[-1]
# hidden_1 = outputs.last_hidden_state
# print(hidden_1.shape)
# hidden_avg = np.mean(hidden.numpy(), axis=(2,3), keepdims=True)
# # hidden_avg2 = torch.mean(hidden, (2,3), True)
# test_list = [hidden_avg,hidden_avg]
# test_np = np.concatenate(test_list).squeeze()
# # test_torch = torch.cat(test_list).squeeze()
# print(hidden_avg.shape)
# # print(hidden_avg.shape)
# print(model)