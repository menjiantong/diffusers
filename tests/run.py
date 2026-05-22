import torch
# 修正 1：导入正确的 Pipeline
from diffusers import OvisImagePipeline

# 设置环境变量后，按官方示例加载模型
pipe = OvisImagePipeline.from_pretrained(
    "/root/autodl-tmp/.cache/modelscope/models/AIDC-AI/Ovis-Image-7B",
    torch_dtype=torch.bfloat16 # 修正 2：使用 torch_dtype
)
pipe.to("cuda")

# 修正 3：按官方推荐参数生成
prompt = "a cup of coffee on the table"
image = pipe(
    prompt,
    num_inference_steps=50,
    guidance_scale=5.0
).images[0]

image.save("coffee.png")