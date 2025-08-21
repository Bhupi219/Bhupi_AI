from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/home/bhupendra_singh/Models/Qwen2-7B-Instruct"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load model (in float16 to save VRAM)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype="auto"
)

# Test inference
inputs = tokenizer("Hello, how are you?", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
