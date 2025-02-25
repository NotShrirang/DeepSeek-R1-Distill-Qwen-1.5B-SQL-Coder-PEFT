# DeepSeek R1 Distill Qwen 1.5B finetuned for SQL query generation

[HuggingFace Repo](https://huggingface.co/NotShrirang/DeepSeek-R1-Distill-Qwen-1.5B-SQL-Coder-PEFT)

This model is a fine-tuned version of DeepSeek R1 Distill Qwen 1.5B, specifically optimized for SQL query generation. It has been trained on the GretelAI Synthetic Text-to-SQL dataset to enhance its ability to convert natural language prompts into accurate SQL queries.

Due to its lightweight architecture, this model can be deployed efficiently on local machines without requiring a GPU, making it ideal for on-premises inference in resource-constrained environments. It offers a balance between performance and efficiency, making it suitable for businesses and developers looking for a cost-effective SQL generation solution.

## Training Methodology
1. Fine-tuning approach: LoRA (Low-Rank Adaptation) for efficient parameter tuning.
2. Precision: bfloat16 (bf16) to reduce memory consumption while maintaining numerical stability.
3. Gradient Accumulation: Used to handle larger batch sizes within GPU memory limits.
4. Optimizer: AdamW with learning rate scheduling.
5. Cosine Scheduler: Used cosine learning rate scheduler for training stability. (500 warm-up steps, 2000 steps for the cosine schedule.)

## Use Cases
1. Assisting developers and analysts in writing SQL queries.
2. Automating SQL query generation from user prompts in chatbots.
3. Enhancing SQL-based retrieval-augmented generation (RAG) systems.

## Limitations & Considerations
1. The model may generate incorrect or suboptimal SQL queries for complex database schemas.
2. It does not perform schema reasoning and requires clear table/column references in the input.
3. Further fine-tuning on domain-specific SQL data may be required for better accuracy.

## How to Use
You can load the model using 🤗 Transformers:

```python
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch

model = AutoPeftModelForCausalLM.from_pretrained("NotShrirang/DeepSeek-R1-Distill-Qwen-1.5B-SQL-Coder-PEFT")
tokenizer = AutoTokenizer.from_pretrained("NotShrirang/DeepSeek-R1-Distill-Qwen-1.5B-SQL-Coder-PEFT")

prompt = "Write a SQL query to get the total revenue from the sales table."
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Training Details

- **Total Steps:** 25,000
- **Batch Size:** 4
- **Optimizer:** AdamW
- **Learning Rate:** 5e-5

### Training and Validation Loss Progression

| Step  | Training Loss | Validation Loss |
|-------|--------------|----------------|
| 1000  | 1.0017       | 1.0256         |
| 2000  | 1.1644       | 0.8818         |
| 3000  | 0.7851       | 0.8507         |
| 4000  | 0.7416       | 0.8322         |
| 5000  | 0.6960       | 0.8184         |
| 6000  | 1.0118       | 0.8068         |
| 7000  | 0.9897       | 0.7997         |
| 8000  | 0.9165       | 0.7938         |
| 9000  | 0.8048       | 0.7875         |
| 10000 | 0.8869       | 0.7822         |
| 11000 | 0.8387       | 0.7788         |
| 12000 | 0.8117       | 0.7746         |
| 13000 | 0.7259       | 0.7719         |
| 14000 | 0.8100       | 0.7678         |
| 15000 | 0.6901       | 0.7626         |
| 16000 | 0.9630       | 0.7600         |
| 17000 | 0.6599       | 0.7571         |
| 18000 | 0.6770       | 0.7541         |
| 19000 | 0.7360       | 0.7509         |
| 20000 | 0.7170       | 0.7458         |
| 21000 | 0.7993       | 0.7446         |
| 22000 | 0.5846       | 0.7412         |
| 23000 | 0.8269       | 0.7411         |
| 24000 | 0.5817       | 0.7379         |
| 25000 | 0.5772       | 0.7357         |

- **Developed by:** [NotShrirang](https://huggingface.co/NotShrirang)
- **Language(s) (NLP):** [en]
- **License:** [apache-2.0]
- **Finetuned from model :** [deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)
