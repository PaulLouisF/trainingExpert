import json
import sys

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ⚡ ONLY CHANGE: quantized faster model
MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ"

app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
)

print("Model loaded")
print("Device map:", getattr(model, "hf_device_map", None))


class ChatRequest(BaseModel):
    message: str
    tables: dict


class ChatResponse(BaseModel):
    response: str


def strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```python"):
        text = text[len("```python"):].strip()
    elif text.startswith("```"):
        text = text[len("```"):].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
@torch.inference_mode()
def chat(payload: ChatRequest) -> ChatResponse:

    messages = [
        {
            "role": "system",
            "content": f"""
    You are a senior data engineer writing Polars code.

    Task:
    Convert natural language questions into correct Polars DataFrame code.

    Rules:
    - Use ONLY provided datasets and columns
    - Never guess columns or tables
    - Output ONLY valid Python code (no markdown, no explanation)
    - Final result MUST be assigned to variable: result

    Core transformation pattern:
    join → filter → compute features → group_by → aggregate → sort

    Important:
    - Use correct join keys
    - Apply filters AFTER joins when needed
    - Compute metrics with .with_columns()
    - Check colums_name

    Example pattern:

    result = (
        nw_order_details
        .join(nw_products, on="product_id")
        .join(nw_categories, on="category_id")
        .filter(pl.col("category_name") == "Seafood")
        .with_columns(
            (pl.col("unit_price") * pl.col("quantity") * (1 - pl.col("discount"))).alias("revenue")
        )
        .join(nw_orders, on="order_id")
        .group_by("customer_id")
        .agg(pl.col("revenue").sum().alias("total_spent"))
        .sort("total_spent", descending=True)
        .head(5)
    )

    Available datasets:
    {json.dumps(payload.tables, ensure_ascii=False)}
    """
        },
        {
            "role": "user",
            "content": payload.message,
        },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt")

    # safe device placement for sharded models
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=False,
    )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    return ChatResponse(response=strip_code_fence(response))