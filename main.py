import json
import sys

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-Coder-3B-Instruct"


app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="auto",
)

print("Model device:", next(model.parameters()).device)


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
    print("Python version:", sys.version)
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())

    if torch.cuda.is_available():
        print("CUDA device name:", torch.cuda.get_device_name(0))

    messages = [
        {
            "role": "system",
            "content": (
                "You are a senior data engineer using Polars.\n\n"

                "Follow this process:\n"
                "1. Identify dataset\n"
                "2. Identify required columns\n"
                "3. Define transformations\n\n"
                
                "Rules:\n"
                "- Use only existing columns\n"
                "- Do not guess\n"
                "- Assign final result to 'result'\n"
                "- No markdown\n\n"

                "Return only valid Python Polars code. "
                "Here is an example of question you will received with the answer expected (gol_code)"
                {
                    "id": "PIP01",
                    "question": "Top 5 customers by total spending on Seafood products. Full chain: order_details → products → categories (filter Seafood) → orders → customers.",
                    "datasets": {
                        "nw_order_details": {
                        "file_name": "data/nw_order_details.parquet",
                        "format": "parquet"
                        },
                        "nw_products": {
                        "file_name": "data/nw_products.parquet",
                        "format": "parquet"
                        },
                        "nw_categories": {
                        "file_name": "data/nw_categories.parquet",
                        "format": "parquet"
                        },
                        "nw_orders": {
                        "file_name": "data/nw_orders.parquet",
                        "format": "parquet"
                        },
                        "nw_customers": {
                        "file_name": "data/nw_customers.parquet",
                        "format": "parquet"
                        }
                    },
                    "gold_code": "result = nw_order_details.join(nw_products, on=\"product_id\").join(nw_categories, on=\"category_id\").filter(pl.col(\"category_name\") == \"Seafood\").with_columns((pl.col(\"unit_price\") * pl.col(\"quantity\") * (1 - pl.col(\"discount\"))).alias(\"revenue\")).group_by(\"order_id\").agg(pl.col(\"revenue\").sum().alias(\"order_rev\")).join(nw_orders, on=\"order_id\").group_by(\"customer_id\").agg(pl.col(\"order_rev\").sum().round(2).alias(\"total_spent\")).sort(\"total_spent\", descending=True).head(5).join(nw_customers, on=\"customer_id\").select(\"company_name\", \"country\", \"total_spent\")"
                }"
                "No markdown fences. "
                "Assign the final Polars DataFrame to result. "
                f"Available datasets: {json.dumps(payload.tables, ensure_ascii=False)}"
            ),
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

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=3000,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    return ChatResponse(response=strip_code_fence(response))