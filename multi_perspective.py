import argparse
import json
import os
from pathlib import Path
import re
import tempfile


MODEL_PATH = str(Path(__file__).resolve().parent.parent / "llama3.2-vision")

FORGED_PROMPT = (
    "The image is predicted to be manipulated by the pre-trained image forgery detector. Please zoom in on all details within the image, paying close attention to object edges to spot unnatural outlines and distorted perspective, inconsistent logical content, unnatural lighting and shadow contrasts, and other flaws. Provide a detailed description within 150 words."
)

ORIGINAL_PROMPT = (
    "Please describe the content of this picture in detail. Cover the main subject, setting, any notable objects or actions, as well as the overall mood or atmosphere. Provide a detailed description within 150 words."
)

PERSPECTIVES = {
    "cs": {
        "use_image": False,
        "instruction": (
            "Focus on factual correctness. Extract the main event, entity, claim, or situation described in the post, and judge whether it is plausible or consistent with real-world knowledge and factual evidence."
        ),
    },
    "td": {
        "use_image": False,
        "instruction": (
            "Focus on emotional tone and possible emotional manipulation. Analyze whether the post uses sensational, exaggerated, provocative, fear-inducing, or sympathy-driven language to mislead readers. Do not judge the news as fake solely because it is emotional; use emotional cues as evidence for credibility assessment."
        ),
    },
    "img": {
        "use_image": True,
        "instruction": (
            "Focus on image-text consistency. Analyze whether the image semantically supports the textual claim, whether there is a mismatch between the visual content and the described event, and whether the image appears to be used out of context. Do not focus primarily on factual verification beyond what can be inferred from the image-text pair."
        ),
    },
}


def parse_response(response):
    label_match = re.match(r"\s*This\s+news\s+is\s+(fake|real)\b", response, re.IGNORECASE)
    if label_match is None:
        return 2, response

    reason_match = re.fullmatch(
        r"\s*\.?\s*Reason\s*:\s*(.+)",
        response[label_match.end():],
        flags=re.IGNORECASE | re.DOTALL,
    )
    if reason_match is None or not reason_match.group(1).strip():
        return 2, response
    pred = 1 if label_match.group(1).lower() == "fake" else 0
    return pred, reason_match.group(1).strip()


def build_user_prompt(content, perspective):
    cfg = PERSPECTIVES[perspective]
    image_line = "\nNews image: [provided image]." if cfg["use_image"] else ""
    return (
        "You are a professional multimodal news fact-checker.  You will be given the news content and its associated image. Your task is to determine whether the news is real or fake from the specified analytical perspective.\n\n"
        f"[Perspective Instruction]\n{cfg['instruction']}\n\n"
        f"News text content: {content}{image_line}\n\n"
        "Output format:\n"
        "This news is fake. Reason: <provide a concise reason within 150 words>\n"
        "or\n"
        "This news is real. Reason: <provide a concise reason within 150 words>\n\n"
        "The first sentence must be exactly either \"This news is fake.\" or \"This news is real.\" "
        "Do not use other labels such as True, False, Uncertain, Misleading, or Cannot determine. "
        "The reason must follow the specified perspective."
    )


def load_model_and_processor(model_path, device):
    import torch
    from transformers import AutoProcessor, MllamaForConditionalGeneration

    dtype = torch.bfloat16 if torch.device(device).type == "cuda" else torch.float32
    model = MllamaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map={"": device},
        local_files_only=True,
    )
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model.eval()
    return model, processor


def generate_text(model, processor, user_prompt, image, max_new_tokens):
    import torch

    message_content = [{"type": "image"}] if image is not None else []
    message_content.append({"type": "text", "text": user_prompt})
    chat_text = processor.apply_chat_template(
        [{"role": "user", "content": message_content}], add_generation_prompt=True
    )
    processor_kwargs = dict(text=chat_text, add_special_tokens=False, return_tensors="pt")
    if image is not None:
        processor_kwargs["images"] = image
    inputs = processor(**processor_kwargs)
    device = next(model.parameters()).device
    inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}

    tokenizer = processor.tokenizer
    eos_ids = []
    if tokenizer.eos_token_id is not None:
        eos_ids.append(tokenizer.eos_token_id)
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if isinstance(eot_id, int) and eot_id >= 0 and eot_id != tokenizer.unk_token_id and eot_id not in eos_ids:
        eos_ids.append(eot_id)
    generation_kwargs = dict(max_new_tokens=max_new_tokens, use_cache=True)
    if eos_ids:
        generation_kwargs.update(eos_token_id=eos_ids, pad_token_id=eos_ids[0])

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_kwargs)
    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    return processor.decode(new_tokens, skip_special_tokens=True)


def generate_response(model, processor, content, perspective, image, max_new_tokens):
    return generate_text(
        model, processor, build_user_prompt(content, perspective),
        image if PERSPECTIVES[perspective]["use_image"] else None, max_new_tokens
    )


def get_description_prompt(forgery):
    if type(forgery) is not int or forgery not in (0, 1):
        raise ValueError("forgery must be the integer 0 (original) or 1 (forged)")
    return ORIGINAL_PROMPT if forgery == 0 else FORGED_PROMPT


def infer_one_item(item, model, processor, input_dir, max_new_tokens):
    from PIL import Image

    description_prompt = get_description_prompt(item.get("forgery"))
    image_path = Path(item["image_path"]).expanduser()
    if not image_path.is_absolute():
        image_path = input_dir / image_path
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    result = dict(item)
    try:
        for perspective in PERSPECTIVES:
            response = generate_response(
                model, processor, str(item["content"]), perspective, image, max_new_tokens
            )
            pred, rationale = parse_response(response)
            result[f"{perspective}_pred"] = pred
            result[f"{perspective}_rationale"] = rationale
        result["forgery_description"] = generate_text(
            model, processor, description_prompt, image, max_new_tokens
        ).strip()
    finally:
        image.close()
    return result


def save_json_in_place(path, data):
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False) as handle:
            temporary_path = Path(handle.name)
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.chmod(path.stat().st_mode & 0o777)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Generate three perspectives and an image description; update the input JSON in place.")
    parser.add_argument("--input_json", required=True, help="Input JSON, include content, image_path and forgery (0 or 1); original fields including label are preserved")
    parser.add_argument("--model_path", default=MODEL_PATH)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("max_new_tokens must be greater than 0")

    input_path = Path(args.input_json).expanduser().resolve()
    with input_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        parser.error("The input JSON must be an array of records.")
    for index, item in enumerate(data):
        if not isinstance(item, dict) or "content" not in item or not item.get("image_path"):
            parser.error(f"Record {index + 1} is missing content or image_path")
        try:
            get_description_prompt(item.get("forgery"))
        except ValueError as error:
            parser.error(f"Record {index + 1}: {error}")

    results = []
    if data:
        from tqdm import tqdm

        model, processor = load_model_and_processor(args.model_path, args.device)
        for item in tqdm(data, desc="Multi-perspective inference", unit="sample"):
            results.append(infer_one_item(item, model, processor, input_path.parent, args.max_new_tokens))

    save_json_in_place(input_path, results)
    print(f"Updated {len(results)} records in {input_path}")


if __name__ == "__main__":
    main()
