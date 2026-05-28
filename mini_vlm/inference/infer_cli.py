from __future__ import annotations

import argparse

from mini_vlm.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="DINOv3 Mini VLM 추론 CLI")
    parser.add_argument("--config", default="configs/dinov3-mini-vlm.json")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--image", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.dry_run:
        print(f"실험명: {config.experiment_name}")
        print(f"이미지: {args.image}")
        print(f"질문: {args.question}")
        print("dry-run: 실제 모델 추론은 실행하지 않았습니다.")
        return

    run_inference(config_path=args.config, checkpoint=args.checkpoint, image=args.image, question=args.question)


def run_inference(config_path: str, checkpoint: str, image: str, question: str) -> None:
    try:
        import torch
        from PIL import Image
    except ImportError as exc:
        raise SystemExit("실제 추론에는 torch와 pillow가 필요합니다. `pip install '.[model]'` 후 다시 실행하세요.") from exc

    from mini_vlm.models.builder import build_mini_vlm
    from mini_vlm.models.generation import greedy_generate_from_visual_prefix
    from mini_vlm.utils.device import select_torch_device
    from mini_vlm.utils.model_loading import load_checkpoint_into_model

    config = load_config(config_path)
    built = build_mini_vlm(config)
    if checkpoint:
        load_checkpoint_into_model(built.model, config, checkpoint)
    device = select_torch_device(config.device)
    model = built.model.to(device)

    image_obj = Image.open(image).convert("RGB")
    processed = built.image_processor(images=[image_obj], return_tensors="pt")
    pixel_values = processed["pixel_values"] if isinstance(processed, dict) else processed.pixel_values
    prompt = f"Question: {question}\nAnswer:"
    prompt_ids = built.tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    result = greedy_generate_from_visual_prefix(
        model=model,
        tokenizer=built.tokenizer,
        pixel_values=pixel_values.to(device),
        prompt_input_ids=input_ids,
        max_new_tokens=config.max_new_tokens,
        repetition_penalty=config.repetition_penalty,
        no_repeat_ngram_size=config.no_repeat_ngram_size,
        stop_strings=config.stop_strings,
    )
    print(f"질문: {question}")
    print(f"답변: {result.text}")

if __name__ == "__main__":
    main()
