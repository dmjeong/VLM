from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from mini_vlm.config import MiniVlmConfig, load_config
from mini_vlm.utils.checkpoints import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Q-Former image-text contrastive 사전정렬 학습")
    parser.add_argument("--config", default="configs/dinov3-local-vits16-qwen-qformer-distilbert-itc-smoke.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.dry_run:
        print(f"실험명: {config.experiment_name}")
        print(f"Vision: {config.vision_model_id}")
        print(f"Q-Former text encoder: {config.qformer_text_model_id}")
        print(f"contrastive_dim: {config.contrastive_dim}")
        print("dry-run: 실제 ITC 학습은 실행하지 않았습니다.")
        return

    pretrain_qformer_itc(config)


@dataclass(frozen=True)
class ItcBatch:
    pixel_values: object
    input_ids: object
    attention_mask: object
    sample_ids: list[str]
    texts: list[str]


class QFormerItcCollator:
    """JSONL VLM sample을 ITC 학습 batch로 변환한다.

    의도: 기존 sample의 `answer` 또는 `question + answer`를 이미지와 맞는 텍스트로 사용한다.
    참고: ITC는 batch 안의 다른 텍스트를 negative로 쓰므로 batch size가 2 이상이어야 의미가 있다.
    """

    def __init__(self, *, image_processor, text_tokenizer, image_root: str | Path, max_text_length: int) -> None:
        self.image_processor = image_processor
        self.text_tokenizer = text_tokenizer
        self.image_root = Path(image_root)
        self.max_text_length = max_text_length

    def __call__(self, samples: Sequence) -> ItcBatch:
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("이미지 처리에는 pillow가 필요합니다.") from exc

        image_paths = [self._resolve_image_path(sample.image) for sample in samples]
        images = [Image.open(path).convert("RGB") for path in image_paths]
        processed_images = self.image_processor(images=images, return_tensors="pt")
        pixel_values = (
            processed_images["pixel_values"] if isinstance(processed_images, dict) else processed_images.pixel_values
        )
        texts = [build_itc_text(sample) for sample in samples]
        tokenized = self.text_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )
        return ItcBatch(
            pixel_values=pixel_values,
            input_ids=tokenized["input_ids"],
            attention_mask=tokenized["attention_mask"],
            sample_ids=[sample.sample_id for sample in samples],
            texts=texts,
        )

    def _resolve_image_path(self, image: str) -> Path:
        path = Path(image)
        if path.is_absolute():
            return path
        return self.image_root / path


def pretrain_qformer_itc(config_path_or_object: str | Path | MiniVlmConfig) -> None:
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise SystemExit("Q-Former ITC 학습에는 torch가 필요합니다. `pip install '.[model]'` 후 다시 실행하세요.") from exc

    from transformers import AutoConfig, AutoImageProcessor, AutoModel, AutoTokenizer

    from mini_vlm.data import MiniVlmDataset
    from mini_vlm.models.builder import build_visual_adapter
    from mini_vlm.models.qformer import QFormerItcModel, initialize_qformer_from_distilbert
    from mini_vlm.models.vision_encoder import DinoVisionEncoder, LocalDinov3ImageProcessor
    from mini_vlm.utils.device import select_torch_device
    from mini_vlm.utils.seed import set_seed

    config = load_config(config_path_or_object) if isinstance(config_path_or_object, (str, Path)) else config_path_or_object
    if config.adapter_type != "qformer":
        raise ValueError("ITC 사전정렬은 adapter_type='qformer' 설정에서만 실행합니다.")
    set_seed(config.seed)
    device = select_torch_device(config.device)

    if config.vision_backend == "hf":
        image_processor = AutoImageProcessor.from_pretrained(config.vision_model_id)
    else:
        image_processor = LocalDinov3ImageProcessor(config.vision_image_size)
    text_tokenizer = AutoTokenizer.from_pretrained(config.qformer_text_model_id)
    text_encoder = AutoModel.from_pretrained(config.qformer_text_model_id)
    text_hidden_dim = int(getattr(text_encoder.config, "hidden_size"))

    vision_encoder = DinoVisionEncoder(
        config.vision_model_id,
        freeze=config.freeze_vision,
        backend=config.vision_backend,
        repo_dir=config.vision_repo_dir,
        model_name=config.vision_model_name,
        weights=config.vision_weights,
    )
    llm_dim = resolve_llm_hidden_dim(config.llm_model_id, AutoConfig)
    visual_adapter = build_visual_adapter(config, vision_dim=int(vision_encoder.hidden_size), llm_dim=llm_dim)
    initialized_layers = 0
    if config.qformer_init_from_text:
        initialized_layers = initialize_qformer_from_distilbert(visual_adapter, text_encoder)
    model = QFormerItcModel(
        vision_encoder=vision_encoder,
        visual_adapter=visual_adapter,
        text_encoder=text_encoder,
        text_hidden_dim=text_hidden_dim,
        contrastive_dim=config.contrastive_dim,
        freeze_vision=config.freeze_vision,
        freeze_text_encoder=True,
    ).to(device)

    train_dataset = MiniVlmDataset(config.train_jsonl, image_root=config.image_root)
    validation_loader = None
    validation_path = Path(config.validation_jsonl)
    collator = QFormerItcCollator(
        image_processor=image_processor,
        text_tokenizer=text_tokenizer,
        image_root=config.image_root,
        max_text_length=config.max_text_length,
    )
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        collate_fn=collator,
        generator=generator,
    )
    if config.validation_jsonl and validation_path.exists():
        validation_dataset = MiniVlmDataset(config.validation_jsonl, image_root=config.image_root)
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=config.train_batch_size,
            shuffle=False,
            collate_fn=collator,
        )

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=config.learning_rate)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "itc_metrics.jsonl"
    summary_path = output_dir / "itc_summary.json"
    if metrics_path.exists():
        metrics_path.unlink()
    write_json(config.to_dict(), output_dir / "config.json")

    print(
        "Q-Former ITC 학습 시작: "
        f"experiment={config.experiment_name}, samples={len(train_dataset)}, epochs={config.num_train_epochs}, "
        f"batch_size={config.train_batch_size}, batches/epoch={len(train_loader)}, device={device}",
        flush=True,
    )
    print(
        f"학습 파라미터: trainable={sum(p.numel() for p in trainable_parameters):,} "
        f"adapter=qformer text_encoder={config.qformer_text_model_id}",
        flush=True,
    )
    if initialized_layers:
        print(f"DistilBERT 초기화: qformer_layers={initialized_layers}", flush=True)
    print(f"metrics: {metrics_path}", flush=True)

    epoch_summaries: list[dict[str, float | int]] = []
    global_step = 0
    model.train()
    for epoch in range(config.num_train_epochs):
        epoch_losses: list[float] = []
        skipped_batches = 0
        print(f"[itc epoch {epoch + 1}/{config.num_train_epochs}] 시작", flush=True)
        for batch_index, batch in enumerate(train_loader):
            if len(batch.sample_ids) < 2:
                skipped_batches += 1
                append_jsonl(
                    metrics_path,
                    {
                        "event": "skip_singleton_batch",
                        "epoch": epoch,
                        "batch": batch_index + 1,
                        "sample_ids": batch.sample_ids,
                    },
                )
                continue
            output = model(
                pixel_values=batch.pixel_values.to(device),
                input_ids=batch.input_ids.to(device),
                attention_mask=batch.attention_mask.to(device),
            )
            loss_value = float(output.loss.item())
            if not math.isfinite(loss_value):
                raise FloatingPointError(
                    f"ITC loss가 비정상 값입니다: epoch={epoch + 1}, batch={batch_index + 1}, "
                    f"sample_ids={batch.sample_ids}, loss={loss_value}"
                )
            output.loss.backward()
            grad_norm = clip_gradients_if_needed(trainable_parameters, config.max_grad_norm)
            optimizer.step()
            assert_trainable_parameters_finite(trainable_parameters)
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            epoch_losses.append(loss_value)
            running_loss = sum(epoch_losses) / len(epoch_losses)
            if should_log_itc_step(batch_index, len(train_loader)):
                print(
                    f"[itc epoch {epoch + 1}/{config.num_train_epochs} batch {batch_index + 1}/{len(train_loader)}] "
                    f"step={global_step} loss={loss_value:.4f} avg={running_loss:.4f}",
                    flush=True,
                )
            append_jsonl(
                metrics_path,
                {
                    "event": "batch",
                    "epoch": epoch,
                    "step": global_step,
                    "batch": batch_index + 1,
                    "batch_count": len(train_loader),
                    "sample_ids": batch.sample_ids,
                    "loss": loss_value,
                    "running_loss": running_loss,
                    "grad_norm": grad_norm,
                    "logit_scale": float(model.logit_scale.exp().detach().cpu().clamp(max=100.0)),
                },
            )
        if not epoch_losses:
            raise RuntimeError("ITC 학습 가능한 batch가 없습니다. train_batch_size를 2 이상으로 설정하세요.")
        validation_summary = None
        if validation_loader is not None:
            validation_summary = evaluate_itc_loss(model=model, data_loader=validation_loader, device=device)
        epoch_summary = summarize_itc_epoch(
            epoch=epoch,
            losses=epoch_losses,
            skipped_batches=skipped_batches,
            validation_summary=validation_summary,
        )
        epoch_summaries.append(epoch_summary)
        validation_text = "" if validation_summary is None else f" val_loss={validation_summary['avg_loss']:.4f}"
        print(
            f"[itc epoch {epoch + 1}/{config.num_train_epochs}] 완료 "
            f"avg_loss={epoch_summary['avg_loss']:.4f} first={epoch_summary['first_loss']:.4f} "
            f"last={epoch_summary['last_loss']:.4f} delta={epoch_summary['loss_change']:.4f}"
            f"{validation_text}",
            flush=True,
        )
        append_jsonl(metrics_path, {"event": "epoch_summary", **epoch_summary})
        torch.save(model.visual_adapter.state_dict(), output_dir / f"visual_adapter_itc_epoch_{epoch + 1}.pt")

    torch.save(model.visual_adapter.state_dict(), output_dir / "visual_adapter.pt")
    torch.save(
        {
            "visual_adapter": model.visual_adapter.state_dict(),
            "image_projection": model.image_projection.state_dict(),
            "text_projection": model.text_projection.state_dict(),
            "logit_scale": model.logit_scale.detach().cpu(),
            "text_model_id": config.qformer_text_model_id,
            "contrastive_dim": config.contrastive_dim,
        },
        output_dir / "qformer_itc.pt",
    )
    write_json(
        {
            "experiment_name": config.experiment_name,
            "epoch_count": config.num_train_epochs,
            "sample_count": len(train_dataset),
            "batches_per_epoch": len(train_loader),
            "final_avg_loss": epoch_summaries[-1]["avg_loss"],
            "epochs": epoch_summaries,
        },
        summary_path,
    )
    print(f"Q-Former ITC 완료: {output_dir}", flush=True)
    print(f"summary: {summary_path}", flush=True)
    print(f"checkpoint: {output_dir / 'qformer_itc.pt'}", flush=True)


def resolve_llm_hidden_dim(llm_model_id: str, auto_config) -> int:
    llm_config = auto_config.from_pretrained(llm_model_id, trust_remote_code=True)
    if hasattr(llm_config, "hidden_size"):
        return int(llm_config.hidden_size)
    if hasattr(llm_config, "n_embd"):
        return int(llm_config.n_embd)
    raise ValueError(f"LLM hidden dimension을 config에서 찾을 수 없습니다: {llm_model_id}")


def build_itc_text(sample) -> str:
    """ITC에 사용할 이미지-텍스트 문장을 만든다."""

    if str(getattr(sample, "task", "")).lower() == "caption":
        return sample.answer
    return f"{sample.question} {sample.answer}".strip()


def should_log_itc_step(batch_index: int, batch_count: int) -> bool:
    if batch_count <= 20:
        return True
    if batch_index == 0 or batch_index + 1 == batch_count:
        return True
    interval = max(1, batch_count // 10)
    return (batch_index + 1) % interval == 0


def summarize_itc_epoch(
    *,
    epoch: int,
    losses: list[float],
    skipped_batches: int,
    validation_summary: dict[str, float | int] | None,
) -> dict[str, float | int]:
    if not losses:
        raise ValueError("ITC epoch loss 목록이 비어 있습니다.")
    if not all(math.isfinite(loss) for loss in losses):
        raise FloatingPointError("ITC epoch loss 목록에 NaN 또는 Inf가 포함되어 있습니다.")
    first_loss = losses[0]
    last_loss = losses[-1]
    summary: dict[str, float | int] = {
        "epoch": epoch,
        "batch_count": len(losses),
        "skipped_batches": skipped_batches,
        "first_loss": first_loss,
        "last_loss": last_loss,
        "loss_change": last_loss - first_loss,
        "loss_decrease": first_loss - last_loss,
        "avg_loss": sum(losses) / len(losses),
        "min_loss": min(losses),
        "max_loss": max(losses),
    }
    if validation_summary is not None:
        summary["validation_loss"] = validation_summary["avg_loss"]
        summary["validation_batch_count"] = validation_summary["batch_count"]
        summary["validation_skipped_batches"] = validation_summary["skipped_batches"]
    return summary


def evaluate_itc_loss(*, model, data_loader, device) -> dict[str, float | int]:
    import torch

    was_training = model.training
    model.eval()
    losses: list[float] = []
    skipped_batches = 0
    with torch.no_grad():
        for batch in data_loader:
            if len(batch.sample_ids) < 2:
                skipped_batches += 1
                continue
            output = model(
                pixel_values=batch.pixel_values.to(device),
                input_ids=batch.input_ids.to(device),
                attention_mask=batch.attention_mask.to(device),
            )
            loss_value = float(output.loss.item())
            if not math.isfinite(loss_value):
                raise FloatingPointError(f"validation ITC loss가 비정상 값입니다: {loss_value}")
            losses.append(loss_value)
    if was_training:
        model.train()
    if not losses:
        raise RuntimeError("ITC validation 가능한 batch가 없습니다. validation batch size를 2 이상으로 설정하세요.")
    return {
        "batch_count": len(losses),
        "skipped_batches": skipped_batches,
        "avg_loss": sum(losses) / len(losses),
        "min_loss": min(losses),
        "max_loss": max(losses),
    }


def clip_gradients_if_needed(parameters, max_grad_norm: float) -> float | None:
    if max_grad_norm <= 0:
        return None
    import torch

    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_grad_norm)
    return float(grad_norm.detach().cpu())


def assert_trainable_parameters_finite(parameters) -> None:
    import torch

    for parameter in parameters:
        if not torch.isfinite(parameter.detach()).all().item():
            raise FloatingPointError("optimizer step 이후 ITC 학습 parameter에 NaN 또는 Inf가 생겼습니다.")


def append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
