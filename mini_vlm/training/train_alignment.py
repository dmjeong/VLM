from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from mini_vlm.config import load_config
from mini_vlm.utils.checkpoints import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="DINOv3 Mini VLM Stage 1 alignment 학습")
    parser.add_argument("--config", default="configs/dinov3-mini-vlm.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.dry_run:
        print(f"실험명: {config.experiment_name}")
        print(f"Vision: {config.vision_model_id}")
        print(f"LLM: {config.llm_model_id}")
        print("dry-run: 실제 학습은 실행하지 않았습니다.")
        return

    train_alignment(config)


def train_alignment(config_path_or_object) -> None:
    """Stage 1 caption alignment 학습을 실행한다.

    의도: DINOv3/LLM은 freeze하고 MLP visual adapter만 먼저 학습한다.
    참고: 설계서 8.2 Stage 1 Caption Alignment.
    선택 이유: adapter-only overfit이 되지 않으면 더 복잡한 Q-Former나 LoRA로 가기 전에 데이터/shape/loss
    계약을 먼저 의심해야 한다.
    """

    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise SystemExit("실제 학습에는 torch가 필요합니다. `pip install '.[model]'` 후 다시 실행하세요.") from exc

    from mini_vlm.data import MiniVlmCollator, MiniVlmDataset
    from mini_vlm.models.builder import build_mini_vlm
    from mini_vlm.reporting import write_training_report
    from mini_vlm.utils.device import select_torch_device
    from mini_vlm.utils.seed import set_seed

    config = load_config(config_path_or_object) if isinstance(config_path_or_object, (str, Path)) else config_path_or_object
    set_seed(config.seed)
    built = build_mini_vlm(config)
    train_dataset = MiniVlmDataset(config.train_jsonl, image_root=config.image_root)
    collator = MiniVlmCollator(
        tokenizer=built.tokenizer,
        image_processor=built.image_processor,
        image_root=config.image_root,
        max_text_length=config.max_text_length,
    )
    data_generator = torch.Generator()
    data_generator.manual_seed(config.seed)
    data_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        collate_fn=collator,
        generator=data_generator,
    )
    batches_per_epoch = len(data_loader)
    validation_loader = None
    validation_path = Path(config.validation_jsonl)
    if config.validation_jsonl and validation_path.exists():
        validation_dataset = MiniVlmDataset(config.validation_jsonl, image_root=config.image_root)
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=config.train_batch_size,
            shuffle=False,
            collate_fn=collator,
        )
    device = select_torch_device(config.device)
    model = built.model.to(device)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("학습 가능한 파라미터가 없습니다. freeze 설정과 adapter 구성을 확인하세요.")
    trainable_parameter_count = sum(parameter.numel() for parameter in trainable_parameters)
    total_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(trainable_parameters, lr=config.learning_rate)
    accumulation_steps = normalize_gradient_accumulation_steps(config.gradient_accumulation_steps)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(config.to_dict(), output_dir / "config.json")
    metrics_path = output_dir / "metrics.jsonl"
    summary_path = output_dir / "training_summary.json"
    if metrics_path.exists():
        metrics_path.unlink()
    global_step = 0
    optimizer_step = 0
    epoch_summaries: list[dict[str, float | int]] = []

    model.train()
    if hasattr(model, "enforce_freeze_modes"):
        model.enforce_freeze_modes()
    optimizer.zero_grad(set_to_none=True)
    print(
        "학습 시작: "
        f"experiment={config.experiment_name}, samples={len(train_dataset)}, epochs={config.num_train_epochs}, "
        f"batch_size={config.train_batch_size}, batches/epoch={batches_per_epoch}, device={device}",
        flush=True,
    )
    print(
        f"학습 파라미터: trainable={trainable_parameter_count:,} / total={total_parameter_count:,} "
        f"adapter={config.adapter_type} lora={config.use_lora}",
        flush=True,
    )
    print(f"metrics: {metrics_path}", flush=True)
    for epoch in range(config.num_train_epochs):
        epoch_batch_count = 0
        epoch_losses: list[float] = []
        pending_optimizer_step = False
        print(f"[epoch {epoch + 1}/{config.num_train_epochs}] 시작", flush=True)
        for batch_index, batch in enumerate(data_loader):
            output = model(
                pixel_values=batch.pixel_values.to(device),
                input_ids=batch.input_ids.to(device),
                attention_mask=batch.attention_mask.to(device),
                labels=batch.labels.to(device),
            )
            if output.loss is None:
                raise RuntimeError("LLM output에 loss가 없습니다.")
            loss_value = float(output.loss.item())
            if not math.isfinite(loss_value):
                append_jsonl(
                    metrics_path,
                    {
                        "event": "nonfinite_loss",
                        "epoch": epoch,
                        "step": global_step + 1,
                        "batch": batch_index + 1,
                        "sample_ids": batch.sample_ids,
                        "loss": str(loss_value),
                    },
                )
                raise FloatingPointError(
                    f"비정상 loss가 발생했습니다: epoch={epoch + 1}, batch={batch_index + 1}, "
                    f"sample_ids={batch.sample_ids}, loss={loss_value}"
                )
            loss = output.loss / accumulation_steps
            loss.backward()
            grad_norm = clip_gradients_if_needed(trainable_parameters, config.max_grad_norm)
            if grad_norm is not None and not math.isfinite(grad_norm):
                append_jsonl(
                    metrics_path,
                    {
                        "event": "nonfinite_gradient",
                        "epoch": epoch,
                        "step": global_step + 1,
                        "batch": batch_index + 1,
                        "sample_ids": batch.sample_ids,
                        "grad_norm": str(grad_norm),
                    },
                )
                raise FloatingPointError(
                    f"비정상 gradient가 발생했습니다: epoch={epoch + 1}, batch={batch_index + 1}, "
                    f"sample_ids={batch.sample_ids}, grad_norm={grad_norm}"
                )
            epoch_batch_count += 1
            epoch_losses.append(loss_value)
            pending_optimizer_step = True
            stepped = should_step_optimizer(batch_index, accumulation_steps)
            if stepped:
                optimizer.step()
                assert_trainable_parameters_finite(trainable_parameters)
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                pending_optimizer_step = False
            global_step += 1
            running_loss = sum(epoch_losses) / len(epoch_losses)
            if should_log_training_step(batch_index, batches_per_epoch):
                print(
                    format_training_progress(
                        epoch=epoch,
                        epoch_count=config.num_train_epochs,
                        batch_index=batch_index,
                        batch_count=batches_per_epoch,
                        global_step=global_step,
                        optimizer_step=optimizer_step if stepped else None,
                        loss=loss_value,
                        running_loss=running_loss,
                    ),
                    flush=True,
                )
            with metrics_path.open("a", encoding="utf-8") as file:
                file.write(
                    json.dumps(
                        {
                            "event": "batch",
                            "epoch": epoch,
                            "step": global_step,
                            "batch": batch_index + 1,
                            "batch_count": batches_per_epoch,
                            "optimizer_step": optimizer_step if stepped else None,
                            "sample_ids": batch.sample_ids,
                            "loss": loss_value,
                            "running_loss": running_loss,
                            "grad_norm": grad_norm,
                        },
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                )
        if epoch_batch_count == 0:
            raise RuntimeError("학습 데이터가 비어 있습니다.")
        if pending_optimizer_step:
            optimizer.step()
            assert_trainable_parameters_finite(trainable_parameters)
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
            append_jsonl(
                metrics_path,
                {
                    "epoch": epoch,
                    "step": global_step,
                    "optimizer_step": optimizer_step,
                    "event": "epoch_tail_optimizer_step",
                    "reason": "gradient_accumulation_steps보다 batch 수가 적어 남은 gradient를 epoch 끝에서 반영",
                },
            )
        epoch_summary = summarize_epoch_losses(epoch=epoch, losses=epoch_losses, optimizer_step=optimizer_step)
        validation_summary = None
        if validation_loader is not None:
            validation_summary = evaluate_validation_loss(model=model, data_loader=validation_loader, device=device)
            epoch_summary["validation_loss"] = validation_summary["avg_loss"]
            epoch_summary["validation_batch_count"] = validation_summary["batch_count"]
        epoch_summaries.append(epoch_summary)
        validation_text = ""
        if validation_summary is not None:
            validation_text = f" val_loss={validation_summary['avg_loss']:.4f}"
        print(
            "[epoch "
            f"{epoch + 1}/{config.num_train_epochs}] 완료 "
            f"avg_loss={epoch_summary['avg_loss']:.4f} "
            f"first={epoch_summary['first_loss']:.4f} "
            f"last={epoch_summary['last_loss']:.4f} "
            f"delta={epoch_summary['loss_change']:.4f} "
            f"min={epoch_summary['min_loss']:.4f} "
            f"max={epoch_summary['max_loss']:.4f}"
            f"{validation_text}",
            flush=True,
        )
        with metrics_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps({"event": "epoch_summary", **epoch_summary}, ensure_ascii=False, allow_nan=False) + "\n")
            if validation_summary is not None:
                file.write(
                    json.dumps(
                        {"event": "validation_summary", "epoch": epoch, **validation_summary},
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                )

    torch.save(model.visual_adapter.state_dict(), output_dir / "visual_adapter.pt")
    if config.use_lora and hasattr(model.llm, "save_pretrained"):
        model.llm.save_pretrained(output_dir / "llm_lora")
    write_json(
        {
            "experiment_name": config.experiment_name,
            "epoch_count": config.num_train_epochs,
            "sample_count": len(train_dataset),
            "batches_per_epoch": batches_per_epoch,
            "final_avg_loss": epoch_summaries[-1]["avg_loss"] if epoch_summaries else None,
            "epochs": epoch_summaries,
        },
        summary_path,
    )
    report_paths = write_training_report(output_dir)
    print(f"Stage 1 alignment 완료: {output_dir}", flush=True)
    print(f"summary: {summary_path}", flush=True)
    print(f"report: {report_paths.markdown_path}", flush=True)
    print(f"loss_curve: {report_paths.svg_path}", flush=True)


def normalize_gradient_accumulation_steps(raw_steps: int) -> int:
    """gradient accumulation 값이 0 이하로 들어와도 학습 루프가 깨지지 않게 정규화한다."""

    return max(1, int(raw_steps))


def should_step_optimizer(batch_index: int, accumulation_steps: int) -> bool:
    """현재 batch에서 optimizer step을 실행해야 하는지 판단한다."""

    normalized_steps = normalize_gradient_accumulation_steps(accumulation_steps)
    return (batch_index + 1) % normalized_steps == 0


def has_pending_optimizer_step(batch_count: int, accumulation_steps: int) -> bool:
    """epoch 끝에서 남은 gradient를 반영해야 하는지 판단한다.

    의도: tiny dataset에서는 batch 수가 gradient accumulation보다 작을 수 있다. 이 helper가 없으면
    loss.backward()만 호출되고 optimizer.step()은 한 번도 실행되지 않는 조용한 실패가 생긴다.
    """

    if batch_count <= 0:
        return False
    normalized_steps = normalize_gradient_accumulation_steps(accumulation_steps)
    return batch_count % normalized_steps != 0


def should_log_training_step(batch_index: int, batch_count: int) -> bool:
    """콘솔에 현재 batch 진행률을 출력할지 판단한다.

    의도: 작은 smoke 학습에서는 모든 batch를 보여주고, 데이터가 커지면 대략 10% 단위로 줄여서 출력한다.
    """

    if batch_count <= 20:
        return True
    if batch_index == 0 or batch_index + 1 == batch_count:
        return True
    interval = max(1, batch_count // 10)
    return (batch_index + 1) % interval == 0


def format_training_progress(
    *,
    epoch: int,
    epoch_count: int,
    batch_index: int,
    batch_count: int,
    global_step: int,
    optimizer_step: int | None,
    loss: float,
    running_loss: float,
) -> str:
    progress = ((batch_index + 1) / batch_count) * 100 if batch_count else 0.0
    optimizer_text = "-" if optimizer_step is None else str(optimizer_step)
    return (
        f"[epoch {epoch + 1}/{epoch_count} batch {batch_index + 1}/{batch_count} {progress:5.1f}%] "
        f"step={global_step} opt_step={optimizer_text} loss={loss:.4f} avg={running_loss:.4f}"
    )


def summarize_epoch_losses(epoch: int, losses: list[float], optimizer_step: int) -> dict[str, float | int]:
    if not losses:
        raise ValueError("epoch loss 목록이 비어 있습니다.")
    if not all(math.isfinite(loss) for loss in losses):
        raise FloatingPointError("epoch loss 목록에 NaN 또는 Inf가 포함되어 있습니다.")
    first_loss = losses[0]
    last_loss = losses[-1]
    return {
        "epoch": epoch,
        "batch_count": len(losses),
        "optimizer_step": optimizer_step,
        "first_loss": first_loss,
        "last_loss": last_loss,
        "loss_change": last_loss - first_loss,
        "loss_decrease": first_loss - last_loss,
        "avg_loss": sum(losses) / len(losses),
        "min_loss": min(losses),
        "max_loss": max(losses),
    }


def evaluate_validation_loss(*, model, data_loader, device) -> dict[str, float | int]:
    """validation set 평균 loss를 계산하고 학습 모드로 복귀한다."""

    import torch

    was_training = model.training
    model.eval()
    validation_losses: list[float] = []
    with torch.no_grad():
        for batch in data_loader:
            output = model(
                pixel_values=batch.pixel_values.to(device),
                input_ids=batch.input_ids.to(device),
                attention_mask=batch.attention_mask.to(device),
                labels=batch.labels.to(device),
            )
            if output.loss is None:
                raise RuntimeError("validation 중 LLM output에 loss가 없습니다.")
            loss_value = float(output.loss.item())
            if not math.isfinite(loss_value):
                raise FloatingPointError(f"validation loss가 비정상 값입니다: {loss_value}")
            validation_losses.append(loss_value)
    if was_training:
        model.train()
        if hasattr(model, "enforce_freeze_modes"):
            model.enforce_freeze_modes()
    if not validation_losses:
        raise RuntimeError("validation 데이터가 비어 있습니다.")
    return {
        "batch_count": len(validation_losses),
        "avg_loss": sum(validation_losses) / len(validation_losses),
        "min_loss": min(validation_losses),
        "max_loss": max(validation_losses),
    }


def clip_gradients_if_needed(parameters, max_grad_norm: float) -> float | None:
    """gradient norm을 제한하고 clipping 전 norm을 반환한다."""

    if max_grad_norm <= 0:
        return None
    import torch

    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_grad_norm)
    return float(grad_norm.detach().cpu())


def assert_trainable_parameters_finite(parameters) -> None:
    """optimizer step 이후 학습 대상 parameter가 finite인지 확인한다."""

    import torch

    for parameter in parameters:
        if not torch.isfinite(parameter.detach()).all().item():
            raise FloatingPointError("optimizer step 이후 학습 parameter에 NaN 또는 Inf가 생겼습니다.")


def append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
