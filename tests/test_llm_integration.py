from __future__ import annotations

import importlib.util
import os
import unittest


RUN_INTEGRATION = os.environ.get("RUN_MODEL_INTEGRATION_TESTS") == "1"


@unittest.skipUnless(RUN_INTEGRATION, "RUN_MODEL_INTEGRATION_TESTS=1일 때만 LLM integration test를 실행합니다.")
class LlmIntegrationTest(unittest.TestCase):
    def test_llm_accepts_inputs_embeds(self) -> None:
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch가 설치되어 있지 않습니다.")
        if importlib.util.find_spec("transformers") is None:
            self.skipTest("transformers가 설치되어 있지 않습니다.")

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from mini_vlm.config import load_config

        config = load_config("configs/dinov3-mini-vlm-smoke.json")
        tokenizer = AutoTokenizer.from_pretrained(config.llm_model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(config.llm_model_id, trust_remote_code=True)
        prompt_ids = tokenizer.encode("Question: What is shown?\nAnswer:", add_special_tokens=False)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        embeddings = model.get_input_embeddings()(input_ids)

        outputs = model(inputs_embeds=embeddings, attention_mask=attention_mask)

        self.assertEqual(outputs.logits.shape[0], 1)
        self.assertEqual(outputs.logits.shape[1], input_ids.shape[1])
        self.assertEqual(outputs.logits.shape[-1], model.get_input_embeddings().num_embeddings)


if __name__ == "__main__":
    unittest.main()
