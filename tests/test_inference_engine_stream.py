from modules.inference_engine import InferenceEngine


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __call__(self, prompt, return_tensors="pt", padding=True, truncation=True, max_length=4096):
        class DummyInputs(dict):
            def to(self, device):
                return self
        return DummyInputs({"input_ids": [[1, 1]], "attention_mask": [[1, 1]]})

    def decode(self, output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True):
        return "x"


class DummyModel:
    def generate(self, **kwargs):
        return [[1, 1, 2]]


def test_deepspeed_stream_branch_uses_incremental_stream(monkeypatch):
    engine = InferenceEngine(model=DummyModel(), tokenizer=DummyTokenizer(), deepspeed_config={"ok": True})
    monkeypatch.setattr(engine, "_generate_deepspeed_stream", lambda **kwargs: iter(["a", "b"]))
    tokens = list(
        engine._generate_stream(
            prompt="hello",
            max_new_tokens=2,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            do_sample=True,
        )
    )
    assert tokens == ["a", "b"]

