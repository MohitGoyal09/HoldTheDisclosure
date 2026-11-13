"""Chat backends: a fake one for tests and dry runs, and a lazy Hugging Face one."""

from __future__ import annotations

from collections.abc import Callable

from htd.sheet import Sheet, load_sheet

Messages = list[dict[str, str]]
Policy = Callable[[Messages], str]


class FakeBackend:
    """Deterministic stand-in. Default policy: state the product's first disclosure
    on turns 1-2, drop it afterwards (control products never get one)."""

    def __init__(self, policy: Policy | None = None, sheet: Sheet | None = None):
        self._sheet = sheet
        self.policy = policy or self._default_policy

    def _default_policy(self, messages: Messages) -> str:
        sheet = self._sheet or load_sheet()
        system = messages[0]["content"]
        turn = sum(1 for m in messages if m["role"] == "user")
        product = next((p for p in sheet.products if p.facts_for_prompt in system), None)
        body = "Here is a short answer to your question."
        if product and product.disclosures and turn <= 2:
            return f"{body} {product.disclosures[0].statement['en']}"
        return body

    def generate_batch(self, messages_list: list[Messages]) -> list[str]:
        return [self.policy(m) for m in messages_list]


# Chat-template markers that end an assistant reply. Tiny Aya's generation_config stops
# only on its own eos id, but each reply ends with <|END_RESPONSE|><|END_OF_TURN_TOKEN|>.
# Without these stops the model can run on and write the next user turn.
REPLY_END_MARKERS = ("<|END_RESPONSE|>", "<|END_OF_TURN_TOKEN|>")


def _dtype_kwarg() -> str:
    """transformers 5 renamed torch_dtype to dtype."""
    import transformers

    major = int(transformers.__version__.split(".")[0])
    return "dtype" if major >= 5 else "torch_dtype"


def stop_token_ids(tokenizer, generation_eos) -> list[int]:
    """All token ids that should end generation: generation eos, tokenizer eos, reply markers."""
    ids: set[int] = set()
    if isinstance(generation_eos, int):
        ids.add(generation_eos)
    elif generation_eos:
        ids.update(generation_eos)
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    unk = getattr(tokenizer, "unk_token_id", None)
    for marker in REPLY_END_MARKERS:
        tid = tokenizer.convert_tokens_to_ids(marker)
        if tid is not None and tid != unk:
            ids.add(tid)
    return sorted(ids)


class HFChatBackend:
    """Greedy chat generation with transformers. torch/transformers import lazily."""

    def __init__(
        self,
        model_id: str,
        dtype: str = "float16",  # Kaggle T4 has no bf16
        max_new_tokens: int = 300,
        batch_size: int = 8,
        load_in_4bit: bool = False,
        hf_token: str | None = None,
        adapter: str | None = None,
        device_map: str | dict | None = None,
        device: str = "auto",
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # One GPU is enough for 3.35B. Sharding with "auto" on 2xT4 is much slower.
        if device_map is None:
            device_map = {"cuda": {"": 0}, "mps": {"": "mps"}, "cpu": {"": "cpu"}}.get(device, "auto")
        kwargs: dict = {"token": hf_token, "device_map": device_map}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=getattr(torch, dtype)
            )
        else:
            kwargs[_dtype_kwarg()] = getattr(torch, dtype)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter, token=hf_token)
        self.model.eval()
        self.stop_ids = stop_token_ids(self.tokenizer, self.model.generation_config.eos_token_id)

    def generate_batch(self, messages_list: list[Messages]) -> list[str]:
        out: list[str] = []
        for i in range(0, len(messages_list), self.batch_size):
            out.extend(self._generate(messages_list[i : i + self.batch_size]))
        return out

    def _generate(self, chunk: list[Messages]) -> list[str]:
        tok = self.tokenizer
        # Tiny Aya's chat template adds a long preamble (about 366 tokens), so never
        # truncate inputs to a short max length.
        texts = [
            tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False, return_tensors=None)
            for m in chunk
        ]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.model.device)
        with self._torch.no_grad():
            gen = self.model.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
                eos_token_id=self.stop_ids,
            )
        new = gen[:, enc["input_ids"].shape[1] :]
        return [s.strip() for s in tok.batch_decode(new, skip_special_tokens=True)]

    def close(self) -> None:
        """Free GPU memory so the next model can load."""
        import gc

        del self.model
        gc.collect()
        self._torch.cuda.empty_cache()


class APIBackend:
    """Chat model behind an API, through an object with `generate(messages).text` (CuratorKIT's
    LiteLLM backend). Calls run in parallel; a call that still fails after `retries` raises, so a
    transcript never holds an error string as a reply."""

    def __init__(self, llm, workers: int = 4, retries: int = 3, wait: float = 5.0):
        self.llm, self.workers, self.retries, self.wait = llm, workers, retries, wait

    def _one(self, messages: Messages) -> str:
        import time

        for attempt in range(self.retries + 1):
            try:
                text = self.llm.generate(messages).text.strip()
                if not text:
                    raise RuntimeError("empty reply")
                return text
            except Exception:
                if attempt == self.retries:
                    raise
                time.sleep(self.wait * (attempt + 1))
        raise AssertionError("unreachable")

    def generate_batch(self, messages_list: list[Messages]) -> list[str]:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(self.workers) as pool:
            return list(pool.map(self._one, messages_list))
