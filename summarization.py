from transformers import pipeline
from transformers import AutoModelForSeq2SeqLM


def summarize_text(text: str, max_length: int | None = 143, min_length: int|None= 10) -> str:
    summarizer = pipeline(
        task="text-generation",
        model="facebook/bart-large-cnn",
        # forced_bos_token_id=0
    )
    text = f"Summarize the following text: \n{text}"
    print('-'*10)
    print(text)
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    print(summary)
    print('='*10)
    return summary[0]['generated_text']
