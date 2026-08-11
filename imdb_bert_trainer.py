"""任务 11（选做）：BERT-base 微调。实现在 hf_trainer.py。

`bert-base-uncased`：12 层，隐藏维度 768，约 1.1 亿参数。
预训练任务是 MLM（随机遮盖 15% 的词让模型还原）+ NSP（判断两句是否相邻）。
双向 Transformer 编码器——这是它相对 GPT 系列单向语言模型的核心差异，
也是它特别适合分类这类「理解型」任务的原因。

    python imdb_bert_trainer.py
    python imdb_bert_trainer.py --epochs 2 --batch-size 16
    python imdb_bert_trainer.py --predict "The screenplay is razor sharp."
"""

import hf_trainer

NAME = "bert"

if __name__ == "__main__":
    parser = hf_trainer.build_parser(model_id="bert-base-uncased",
                                     batch_size=16, lr=2e-5, epochs=2)
    hf_trainer.run(NAME, parser.parse_args())
