datasets=(
    "passage_count"
    "narrativeqa"
    "qasper"
    "multifieldqa_en"
    "hotpotqa"
    "2wikimqa"
    "musique"
    "qmsum"
    "trec"
    "triviaqa"
    "samsum"
    "passage_retrieval_en"
    "lcc"
    "repobench-p"
    "gov_report"
    "multi_news"
)

# IceCache
for dataset in "${datasets[@]}"; do
    python longbench_pred.py --icecache  --datasets ${dataset}  --page-budgets 16 --page-size 16 --n-win-pages 2 --n-sink-pages 2 --model llama-3.1 --name test
done

# IceCache-reuse
for dataset in "${datasets[@]}"; do
    python longbench_pred.py --icecache  --datasets ${dataset}  --page-budgets 16 --page-size 16 --n-win-pages 2 --n-sink-pages 2 --model llama-3.1 --n_reuse_layers 3  --name test-reuse
done