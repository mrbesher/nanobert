# NanoBERT
A simple implementation of BERT to imrpove our understanding.

**Co-authores**: [Mounes Zaval](https://github.com/mouneszawal), [Zeynep Akkoc](https://github.com/akkocz17)

## Training over Epochs
![Train and test loss and socre over epochs](assets/loss_score_graph.png)
*The graph above illustrates the train / test loss and scores of our NanoBERT model over 50 epochs. The decreasing trend in loss indicates the model's improving ability to predict masked tokens in the Quran dataset.*

## Usage
To train the model with the provided configurations, use the following command:
```bash
train.py --model_config_path configs/model_config.json --tokenizer_config_path configs/tokenizer_config.json --train_config_path configs/train_config.json --data_path data/quran.jsonl
```
This command specifies the paths to the model, tokenizer, and training configurations, as well as the data to be used for training. The model will be trained according to the specified configurations.