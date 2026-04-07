<라이브러리 다운>
pip install datasets transformers torch scikit-learn accelerate

<학습>
python mk2/code/toxic_comment_cli.py train --dataset kold --output_dir ./outputs/toxic_kold

<실행>
python toxic_comment_cli.py predict --model_dir ./outputs/toxic_kold