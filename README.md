# pointer-net-for-nested
The official implementation of ACL2022: [Bottom-Up Constituency Parsing and Nested Named Entity Recognition with Pointer Networks](https://arxiv.org/pdf/2110.05419.pdf)


## Setup
Environment 
```
conda create -n parsing python=3.7
conda activate parsing
while read requirement; do pip install $requirement; done < requirements.txt 
```

Download preprocessed PTB, CTB7, GENIA from: [link](https://drive.google.com/drive/folders/1qFP2JbcltAJ-Jq3MpkS--0MGEIgyE6vQ?usp=sharing)

For ACE04 and ACE05, send me e-mails.



 
# Run
```
python train.py +exp=ft_10 datamodule=a model=pointer 
a={ptb, ctb7}

python train.py +exp=ft_10 datamodule=genia model=pointer model.use_prev_label=True 

python train.py +exp=ft_50 datamodule=b model=pointer model.use_prev_label=True
b={ace04, ace05}
```

multirun example:
```
python train.py +exp=base model=pointer datamodule=ptb,ctb7 seed=0,1,2 --mutlirun
```

evaluation:
```
python evaluate.py +load_from_checkpoint=your/checkpoint/dir
```   


# Contact
Please let me know if there are any bugs. Also, feel free to contact bestsonta@gmail.com if you have any questions.

# Citation
```
@misc{yang2021bottomup,
      title={Bottom-Up Constituency Parsing and Nested Named Entity Recognition with Pointer Networks}, 
      author={Songlin Yang and Kewei Tu},
      year={2021},
      eprint={2110.05419},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

# Credits
The code is based on [lightning+hydra](https://github.com/ashleve/lightning-hydra-template) template. I use [FastNLP](https://github.com/fastnlp/fastNLP) for loading data. I use lots of built-in modules (LSTMs, Biaffines, Triaffines, Dropout Layers, etc) from [Supar](https://github.com/yzhangcs/parser/tree/main/supar).  



