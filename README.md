# Knitting with Short-Rows Shaping

<img src="https://github.com/zishun/KnittingShortRows2021/raw/main/input/overview.png" width="800"/>

## Usage

1. Clone ```git clone https://github.com/zishun/knitting2021 && cd knitting2021```
2. Install dependencies 
```
pip install -r requirements.txt
conda install -c conda-forge igl
```
3. Run the examples
```
cd examples/
python ./hemisphere.py
python ./skullcap.py
python ./triple_peak.py
python ./mannequin.py 0
python ./mannequin.py 1
```


## Bibtex

```
@article{Knitting4D2021,
    author = {Liu, Zishun and Han, Xingjian and Zhang, Yuchen and Chen, Xiangjia and Lai, Yu-Kun and Doubrovski, Eugeni L. and Whiting, Emily and Wang, Charlie C. L.},
    title = {Knitting {4D} Garments with Elasticity Controlled for Body Motion},
    year = {2021},
    issue_date = {August 2021},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    volume = {40},
    number = {4},
    issn = {0730-0301},
    url = {https://doi.org/10.1145/3450626.3459868},
    doi = {10.1145/3450626.3459868},
    journal = {ACM Trans. Graph.},
    month = jul,
    articleno = {62},
    numpages = {16},
    keywords = {computational fabrication, 4D garment, knitting, elasticity control}
}
```


## Acknowledgements
I would like to acknowledge Dr. Zhan Song, Lijun Shu, and Yuping Ye, from [SIAT](http://english.siat.cas.cn), for their help on 3D scanning.
