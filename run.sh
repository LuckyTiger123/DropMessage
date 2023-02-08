# GCN Cora
python train.py -dr 0.9 -bb GCN -dm DropMessage -d Cora -r 2

# GCN CiteSeer
python train.py -dr 0.9 -bb GCN -dm DropMessage -d CiteSeer -r 0

# GCN PubMed
python train.py -dr 0.15 -bb GCN -dm DropMessage -d PubMed -r 0

# GCN Flickr
python train_flickr.py -dr 0.20 -bb GCN -dm DropMessage -r 0

# GCN ogbn-arxiv
python train_ogb_arxiv.py -dr 0.20 -bb GCN -dm DropMessage -r 0

# GAT Cora
python train.py -dr 0.9 -bb GAT -hs 8 -fyd 8 -dm DropMessage -d Cora -r 0

# GAT CiteSeer
python train.py -dr 0.9 -bb GAT -hs 8 -fyd 8 -dm DropMessage -d CiteSeer -r 2

# GAT PubMed
python train.py -dr 0.85 -bb GAT -hs 8 -fyd 8 -dm DropMessage -d PubMed -r 5

# GAT Flickr
python train_flickr.py -dr 0.40 -bb GAT -dm DropMessage -r 0

# GAT ogbn-arxiv
python train_ogb_arxiv.py -dr 0.20 -bb GAT -dm DropMessage -r 0

# APPNP Cora
python train.py -dr 0.8 -bb APPNP -k 10 -a 0.1 -fyd 64 -dm DropMessage -d Cora -r 5

# APPNP CiteSeer
python train.py -dr 0.7 -bb APPNP -k 10 -a 0.1 -fyd 64 -dm DropMessage -d CiteSeer -r 2

# APPNP PubMed
python train.py -dr 0.75 -bb APPNP -k 10 -a 0.1 -fyd 64 -dm DropMessage -d PubMed -r 0

# APPNP Flickr
python train_flickr.py -dr 0.35 -bb APPNP -dm DropMessage -r 0

# APPNP ogbn-arxiv
python train_ogb_arxiv.py -dr 0.20 -bb APPNP -hd 64 -dm DropMessage -r 0