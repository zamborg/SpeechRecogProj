#!/bin/bash
pip install -r req.txt
mkdir audio
git clone https://github.com/facebookresearch/voxpopuli.git
cd voxpopuli




# download data:
ROOT='../../data'
GREEK='el'
LATVIAN='lv'
MALTESE='mt'

python -m voxpopuli.download_audios --root $ROOT --subset $GREEK
# python -m voxpopuli.download_audios --root $ROOT --subset $LATVIAN
# python -m voxpopuli.download_audios --root $ROOT --subset $MALTESE

# now segment the data:
python -m voxpopuli.get_unlabelled_data --root $ROOT --subset $GREEK
# python -m voxpopuli.get_unlabelled_data --root $ROOT --subset $LATVIAN
# python -m voxpopuli.get_unlabelled_data --root $ROOT --subset $MALTESE

cd .. # move back dir