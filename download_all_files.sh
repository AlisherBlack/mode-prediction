mkdir input/
bash download_modes.sh
bash download_sweep.sh

find input/ -name __MACOSX -exec rm -rf {} \;
find input/ -name 'Icon' -type f -delete 