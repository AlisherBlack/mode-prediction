cd input/
rm -rf modes
curl -L $(yadisk-direct https://disk.yandex.ru/d/iKwNbO25Xd4XsA) -o modes.zip
unzip modes.zip
rm modes.zip
cd ..