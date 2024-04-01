cd input/
rm -rf sweep
curl -L $(yadisk-direct https://disk.yandex.ru/d/Yu8X0jLJ0F4K1w) -o sweep.zip
unzip sweep.zip
rm sweep.zip
cd ..
