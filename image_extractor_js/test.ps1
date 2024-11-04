echo "node .\image-extractor\main.js -f .\images\ -m google -e jfif"
node .\image-extractor\main.js -f .\images\ -m google -e jfif
echo "node .\image-extractor\main.js --folder .\images\ --model google --extension jfif --batch_size 2"
node .\image-extractor\main.js --folder .\images\ --model google --extension jfif --batch_size 2
echo "node .\image-extractor\main.js --folder .\images\ --model openai --extension jfif"
node .\image-extractor\main.js --folder .\images\ --model openai --extension jfif
echo "node .\image-extractor\main.js --folder .\images\ --model openai --extension jfif --batch_size 2"
node .\image-extractor\main.js --folder .\images\ --model openai --extension jfif --batch_size 2