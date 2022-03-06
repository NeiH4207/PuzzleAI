# Procon 2022

## Install Python Libraries
```
pip install -r requirements.txt
```

## Data Source
https://github.com/openimages/dataset/blob/main/READMEV3.md

## How to run code

### Pull image from server
```
usage: interaction.py [-h] [--sol-path SOL_PATH] [--output-path OUTPUT_PATH] [--token TOKEN]
                      [-s TOURNAMENT_NAME] [-r ROUND_NAME] [-m MATCH_NAME] [-p MODE]

optional arguments:
  -h, --help            show this help message and exit
  --sol-path SOL_PATH
  --output-path OUTPUT_PATH
  --token TOKEN
  -s TOURNAMENT_NAME, --tournament_name TOURNAMENT_NAME
  -r ROUND_NAME, --round_name ROUND_NAME
  -m MATCH_NAME, --match_name MATCH_NAME
  -p MODE, --mode MODE [r/read, w/submmit, show]
```
Example:

```
python3 interaction.py --token 'abc' -s 'Tournament_1' -r 'Round_1' -m 'Match_1' -p 'r'
```
---
**NOTE**

Check ENDPOINT in `src/request.py` file.

---

The Match_1's information 'Match_1' saved on `input/game_info/Match_1.json` includes:

1. Block dimentions
2. Block size
3. original block
4. max number of selects
5. ratio of select and swap operator
6. image size

### Image Recovery

```
usage: recover_image.py [-h] [--game-info-path GAME_INFO_PATH] [--model-path MODEL_PATH]
                        [--model-name MODEL_NAME] [--output-path OUTPUT_PATH] [-f FILE_NAME]
                        [--image-size-out IMAGE_SIZE_OUT] [-s BLOCK_SIZE] [-a ALGORITHM] [-v] [-m]
                        [-t THRESHOLD] [-j N_JUMPS]

optional arguments:
  -h, --help            show this help message and exit
  --game-info-path GAME_INFO_PATH
  --model-path MODEL_PATH
  --model-name MODEL_NAME
  --output-path OUTPUT_PATH
  -f FILE_NAME, --file-name FILE_NAME
  --image-size-out IMAGE_SIZE_OUT
  -a ALGORITHM, --algorithm ALGORITHM
  -v, --verbose         save every on output/sample.png file
  -m, --monitor         for interfere
```

Example:
```
python3 recover_image.py -f Match_1 -v -m
```

Recovery solution are stored in `input/states/Match_1.bin` binary file


### Image Sorting
```
usage: procon.py [-h] [--state-path STATE_PATH] [-f ITEM_NAME] [--model-name MODEL_NAME]
                 [--output-path OUTPUT_PATH] [-a ALGORITHM] [-v] [-t SLEEP] [-k SKIP] [-s N_FAST_MOVES]
                 [-r RATE] [-c MAX_SELECT] [-d DEPTH] [-b BREADTH]

optional arguments:
  -h, --help            show this help message and exit
  --state-path STATE_PATH
  -f ITEM_NAME, --item-name ITEM_NAME
  --model-name MODEL_NAME
  --output-path OUTPUT_PATH
  -a ALGORITHM, --algorithm ALGORITHM
                        algorithm to use
  -v, --verbose
  -t SLEEP, --sleep SLEEP
  -k SKIP, --skip SKIP
  -s N_FAST_MOVES, --n_fast_moves N_FAST_MOVES
  -r RATE, --rate RATE
  -c MAX_SELECT, --max_select MAX_SELECT
  -d DEPTH, --depth DEPTH               (using for tree search)
  -b BREADTH, --breadth BREADTH         (using for tree search)
```

Example
```
python3 procon.py -f 'Match_1' -a 'standard' -s 1 -v 
```
Solution are stored in 'output/solutions/Match_1.json'

Finally, run this command to submit solution:
```
python3 interaction.py --token 'abc' -s 'Tournament_1' -r 'Round_1' -m 'Match_1' -p 'w'
```

Server return:
```
{
    'message': 'Invalid Solution/Submit solution success', 
    'cost': 10, 
    'correctPosition': 10, 
    'correctAngle': 10
}
```
Show all submissions:
```
python3 interaction.py --token 'abc' -s 'Tournament_1' -r 'Round_1' -m 'Match_1' -p 'show'
```
