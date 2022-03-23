import json
import os
import cv2
from src.request import Socket
from src.recover.environment import GameInfo
import warnings
warnings.filterwarnings("ignore")
import argparse

'''
python3 interaction.py \
    -s 'Computer_Tour' -r Computer_Round -p 'r' \
         --token '' \
              -m Natural_8
'''

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sol-path", type=str, default="output/solutions")
    parser.add_argument("--output-path", type=str, default="./output/recovered_images/")
    parser.add_argument( "--token", type=str, 
        default="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6NSwiaWF0IjoxNjQ3NTc2NzEzLCJleHAiOjE2NDc1OTQ3MTN9.pQAC14HqPwvDAwD00uB1xQNu0iGjOwZzSVvWSaLsqPQ"
    )
    parser.add_argument("-s", "--tournament_name", type=str, default='Computer_Tour')
    parser.add_argument("-r", "--round_name", type=str, default='Computer_Round')
    parser.add_argument("-m", "--match_name", type=str, default='bird')
    parser.add_argument("-p", "--mode", type=str, default='r')
    args = parser.parse_args()
    return args


def save(image, filename):
    cv2.imwrite(filename, image)

def read(socket: Socket, tournament_name, round_name, match_name, get_img_info=False):
    tournaments = socket.get_tournament()
    for tournament in tournaments:
        if tournament['name'] == tournament_name:
            tournament_id = tournament['id']
            break
    tournament_info = socket.get_tournament_info(tournament_id)
    rounds = tournament_info['Rounds']
    for round in rounds:
        if round['name'] == round_name:
            round_id = round['id']
            break
    round_info = socket.get_round_info(round_id)
    matches = round_info['Matches']
    match_id = None
    for match in matches:
        if match['name'] == match_name:
            match_id = match['id']
            break
    if match_id is None:
        print("No match found")
        return
    match_info = socket.get_match_info(match_id)
    id_challenge = match_info['id_challenge']
    # id_challenge = 3
    # challenge_info = socket.get_challenge_raw_info(id_challenge)
    # image_blocks = socket.get_challenge_image_info(id_challenge)
    
    game_info = GameInfo()
    game_info.name = match_name
    game_info.challenge_id = id_challenge
    if not get_img_info:
        return game_info
    challenge_info, image_blocks = socket.get_raw_info(id_challenge)
    game_info.block_dim = tuple(challenge_info[1])
    game_info.max_n_selects = challenge_info[2][0]
    game_info.select_swap_ratio = challenge_info[3]
    game_info.image_size = challenge_info[4]
    game_info.max_image_point_value = challenge_info[5]
    game_info.original_block_size = image_blocks.shape[-3:-1]
    game_info.original_blocks = image_blocks.tolist()
    game_info.mode = 'rgb'
    return game_info

def main():
    args = parse_args()
    tournament_name = args.tournament_name
    round_name = args.round_name
    match_name = args.match_name
    args = parse_args()
    socket = Socket(args.token)
    if args.mode == 'r':
        game_info = read(socket, tournament_name, round_name, match_name, get_img_info=True)
    else:
        game_info = read(socket, tournament_name, round_name, match_name)
    if args.mode == 'r':
        game_info = read(socket, tournament_name, round_name, match_name, get_img_info=True)
        game_info.save_to_json()
    elif args.mode == 'w':
        file_path = os.path.join(args.sol_path, args.match_name + '.txt')
        f = open(file_path, 'r')
        lines = f.readlines()
        lines = [line.replace('\n', '') for line in lines]
        data_text = '\n'.join(lines)
        data_text  = data_text.encode('utf-8')
        res = socket.send(challenge_id=game_info.challenge_id, data_text=data_text)
        print(res)
    elif args.mode == 'show':
        solution = socket.get_all_answer_info(challenge_id=game_info.challenge_id)
        print(json.dumps(solution, indent = 1))
    elif args.mode == 'del':
        res = socket.del_all_answer(challenge_id=game_info.challenge_id)
        print(res)
    else:
        print('mode invalid, please use -m w or -m r')
    
if __name__ == '__main__':
    main()