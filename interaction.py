import os
import cv2
from src.request import Socket
from src.recover.environment import GameInfo

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sol-path", type=str, default="output/solutions")
    parser.add_argument("-f", "--item-path", type=str, default="SCI_4x4.txt")
    parser.add_argument("--model-name", type=str, default="model_2_0.pt")
    parser.add_argument("--output-path", type=str, default="./output/recovered_images/")
    parser.add_argument( "--token", type=str, 
        default="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6NiwiaWF0IjoxNjQxOTIzMjU0LCJleHAiOjE2NDE5NDEyNTR9.OjH-9s87P5lqIn7CU6AJ2-vVGRJ-fhs0iX4cSV_CJKs"
    )
    parser.add_argument("-s", "--tournament_name", type=str, default='BKA_Tour')
    parser.add_argument("-r", "--round_name", type=str, default='Round_BKA')
    parser.add_argument("-m", "--match_name", type=str, default='Match_BKA_4x4')
    parser.add_argument("-p", "--mode", type=str, default='w')
    args = parser.parse_args()
    return args


def save(image, filename):
    cv2.imwrite(filename, image)

def read(socket, tournament_name, round_name, match_name):
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
    for match in matches:
        if match['name'] == match_name:
            match_id = match['id']
            break
    match_info = socket.get_match_info(match_id)
    id_challenge = match_info['id_challenge']
    challenge_info = socket.get_challenge_raw_info(id_challenge)
    image_blocks = socket.get_challenge_image_info(id_challenge)
    
    game_info = GameInfo()
    game_info.name = challenge_info[0][0]
    game_info.block_dim = tuple(challenge_info[1])
    game_info.max_n_chooses = challenge_info[2][0]
    game_info.choose_swap_ratio = challenge_info[3]
    game_info.image_size = challenge_info[4]
    game_info.max_image_point_value = challenge_info[5]
    game_info.original_block_size = image_blocks[0].shape[:2]
    image_blocks = image_blocks.reshape(game_info.block_dim[0], game_info.block_dim[1],
                   game_info.original_block_size[0], game_info.original_block_size[1], -1)
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
    if args.mode == 'read':
        game_info = read(socket, tournament_name, round_name, match_name)
        game_info.save_to_json()
    else:
        file_path = os.path.join(args.sol_path, args.item_path)
        f = open(file_path, 'r')
        lines = f.readlines()
        data_text = ''.join(lines)
        res = socket.send(challenge_id=6, data_text=data_text)
        print(res)
        solution = socket.get_all_answer_info(challenge_id=6)
        print(solution)
    
if __name__ == '__main__':
    main()