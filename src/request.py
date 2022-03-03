import requests
import base64
import cv2
import requests
import numpy as np
from src.recover.environment import GameInfo
from src.data_helper import DataProcessor

# TOKEN = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6MSwiaWF0IjoxNjQ1OTU2MzYyLCJleHAiOjE2NDU5NzQzNjJ9.a_O5bxFBlPZZRamp5XaiLlxcH7dLbWoHlhP7cZGehGA'
END_POINT_API = 'http://112.137.129.202:8016'

class Socket:
    def __init__(self, token):
        self.token = token
        self.headers = {
            'Authorization': 'Bearer {}'.format(token)
        }

    def get_tournament(self):
        url = END_POINT_API + '/tournament'
        response = requests.get(url, headers=self.headers, verify=False)
        return response.json()
    
    def get_tournament_info(self, tournament_id):
        url = END_POINT_API + '/tournament/{}'.format(tournament_id)
        response = requests.get(url, headers=self.headers, verify=False)
        return response.json()
    
    def get_round_info(self, round_id):
        url = END_POINT_API + '/round/{}'.format(round_id)
        response = requests.get(url, headers=self.headers, verify=False)
        return response.json()
        
    def get_match_info(self, match_id):
        url = END_POINT_API + '/match/{}'.format(match_id)
        response = requests.get(url, headers=self.headers, verify=False)
        return response.json()
    
    def get_raw_info(self, challenge_id):
        url = END_POINT_API + '/challenge/raw-challenge/{}'.format(challenge_id)
        response = requests.get(url, headers=self.headers, verify=False)
        ppm_image = response.content
        with open('output/temp_image.ppm', 'wb') as f:
            f.write(ppm_image)
            
        headers = ppm_image.split(b'\n')[:6]
        for i, header in enumerate(headers):
            headers[i] = header.decode().replace('# ', '').split(' ')
            if i > 0:
                for j, x in enumerate(headers[i]):
                    headers[i][j] = int(x)
                    
        image = cv2.imread('output/temp_image.ppm', cv2.IMREAD_ANYCOLOR)
        print(headers)
        # headers[1][0], headers[1][1] = headers[1][1], headers[1][0]
        blocks = DataProcessor.split_image_to_blocks(image, (headers[1][0], headers[1][1]))
        return headers, blocks
    
    def get_challenge_raw_info(self, challenge_id):
        url = END_POINT_API + '/challenge/raw/{}'.format(challenge_id)
        response = requests.get(url, headers=self.headers, verify=False)
        contents = response.content.strip().decode('utf-8').replace('# ', '').split('\n')
        for i, content in enumerate(contents):
            contents[i] = content.split(' ')
            if i > 0:
                for j, x in enumerate(contents[i]):
                    contents[i][j] = int(x)
                    
        return contents
    def get_challenge_image_info(self, challenge_id):
        url = END_POINT_API + '/challenge/image/{}'.format(challenge_id)
        response = requests.get(url, headers=self.headers, verify=False)
        pieces = response.json()
        for i, piece in enumerate(pieces):
            b64_image = base64.b64decode(piece)
            piece = cv2.imdecode(np.frombuffer(b64_image, np.uint8), cv2.IMREAD_ANYCOLOR)
            pieces[i] = piece
        return np.array(pieces)

    def get_all_answer_info(self, challenge_id):
        url = END_POINT_API + '/solution/team/{}'.format(challenge_id)
        response = requests.get(url, headers=self.headers, verify=False)
        return response.json()
    
    def del_all_answer(self, challenge_id):
        url = END_POINT_API + '/solution/delete/{}'.format(challenge_id)
        response = requests.delete(url, headers=self.headers, verify=False)
        return response.json()
    
    def send(self, challenge_id, data_text):
        url = END_POINT_API + '/solution/submit/{}'.format(challenge_id)
        header = self.headers.copy()
        header['Content-Type'] = 'text/plain'
        response = requests.post(url, headers=header, data=data_text, verify=False)
        return response.json()
    