import random

from lab3.exceptions import AgentException


class MinMaxAgentH:
    def __init__(self, my_token='o'):
        self.my_token = my_token

    def decide(self, connect4, depth=3):
        if connect4.who_moves != self.my_token:
            raise AgentException('not my round')

        best_move = None
        best_score = -1

        for move in connect4.possible_drops():
            connect4_copy = connect4.deep_copy()
            connect4_copy.drop_token(move)

            score = self.min_max_h(connect4_copy, 0, depth)

            if score > best_score:
                best_move = move
                best_score = score

        return best_move

    def heuristic(self, connect4, d):
        all_seq = list(connect4.iter_fours())
        final_score = 0

        for seq in all_seq:
            enemy_token = 'x' if self.my_token == 'o' else 'o'
            player_count = 0
            enemy_count = 0
            empty_count = 0

            for token in seq:
                if token == self.my_token:
                    player_count += 1
                elif token == enemy_token:
                    enemy_count += 1
                else:
                    empty_count += 1

            if player_count == 4:
                return 1*d
            elif enemy_count == 4:
                return -1

            if player_count == 3 and empty_count == 1:
                final_score += 0.7
            if enemy_count == 3 and empty_count == 1:
                final_score -= 0.7
            if player_count == 2 and enemy_count == 2:
                final_score += 0.35
            if enemy_count == 2 and empty_count == 2:
                final_score -= 0.35
            if player_count == 1 and empty_count == 3:
                final_score += 0.1
            if enemy_count == 1 and empty_count == 3:
                final_score -= 0.1

        return final_score/len(all_seq)

    def min_max_h(self, connect4, x, d):
        game_over = connect4._check_game_over()
        if game_over:
            if connect4.wins is None:
                return 0
            elif connect4.wins == self.my_token:
                return 1
            elif connect4.wins != self.my_token:
                return -1
        elif d == 0:
            return self.heuristic(connect4, d)
        elif x == 1:
            mh_next = []
            for move in connect4.possible_drops():
                connect4_copy = connect4.deep_copy()
                connect4_copy.drop_token(move)
                mh_next.append(self.min_max_h(connect4_copy, 0, d - 1))
            return max(mh_next)
        else:
            mh_next = []
            for move in connect4.possible_drops():
                connect4_copy = connect4.deep_copy()
                connect4_copy.drop_token(move)
                mh_next.append(self.min_max_h(connect4_copy, 1, d - 1))
            return min(mh_next)
