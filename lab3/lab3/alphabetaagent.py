import random

from lab3.exceptions import AgentException


class AlphaBetaAgent:
    def __init__(self, my_token='o'):
        self.my_token = my_token

    def decide(self, connect4, depth=3):
        if connect4.who_moves != self.my_token:
            raise AgentException('not my round')

        alpha = float('-inf')
        best_move = None
        best_score = float('-inf')

        for move in connect4.possible_drops():
            connect4_copy = connect4.deep_copy()
            connect4_copy.drop_token(move)

            score = self.alpha_beta(connect4_copy, 0, depth-1, alpha, float("inf"))

            if score > best_score:
                best_move = move
                best_score = score

            alpha = max(alpha, best_score)

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

    def alpha_beta(self, connect4, x, d, alpha, beta):
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
            #return 0
        elif x == 1:
            score = float("-inf")
            for move in connect4.possible_drops():
                connect4_copy = connect4.deep_copy()
                connect4_copy.drop_token(move)
                score = max(score, self.alpha_beta(connect4_copy, 0, d - 1, alpha, beta))
                alpha = max(alpha, score)
                if score >= beta:
                    break
            return score
        else:
            score = float("inf")
            for move in connect4.possible_drops():
                connect4_copy = connect4.deep_copy()
                connect4_copy.drop_token(move)
                score = min(score, self.alpha_beta(connect4_copy, 1, d - 1, alpha, beta))
                beta = min(beta, score)
                if score <= alpha:
                    break
            return score
