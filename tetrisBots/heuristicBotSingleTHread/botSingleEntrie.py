import copy, main, os, time, math, random
from copy import deepcopy
import numpy as np

def get_column_heights(board):
    heights = []
    num_rows = len(board)
    num_cols = len(board[0])
    for col in range(num_cols):
        col_height = 0
        for row in range(num_rows):
            if board[row][col] != 0:
                col_height = num_rows - row
                break
        heights.append(col_height)
    return heights

def checkForRowsBumpiness(board):
    heights = get_column_heights(board)
    bumpiness = sum(abs(heights[i] - heights[i+1]) for i in range(len(heights) - 1))
    max_bumpiness = (len(board[0]) - 4) * len(board)
    return bumpiness / max_bumpiness

def findHolesInBoard(board):
    holes = 0
    for x in range(len(board[0])):
        foundHole = False
        for y in range(len(board)):
            positionValue = board[y][x]
            if positionValue == 1 or y <= 0:
                continue
            upValue = board[y-1][x]
            if upValue == 1 or foundHole:
                foundHole = True
                holes += 1
    return holes / (len(board) * (len(board[0]) - 4))

def findAggHeight(board):
    heights = get_column_heights(board)
    return sum(heights) / (len(board) * (len(board[0]) - 4))

def calculatePoints(tetrisBoard : main.TetrisBoard, w1, w2, w3):
    board = tetrisBoard.matrix
    heights = get_column_heights(board)
    holes = findHolesInBoard(board)
    avgh = sum(heights)
    bumpiness = checkForRowsBumpiness(board)
    points = 0
    points -= holes * w1
    points -= bumpiness * w3
    points -= avgh * w2
    return points

class trainingScene():
    def __init__(self):
        self.board = main.TetrisBoard(8, 20)
        self.hasEnded = False
        self.weights = [1, 1, 1, 1]
        self.totalSteps = 0
        self.linesClearedByPoints = {0:0, 40:1, 100:2, 300:3, 1200:4}

    def randomizeWeights(self):
        for i in range(len(self.weights)):
            self.weights[i] += random.uniform(-1, 1)

    def getPossibleBoard(self, action, board=None):
        board = board if board else self.board
        phantomBoard = deepcopy(board)
        phantomBoard.step(action + 'l')
        return phantomBoard

    def getPossibleScene(self, action, tr=None):
        tr = tr if tr else self
        phantomScene = deepcopy(tr)
        phantomScene.board.step(action + 'l')
        return phantomScene

    def getRawCalculation(self, action, p=''):
        boardPoints = self.board.points
        phantomBoard = self.getPossibleBoard((action + 'x'))
        points = 0
        pointsGained = max(0, phantomBoard.points - boardPoints)
        if pointsGained in self.linesClearedByPoints:
            points += self.linesClearedByPoints[pointsGained] * self.weights[3]
        else:
            points += 1 * self.weights[3]
        points += calculatePoints(phantomBoard, *self.weights[:3])
        return points

    def testActionPath(self, trainingScene, actionj, depth=0, pastActions=''):
        if len(trainingScene.board.activePieces) == 0:
            return 0
        if depth >= 3:
            return trainingScene.getRawCalculation(actionj, pastActions)

        actions = ['a', 'd', 'r', 'q']
        tr = trainingScene.getPossibleScene(actionj)
        rewards = [self.testActionPath(tr, a, depth + 1, pastActions + a) for a in actions]
        rewards.append(tr.getRawCalculation('n', pastActions))
        return max(rewards)

    def runGame(self, showBard=False, boardText='', on_update=None):
        start = time.time()
        lost = False
        print('start')
        while not lost:
            self.totalSteps += 1
            boardPoints = self.board.points
            actions = ['n', 'a', 'd', 'r', 'q']
            estimatedRewards = [self.testActionPath(self, a) for a in actions]
            index = estimatedRewards.index(max(estimatedRewards))
            action = actions[index]
            lost = self.board.step(action)
            if on_update:
                on_update()
            if showBard:
                os.system('clear')
                print(boardText)
                print(f"""
-----------------------------
POINTS: {self.board.points}
-----------------------------
NEXT PIECE: {self.board.nextPiece}
{self.board.activePieces}
""")
                self.board.renderPieces()
                print(action)
                holesNumber = findHolesInBoard(self.board.matrix)
                avg = findAggHeight(self.board.matrix)
                bumpiness = checkForRowsBumpiness(self.board.matrix)
                print('Weights: ', self.weights)
                print('Holes: ', holesNumber)
                print('Height: ', avg)
                print('Bumpiness: ', bumpiness)
                for i in range(len(actions)):
                    print('   ', actions[i], ':', estimatedRewards[i])
                time.sleep(0.001)
        end = time.time()
        print(f"finished Game : \nholes = {findHolesInBoard(self.board.matrix)}\ntime duration = {end-start:.2f}s\nSteps = {self.totalSteps}\n")
        return self.board.points

def trainGenerations(gens=10000):
    for i in range(gens):
        trsc = trainingScene()
        _ = trsc.runGame(True, f'GENERATION {i}')
        time.sleep(1)
        input('continue?')

if __name__ == "__main__":
    trainGenerations()