import copy, main, os, time, math, random, json
from copy import deepcopy
import hashlib
import asyncio
import pickle
# import keyboard
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import threading
from queue import Queue

# score = w1 * lines_cleared - w2 * holes - w3 * bumpiness - w4 * aggregate_height
MAX_WORKERS = 7
executor = ProcessPoolExecutor(max_workers=MAX_WORKERS)


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
    return holes/(len(board)*(len(board[0])-4))

def findAggHeight(board):
    heights = get_column_heights(board)
    return sum(heights)/(((len(board)-4)*len(board[0])))

def getPieceHeight(piecesPosition, board):
    height = 0
    for i in piecesPosition:
        if i[1] > height:
            height = i[1]
    return len(board)-height

def getHeightDiff(board):
    heights = get_column_heights(board)
    height_diff = max(heights) - min(heights)
    return height_diff/(len(board)-4)

def calculatePoints(old_board, new_board, w1, w2, w3):
    # OLD
    old_heights = get_column_heights(old_board)
    old_holes = findHolesInBoard(old_board)
    old_agg = sum(old_heights)
    old_bump = checkForRowsBumpiness(old_board)

    # NEW
    new_heights = get_column_heights(new_board)
    new_holes = findHolesInBoard(new_board)
    new_agg = sum(new_heights)
    new_bump = checkForRowsBumpiness(new_board)

    # DELTAS
    delta_holes = (new_holes - old_holes)
    delta_bump = new_bump - old_bump
    delta_agg = new_agg - old_agg

    # FINAL SCORE (mixing delta and absolute values)
    score = (
        - w1 * (0.5 * delta_holes + 0.5 * new_holes)
        - w2 * (0.5 * delta_agg + 0.5 * new_agg)
        - w3 * (0.5 * delta_bump + 0.5 * new_bump)
    )
    return score

def flattenMatrix(matrix):
    flat = [item for row in matrix for item in row.copy()]
    return flat

class trainingScene():
    def __init__(self, depth = 4):
        self.depth = depth
        self.board = main.TetrisBoard(10, 30)
        self.hasEnded = False
        self.weights = [1, 1, 1, 1]
        self.totalSteps = 0
        self.linesClearedByPoints = {0:0, 40:1, 100:2, 300:3, 1200:4}

    def clone(self):
        new_scene = trainingScene()
        new_scene.board = self.board.clone()
        new_scene.hasEnded = self.hasEnded
        new_scene.weights = self.weights[:]
        new_scene.totalSteps = self.totalSteps
        new_scene.linesClearedByPoints = self.linesClearedByPoints.copy()
        # new_scene.cache = self.cache
        new_scene.depth = self.depth
        return new_scene

    def randomizeWeights(self):
        for i in range(len(self.weights)):
            self.weights[i] += random.uniform(-1, 1)
            if self.weights[i] <= 0:
                self.weights[i] = 2

    def getPossibleBoard(self, action, board=None):
        board = board if board else self.board
        phantomBoard = board.clone()
        a = phantomBoard.step(action + 'l')
        return phantomBoard, a

    def getPossibleScene(self, action, tr=None):
        tr = tr if tr else self
        phantomScene = tr.clone()
        phantomScene.board.step(action + 'l')
        return phantomScene

    def getRawCalculation(self, oldBoard, action):
        boardPoints = self.board.clearedLines
        phantomBoard, a = self.getPossibleBoard((action + 'x'))
        points = 0
        linesCleared = max(0, phantomBoard.clearedLines - boardPoints)/4
        points = linesCleared * self.weights[3]
        
        calcPoints = calculatePoints(oldBoard.matrix, phantomBoard.matrix, *self.weights[:3])
        points += calcPoints
        return points, phantomBoard, a

    def findBestActions(self):
        scene = self.clone()
        if not scene.board.activePieces:
            return '', 0
        activePiece = scene.board.activePieces[0]
        
        rotationAmount = len(activePiece.forms)
        directions = ['a', 'd']
        currentBoard = scene.board.clone().matrix
        actions = []
        points = []
        for direction in directions:
            for rotation in range(rotationAmount):
                lastBoard = None
                mv = ''
                while lastBoard != currentBoard:
                    scene = self.clone()
                    currentBoard = lastBoard
                    action = ('r' * rotation) + mv
                    reward, board, a = self.getRawCalculation(scene.board, action)

                    if a:
                        points.append(-10000)
                    else:
                        points.append(reward)

                    actions.append(action + 'x')
                    lastBoard = board.matrix

                    # os.system('clear')
                    # board.renderPieces()
                    # print(rotationAmount, rotation)
                    # print(action)
                    # print(reward)

                    # print(lastBoard == currentBoard)

                    # input()

                    mv += direction
        
        actionIndex = points.index(max(points))

        return actions[actionIndex], points[actionIndex]

    def runGame(self, showBard=False, boardText='', on_update=None):
        start = time.time()
        print('start game')
        lost = False
        # lastStepsUpdate = 1000
        while not lost:
            self.totalSteps += 1
            boardPoints = self.board.points
            # actions = ['n', 'a', 'd', 'r', 'q']
            actions, reward = self.findBestActions()
            lost = self.board.step(actions + 'l') 
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
                print(actions)
                holesNumber = findHolesInBoard(self.board.matrix) * self.weights[0]
                avg = findAggHeight(self.board.matrix) * self.weights[1]
                bumpiness = checkForRowsBumpiness(self.board.matrix) * self.weights[2]
                print('Weights: ', self.weights)
                print('Holes: ', holesNumber * self.weights[0])
                print('Height: ', avg * self.weights[1])
                print('Bumpiness: ', bumpiness * self.weights[2])
                print()
                print('REWARD: ', reward)
                # for i in range(len(actions)):
                #     print('   ', actions[i], ':', estimatedRewards[i])
                # time.sleep(0.01)
                # input()
        end = time.time()
        print(f"finished Game : \nholes = {findHolesInBoard(self.board.matrix)}\ntime duration = {end-start:.2f}s\nSteps = {self.totalSteps}\n")
        return self.board.points


# depth = 4
weights = [1, 2, 2, 2]
# weights = [1,1,1,1]
weights = [1.8780708702495557, 5.069348613042007, 1.2332105271520826, 8.833876889186808]
weights = [20.030947971253617, 0.5683899675150448, 1.620549605794368, 0.27209254603798105]
weights = [20, 3, 6, 10]


# trsc.weights = [0.9751857387520118, 0.15877901592320565, 0.030644536663885003, 1.4482535692018872]
# trsc.weights = [0.5585452951752892, 0.7242329203865334, 0.44584922616611333, 1.8281236738434516]
# trsc.weights = [8, 4, 0.5, 1]
# trsc.weights = [5, 3, 1, 4]

def trainGenerations(gens=10000):
    os.system('clear')
    gmaes = 0
    for i in range(gens):
        trsc = trainingScene()
        trsc.weights = weights
        print()
        print(f'GENERATION: {i}')
        
        _ = trsc.runGame(True, f'GENERATION {i}, {gmaes}')
        input()
        


if __name__ == "__main__":

    trainGenerations()

# TASKS #
# check for new info instead of total info -> new info will make caching easier