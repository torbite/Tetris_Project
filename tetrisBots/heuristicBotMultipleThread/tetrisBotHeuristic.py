import copy, main, os, time, math, random, json
from copy import deepcopy
import hashlib
import asyncio
import pickle
import keyboard
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
    return sum(heights)/((len(board)*(len(board[0])))-4)

def getPieceHeight(piecesPosition, board):
    height = 0
    for i in piecesPosition:
        if i[1] > height:
            height = i[1]
    return len(board)-height

def calculatePoints(tetrisBoard : main.TetrisBoard, w1, w2, w3):
    board = tetrisBoard.matrix
    heights = get_column_heights(board)
    holes = findHolesInBoard(board)
    avgh = sum(heights)
    bumpiness = checkForRowsBumpiness(board)
    points = 0
    points -= holes * w1
    points -= bumpiness * w2
    points -= avgh * w3
    return points

def simulate_action(tr_data):
    scene_data, weights, action, linesClearedByPoints = tr_data
    tr = deepcopy(scene_data)
    tr.weights = weights
    boardPoints = tr.board.points
    phantomBoard = tr.getPossibleBoard((action + 'x'))
    pointsGained = max(0, phantomBoard.points - boardPoints)
    if pointsGained in linesClearedByPoints:
        reward = linesClearedByPoints[pointsGained] * weights[3]
    else:
        reward = 1 * weights[3]
    reward += calculatePoints(phantomBoard, *weights[:3])
    return reward

def flattenMatrix(matrix):
    flat = [item for row in matrix for item in row.copy()]
    return flat

def hash_key(matrix, action, piece_name, pos, form_index):
    flat = flattenMatrix(matrix)
    key_data = flat + [action] + [piece_name] + pos + [form_index]
    key_str = json.dumps(key_data)  # consistent string representation
    return hashlib.md5(key_str.encode()).hexdigest()


class trainingScene():
    def __init__(self, depth = 4):
        self.depth = depth
        self.board = main.TetrisBoard(6, 20)
        self.hasEnded = False
        self.weights = [1, 1, 1, 1]
        self.totalSteps = 0
        self.cache = {}
        self.collecting = True
        self.linesClearedByPoints = {0:0, 40:1, 100:2, 300:3, 1200:4}

    def clone(self):
        new_scene = trainingScene()
        new_scene.board = self.board.clone()
        new_scene.hasEnded = self.hasEnded
        new_scene.weights = self.weights[:]
        new_scene.totalSteps = self.totalSteps
        new_scene.linesClearedByPoints = self.linesClearedByPoints.copy()
        new_scene.cache = self.cache
        new_scene.depth = self.depth
        return new_scene

    def randomizeWeights(self):
        for i in range(len(self.weights)):
            self.weights[i] += random.uniform(-1, 1)

    def getPossibleBoard(self, action, board=None):
        board = board if board else self.board
        phantomBoard = board.clone()
        phantomBoard.step(action + 'l')
        return phantomBoard

    def getPossibleScene(self, action, tr=None):
        tr = tr if tr else self
        phantomScene = tr.clone()
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
        calcPoints = calculatePoints(phantomBoard, *self.weights[:3])
        points += calcPoints
        # print(calcPoints)
        return points

    async def testActionPath(self, trainingScene, actionj, depth=0, pastActions=''):
        if len(trainingScene.board.activePieces) == 0:
            return 0
        

        if depth == 0:
            piece = self.board.activePieces[0]
            pName = piece.name
            pPos = piece.position
            pForm = piece.formIndex
            # if self.collecting:
                
            key = hash_key(self.board.matrix, actionj, pName, pPos, pForm)
            if key in self.cache.keys():
                return self.cache[key]


        if depth >= self.depth:
            # return 0
            return await asyncio.get_event_loop().run_in_executor(
                executor,
                simulate_action,
                (trainingScene, trainingScene.weights, actionj, self.linesClearedByPoints)
            )
        actions = ['a', 'd', 'r', 'q']
        if pastActions != '':
            if 'q' in list(pastActions)[len(pastActions)-1]:
                actions.remove('r')
            if 'd' in list(pastActions)[len(pastActions)-1]:
                actions.remove('a')
            if 'r' in list(pastActions)[len(pastActions)-1]:
                actions.remove('q')
            if 'a' in list(pastActions)[len(pastActions)-1]:
                actions.remove('d')
        tr = trainingScene.getPossibleScene(actionj)
        tasks = [self.testActionPath(tr, a, depth + 1, pastActions + a) for a in actions]
        rewards = await asyncio.gather(*tasks)
        rewards.append(tr.getRawCalculation('n', pastActions))
        if depth == 0:
            if not self.collecting:
                self.cache[key] = max(rewards)
            else:
                addCache(self.board.matrix, actionj, pName, pPos, pForm,max(rewards))

        return max(rewards)

    def runGame(self, showBard=False, boardText='', on_update=None):
        start = time.time()
        print('start game')
        lost = False
        # lastStepsUpdate = 1000
        while not lost:
            self.totalSteps += 1
            boardPoints = self.board.points
            actions = ['n', 'a', 'd', 'r', 'q']
            estimatedRewards = [asyncio.run(self.testActionPath(self, a)) for a in actions]
            index = estimatedRewards.index(max(estimatedRewards))
            action = actions[index]
            lost = self.board.step(action + 'l') 
            if on_update:
                on_update()

            # if self.totalSteps > lastStepsUpdate:
            #     dumpCachePickle(self.cache, depth, self.weights)

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
                holesNumber = findHolesInBoard(self.board.matrix) * self.weights[0]
                avg = findAggHeight(self.board.matrix) * self.weights[1]
                bumpiness = checkForRowsBumpiness(self.board.matrix) * self.weights[2]
                print('Weights: ', self.weights)
                print('Holes: ', holesNumber * self.weights[0])
                print('Height: ', avg * self.weights[1])
                print('Bumpiness: ', bumpiness * self.weights[2])
                print()
                cl = self.cache if not self.collecting else allCache
                print('CacheLength: ', len(allCache))
                print()
                for i in range(len(actions)):
                    print('   ', actions[i], ':', estimatedRewards[i])
                # time.sleep(0.001)
        end = time.time()
        print(f"finished Game : \nholes = {findHolesInBoard(self.board.matrix)}\ntime duration = {end-start:.2f}s\nSteps = {self.totalSteps}\n")
        return self.board.points

def dumpCacheJson(cache, depth):
    global allCache
    allCache = {**allCache, **cache}
    print('dumped')
    fileName = f'{depth}depthCache.json'
    with open(fileName, 'w') as f:
        json.dump(allCache, f, indent=4)

def dumpCachePickle(cache, depth, weights):
    global allCache
    allCache = {**allCache, **cache}
    a = ''
    for i in weights:
        a += str(i)
    with open(f'{depth}{a}depthCache.pkl', 'wb') as f:
        pickle.dump(allCache, f)
    
def loadCache(depth, weights):
    try:
        a = ''
        for i in weights:
            a += str(i)
        with open(f'{depth}{a}depthCache.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        return {}
    
cache_queue = Queue()
cache_lock = threading.Lock()

allCache = {}
depth = 3
weights = [3, 2, 2, 5]

def addCache(state, action, pName, pPos, pForm, reward):
    # global allCache
    # input(allCache)
    cache_queue.put((state, action, pName, pPos, pForm, reward))

def cache_worker():
    global allCache
    while True:
        item = cache_queue.get()  # blocks until item available
        if item is None:
            break  # graceful shutdown

        state, action, pName, pPos, pForm, reward = item
        key = hash_key(state, action, pName, pPos, pForm)

        with cache_lock:
            if key not in allCache:
                allCache[key] = reward

        cache_queue.task_done()

# trsc.weights = [0.9751857387520118, 0.15877901592320565, 0.030644536663885003, 1.4482535692018872]
# trsc.weights = [0.5585452951752892, 0.7242329203865334, 0.44584922616611333, 1.8281236738434516]
# trsc.weights = [8, 4, 0.5, 1]
# trsc.weights = [5, 3, 1, 4]

allCache = loadCache(depth, weights)
def trainGenerations(gens=10000):
    global allCache
    collectingCache = True
    for i in range(gens):
        trsc = trainingScene()
        if not collectingCache:
            trsc.cache = allCache
        trsc
        trsc.weights = weights
        trsc.depth = depth

        _ = trsc.runGame(True, f'GENERATION {i}')
        gameCache = trsc.cache
        # time.sleep(1)
        # rsp = input('continue? ')
        if collectingCache:
            gameCache = allCache
        dumpCachePickle(gameCache, depth, weights)
        # if rsp == 'cc':
        #     collectingCache = True
        # if rsp == 'ss':
        #     collectingCache = False
        # if rsp == 'st':
        #     break


if __name__ == "__main__":
    worker_thread = threading.Thread(target=cache_worker, daemon=True)
    worker_thread.start()

    trainGenerations()

# TASKS #
# add more conditions, such as 