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
    delta_holes = new_holes - old_holes
    delta_bump = new_bump - old_bump
    delta_agg = new_agg - old_agg

    # SCORE (negative deltas are good)
    score = -w1 * delta_holes - w3 * delta_bump - delta_agg * w2
    return score

def simulate_action(tr_data):
    scene_data, weights, action, linesClearedByPoints, oldBoard = tr_data
    tr = deepcopy(scene_data)
    tr.weights = weights
    boardPoints = tr.board.points
    phantomBoard = tr.getPossibleBoard((action + 'x'))
    pointsGained = max(0, phantomBoard.points - boardPoints)
    if pointsGained in linesClearedByPoints:
        reward = linesClearedByPoints[pointsGained] * weights[3]
    else:
        reward = 1 * weights[3]
    reward += calculatePoints(oldBoard.matrix, phantomBoard.matrix, *weights[:3])
    return reward

def flattenMatrix(matrix):
    flat = [item for row in matrix for item in row.copy()]
    return flat

def hash_key(heights, holes, action, piece_name, pos, form_index):
    # flat = flattenMatrix(matrix)
    key_data = heights + [holes] + [action] + [piece_name] + pos + [form_index]
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
        # new_scene.cache = self.cache
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

    def getRawCalculation(self, oldBoard, action, p=''):
        boardPoints = self.board.points
        phantomBoard = self.getPossibleBoard((action + 'x'))
        points = 0
        pointsGained = max(0, phantomBoard.points - boardPoints)
        if pointsGained in self.linesClearedByPoints:
            points += self.linesClearedByPoints[pointsGained] * self.weights[3]
        else:
            points += 1 * self.weights[3]
        calcPoints = calculatePoints(oldBoard.matrix, phantomBoard.matrix, *self.weights[:3])
        points += calcPoints
        # print(calcPoints)
        return points

    async def testActionPath(self, trainingScene, actionj,oldBoard,depth=0, pastActions=''):
        if len(trainingScene.board.activePieces) == 0:
            return 0
        

        if depth == 0:
            state = self.board.matrix

            heights = get_column_heights(state)
            lowestHeight = min(heights)
            normlHeights = [h - lowestHeight for h in heights]
            holes = findHolesInBoard(state)

            piece = self.board.activePieces[0]
            pName = piece.name
            pPos = piece.position
            pForm = piece.formIndex
            # if self.collecting:
                
            key = hash_key(normlHeights,holes, actionj, pName, pPos, pForm)
            if key in self.cache.keys():
                return self.cache[key]


        if depth >= self.depth:
            # return 0
            return await asyncio.get_event_loop().run_in_executor(
                executor,
                simulate_action,
                (trainingScene, trainingScene.weights, actionj, self.linesClearedByPoints, oldBoard)
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
        tasks = [self.testActionPath(tr, a,oldBoard, depth + 1, pastActions + a) for a in actions]
        rewards = await asyncio.gather(*tasks)
        rewards.append(tr.getRawCalculation(oldBoard, 'n', pastActions))
        if depth == 0:
            addCache(holes, normlHeights, actionj, pName, pPos, pForm,max(rewards))

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
            estimatedRewards = [asyncio.run(self.testActionPath(self, a, self.board)) for a in actions]
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
                print('CacheLength: ', len(allCache))
                print()
                for i in range(len(actions)):
                    print('   ', actions[i], ':', estimatedRewards[i])
                # time.sleep(0.01)
        end = time.time()
        print(f"finished Game : \nholes = {findHolesInBoard(self.board.matrix)}\ntime duration = {end-start:.2f}s\nSteps = {self.totalSteps}\nCACHE LENGTH: {len(allCache)}")
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
depth = 4
weights = [3, 4, 2, 5]

def addCache(holes, heihgs, action, pName, pPos, pForm, reward):
    # global allCache
    # input(allCache)
    cache_queue.put((holes, heihgs, action, pName, pPos, pForm, reward))

def cache_worker():
    global allCache
    while True:
        item = cache_queue.get()  # blocks until item available
        if item is None:
            break  # graceful shutdown

        holes, heights, action, pName, pPos, pForm, reward = item
        key = hash_key(heights, holes, action, pName, pPos, pForm)

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
    os.system('clear')
    global allCache
    collectingCache = False
    gmaes = 0
    for i in range(gens):
        trsc = trainingScene()
        if not collectingCache:
            trsc.cache = allCache
        trsc
        trsc.weights = weights
        trsc.depth = depth
        al = len(allCache)
        print()
        print(f'GENERATION: {i}')
        if gmaes >= 50:
            _ = trsc.runGame(True, f'GENERATION {i}, {gmaes}')
        else:
            _ = trsc.runGame(True, f'GENERATION {i}, {gmaes}')
        print()
        alDps = len(allCache)
        delta = alDps - al
        if delta < 100:
            gmaes += 1
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
# check for new info instead of total info -> new info will make caching easier