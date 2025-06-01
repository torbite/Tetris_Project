import copy, main, os, time, math
from copy import deepcopy
import asyncio
import tensorflow as tf
import numpy as np


def checkForHolesDistence(board):
    rowsamount = len(board)
    rowsDistances = []
    holesAmound = findHolesInBoard(board)
    if holesAmound <= 0:
        return 0
    for row in board:
        oneCount = row.count(1)
        zeroCount = row.count(0)
        if oneCount < 2 or zeroCount < 2:
            rowsDistances.append(0)
            continue

        lst =  (zeroCount, oneCount)
        numb = lst.index(min(lst))
        indices = [i for i, val in enumerate(row) if val == numb]
        distances = [j - i for i, j in zip(indices[:-1], indices[1:])]
        rowsDistances.append(sum(rowsDistances))

    
    return sum(rowsDistances)/holesAmound


def checkForRowsBumpiness(board):
    rowsHoles = []
    rowsamount = len(board)
    for row in board:
        if 1 not in row or 0 not in row:
            rowsHoles.append(0)
            continue
        oneCount = row.count(1)
        zeroCount = row.count(0)

        holes = min((oneCount, zeroCount))
        rowsHoles.append(holes)
    
    bumpiness = sum(rowsHoles)/rowsamount
    return bumpiness
    

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
    return holes

def findAvrgHeight(board):
    board_height = len(board)
    max_height = 0

    for col in range(len(board[0])):
        for row in range(board_height):  # de cima para baixo
            if board[row][col] != 0:
                height = board_height - row
                if height > max_height:
                    max_height = height
                break  # achou o topo da pilha nessa coluna

    return max_height / board_height if board_height > 0 else 0

def getPieceHeight(piecesPosition, board):
    height = 0
    for i in piecesPosition:
        if i[1] > height:
            height = i[1]

    actuallHeight = len(board)-height
    return actuallHeight


def calculatePoints(tetrisBoard : main.TetrisBoard):
    board = tetrisBoard.matrix
    points = 0

    holes = findHolesInBoard(board=board)
    avgh = findAvrgHeight(board)
    bumpiness = checkForRowsBumpiness(board)
    distances = checkForHolesDistence(board)

    piecesPos = tetrisBoard.activePieces[0].getPiecePositions()
    # pieceHeight = getPieceHeight(piecesPos, tetrisBoard.matrix)
    # points += (len(tetrisBoard.matrix) - pieceHeight)/len(tetrisBoard.matrix)*0.1
    # points -= 
    points -= distances
    points -= (holes)/(len(board)*len(board[0]))
    points -= bumpiness * avgh

    return points
    

class tetrisBotQLearning():
    def __init__(self):
        # consider adding: input_size = len(np.zeros((20,5)).flatten()) + 2 + 3  # board size + features + action
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(107,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1) 
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='mean_squared_error'
        )

        self.actionConverter = {
            'n' : [0,0,0,0],
            'a' : [1,0,0,0],
            'd' : [0,1,0,0],
            'r' : [0,0,1,0],
            'q' : [0,0,0,1]
        }

    def __call__(self, board: list, action: str, usefullInfo: list = []):
        flat_board = np.ravel(board)  # faster than np.array(...).flatten()
        action_vector = self.actionConverter[action]
        model_input = np.concatenate((flat_board, usefullInfo, action_vector)).reshape(1, -1)

        return self.model(model_input, training=False).numpy()[0][0]  # avoids .predict() overhead

    def train(self, X_, y_):
        X = np.array(X_)
        y = np.array(y_)

        self.model.fit(X, y, epochs = 75)
    
async def predict_async(model, board, action, usef):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, model, board, action, usef)
    


class trainingScene():
    def __init__(self, model : tetrisBotQLearning = tetrisBotQLearning()):
        self.model = model
        self.board = main.TetrisBoard()
        # state, action, momentReward, estimatedReward, nextstate
        self.hasEnded = False

    def getPossibleBoard(self, action, board = None):
        board = board if board else self.board
        phantomBoard = deepcopy(board)
        phantomBoard.step(action + 'l')
        # possibleBoard = phantomBoard
        return phantomBoard
    
    def getPossibleScene(self, action, tr = None):
        tr = tr if tr else self
        phantomScene = deepcopy(tr)
        phantomScene.board.step(action + 'l')
        # possibleBoard = phantomBoard
        return phantomScene

    async def getModelPredition(self, action):
        board = deepcopy(self)
        phantomBoard = board.getPossibleBoard((action + 'x'))
        possibleBoard = phantomBoard.getBoardWithPieces()
        holes = (findHolesInBoard(possibleBoard)*4)/(len(possibleBoard)*len(possibleBoard[0]))
        avg = findAvrgHeight(possibleBoard)

        piecesPos = phantomBoard.activePieces[0].getPiecePositions()

        pieceHeight = getPieceHeight(piecesPos, phantomBoard.matrix)

        usefullStuff = [holes, avg, pieceHeight]
        reward = await predict_async(self.model, phantomBoard.getBoardWithPieces(), action, usefullStuff)
        # estimatedRewards.append(reward)
        return reward

    def getRawCalculation(self, action, p = ''):
        boardPoints = self.board.points
        phantomBoard = self.getPossibleBoard((action + 'x'))

        points = 0
        pointsGained = 0
        if phantomBoard.points != boardPoints:
            pointsGained = phantomBoard.points - boardPoints
        points += pointsGained/175
        points += calculatePoints(phantomBoard)


        # os.system("clear")
        # phantomBoard.renderPieces()
        # input(str(points) + ' ' +  p + action)

        return points

    async def testActionPath(self, trainingScene, actionj,depth = 0, pastActions = '', wModel = False):
        if depth >= 3:
            if wModel:
                reward = await self.getModelPredition(actionj)
            else:
                reward = trainingScene.getRawCalculation(actionj, pastActions)
            return reward

        actions = ['n', 'a', 'd', 'r', 'q']
        rewards = []
        for action in actions:
            tr = trainingScene.getPossibleScene(actionj)
            reward = await trainingScene.testActionPath(tr, action, depth + 1, actionj, wModel)
            if isinstance(reward, list):
                reward = max(reward)
            rewards.append(reward)
        
        maxReward = max(rewards)

        return maxReward


    async def runGame(self, epsilon, showBard = False, boardText = '', withModel = True):
        print('game started')
        start = time.time()
        trainData = []
        lost = False
        while not lost:
            data = []
            data.append(self.board.getBoardWithPieces())
            boardPoints = self.board.points
            actions = ['n', 'a', 'd', 'r', 'q']
            estimatedRewards = []
            for action in actions:
                # reward = await self.getModelPredition(action)
                # reward = self.getRawCalculation(action)
                reward = await self.testActionPath(self, action, wModel=withModel)
                estimatedRewards.append(reward)




            if np.random.rand() < 0:
                index = np.random.randint(len(actions))
            else:
                index = estimatedRewards.index(max(estimatedRewards))
            data.append(index)
            action = actions[index]
            lost = self.board.step(action)

            # POINTS CALCULATION #
            points = 0
            pointsGained = 0
            if self.board.points != boardPoints:
                pointsGained = self.board.points - boardPoints
            points += pointsGained/175
            points += calculatePoints(self.board)
            
            piecesPos = self.board.activePieces[0].getPiecePositions()

            pieceHeight = getPieceHeight(piecesPos, self.board.matrix)

            holes = (findHolesInBoard(self.board.matrix)*4)/(len(self.board.matrix)*len(self.board.matrix[0]))
            avg = findAvrgHeight(self.board.matrix)

            usefullStuff = [holes, avg, pieceHeight]
            data.append(usefullStuff)


            if lost:
                data.append(-5)
            else:
                # print(points)
                data.append(points)


            data.append(estimatedRewards[index])
            data.append(self.board.getBoardWithPieces())

            trainData.append(data)
            if showBard:
                os.system('clear')
                print(epsilon)
                print(boardText)
                self.board.renderPieces()
                print(action)
                holesNumber = findHolesInBoard(self.board.matrix)
                print(holesNumber)
                print(avg)
                # print(usefullStuff)
                for i in range(len(actions)):
                    print('   ', actions[i], ':', estimatedRewards[i])
                # input()
                
                time.sleep(0.01)
        end = time.time()
        timeDuration = end - start
        print(f"finished Game : \nholes = {holesNumber} || {holes}\ntime duration = {timeDuration:.2f} seconds\n")
        return trainData, self.board.points

async def getTrainingData(trainingScenes, epsilon):
    tasks = [scene.runGame(epsilon) for scene in trainingScenes]
    results = await asyncio.gather(*tasks)
    return results

def trainGeneration(botsAmount, epsilon, oldBot : tetrisBotQLearning = tetrisBotQLearning()):
    trainingScenes = [trainingScene(deepcopy(oldBot)) for _ in range(botsAmount)]

    results = asyncio.run(getTrainingData(trainingScenes, epsilon))
    trainDataTotal = []
    modelPoints = []
    for trainData, points in results:
        trainDataTotal = trainDataTotal + trainData
        modelPoints.append(points)

    index = modelPoints.index(max(modelPoints))

    bestModel = trainingScenes[index].model
    X = []
    y = []
    for i in trainDataTotal:
        state = i[0]
        state = np.array(state).flatten()
        action = i[1]
        action = np.array([action])
        usefullStuff = i[2]
        momentReward = i[3]
        estimatedReward = i[4]
        nextState = i[5]
        nextState = np.array(nextState).flatten()

        reward = momentReward if momentReward != 0 else float(estimatedReward)

        action_str = ['n','a', 'd', 'r'][action[0]]
        action = np.array(oldBot.actionConverter[action_str])
        X.append(np.concatenate([state, usefullStuff, action]))
        y.append(reward)

    bestModel.train(X, y)
    return deepcopy(bestModel)

def trainGenerations(gens = 10000):
    bot = tetrisBotQLearning()
    for i in range(gens):
        epsilon = 0.5
        if i > 5 and i < 15:
            epsilon = 0.1
        elif i >= 15:
            epsilon = 0
        trsc = trainingScene(deepcopy(bot))
        asyncio.run(trsc.runGame(epsilon, True, f'GENERATION {i}', withModel=False))
        time.sleep(1)
        # input('continue?')
        # bot = trainGeneration(15, epsilon, deepcopy(bot))
    return bot
        
if __name__ == "__main__":
    trainGenerations()


### TASKS ###
# make so model choose based on when piece is on the ground
# make model see 2 moves into the future







