import copy, os, time, random, numpy as np, math, platform
from copy import deepcopy
from pynput import keyboard
from threading import Thread

def sumLists(list_a, list_b):
    if len(list_a) != len(list_b):
        raise ValueError("The lists must be the same length")
    result = []
    for i in range(len(list_a)):
        result.append(list_a[i] + list_b[i])
    return result

# --------------------- DEFINING PIECES CLASSES ---------------------

class Piece():
    def __init__(self, name = '', forms = [[[0]*3]*3], position = [0, 0]):
        self.name = name
        self.forms = forms
        self.formIndex = 0
        self.position = position
        self.active = False

    def rotate(self):
        self.formIndex += 1
        self.formIndex = 0 if self.formIndex >= len(self.forms) else self.formIndex
    
    def unrotate(self):
        self.formIndex -= 1
        self.formIndex = len(self.forms) - 1 if self.formIndex < 0  else self.formIndex
    
    def getPiecePositions(self):
        form = self.forms[self.formIndex]
        positions = []
        for y in range(len(form)):
            for x in range(len(form[y])):
                item = form[y][x]
                if item == 0:
                    continue
                xpos = self.position[0] + x
                ypos = self.position[1] + y
                positions.append((xpos, ypos))
        return positions
    
class SquarePiece(Piece):
    def __init__(self, position = [0, 0]):
        forms = [
            [[1, 1],
             [1, 1]]
        ]
        super().__init__('Square piece', forms, position=position)

class LPiece(Piece):
    def __init__(self, position=[0,0]):
        forms = [
            [
                [0, 1, 0],
                [0, 1, 0],
                [1, 1, 0]
            ],
            [
                [1, 0, 0],
                [1, 1, 1],
                [0, 0, 0]
            ],
            [
                [0, 1, 1],
                [0, 1, 0],
                [0, 1, 0]
            ],
            [
                [0, 0, 0],
                [1, 1, 1],
                [0, 0, 1]
            ],
        ]
        super().__init__('L piece', forms, position=position)

class InvertedLPiece(Piece):
    def __init__(self, position=[0,0]):
        forms = [
            [
                [0, 1, 0],
                [0, 1, 0],
                [0, 1, 1]
            ],
            [
                [0, 0, 0],
                [1, 1, 1],
                [1, 0, 0]
            ],
            [
                [1, 1, 0],
                [0, 1, 0],
                [0, 1, 0]
            ],
            [
                [0, 0, 1],
                [1, 1, 1],
                [0, 0, 0]
            ]
        ]
        super().__init__('Inverted L piece', forms, position=position)

class SquigglePiece(Piece):
    def __init__(self, position=[0,0]):
        forms = [
            [
                [0, 0, 0],
                [0, 1, 1],
                [1, 1, 0]
            ],
            [
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0]
            ]
        ]
        super().__init__('Squiggle piece', forms, position=position)

class InvertedSquigglePiece(Piece):
    def __init__(self, position=[0,0]):
        forms = [
            [
                [0, 0, 0],
                [1, 1, 0],
                [0, 1, 1]
            ],
            [
                [0, 1, 0],
                [1, 1, 0],
                [1, 0, 0]
            ]
        ]
        super().__init__('Inverted squiggle piece', forms, position=position)
    
class TPiece(Piece):
    def __init__(self, position=[0,0]):
        forms = [
            [
                [0, 0, 0],
                [1, 1, 1],
                [0, 1, 0]
            ],
            [
                [0, 1, 0],
                [1, 1, 0],
                [0, 1, 0]
            ],
            [
                [0, 1, 0],
                [1, 1, 1],
                [0, 0, 0]
            ],
            [
                [0, 1, 0],
                [0, 1, 1],
                [0, 1, 0]
            ]
        ]
        super().__init__('T piece', forms, position=position)

class LongbarPiece(Piece):
    def __init__(self, position=[0,0]):
        forms = [
            [
                [0, 1, 0, 0],
                [0, 1, 0, 0],
                [0, 1, 0, 0],
                [0, 1, 0, 0]
            ],
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 1, 1, 1],
                [0, 0, 0, 0]
            ]
        ]
        super().__init__('Inverted L piece', forms, position=position)

pieces = [
    SquarePiece,
    LPiece,
    InvertedLPiece,
    LongbarPiece,
    SquigglePiece,
    InvertedSquigglePiece,
    TPiece
]

# -------------------------------------------------------------------------------------------------------------------------------- #
# --------------------- DEFINING BOARD CLASS ------------------------------------------ DEFINING BOARD CLASS --------------------- #
# -------------------------------------------------------------------------------------------------------------------------------- #

class TetrisBoard():
    def __init__(self, columns = 5, rows = 20):
        self.rowsLength = columns
        self.columnsLength = rows
        self.matrix = [[0]*columns for _ in range(rows)]
        self.activePieces = []
        self.points = 0
        self.pieces = [
                        SquarePiece,
                        LPiece,
                        InvertedLPiece,
                        LongbarPiece,
                        SquigglePiece,
                        InvertedSquigglePiece,
                        TPiece
                    ]
        self.nextPiece = random.choice(self.pieces)
        self.clearedLines = 0
    
    def clone(self):
        new_board = TetrisBoard(self.rowsLength, self.columnsLength)
        new_board.matrix = [row[:] for row in self.matrix]
        new_board.points = self.points
        new_board.activePieces = copy.deepcopy(self.activePieces)
        new_board.nextPiece = self.nextPiece
        return new_board

    def renderBoard(self, board = None):
        board = board if board else self.matrix
        board = deepcopy(board)
        prettyBoard = deepcopy(board)
        a = 0
        for i in range(len(prettyBoard)):
            for j in range(len(prettyBoard[i])):
                a+= 1
                value = prettyBoard[i][j]
                # if value != 0:
                #     print(value)
                #     # input()
                prettyBoard[i][j] = ' . ' if value == 0 else '[=]'
        # print(a)
        for row in prettyBoard:
            rowString = '| '
            for item in row:
                rowString += item
            rowString += ' |'
            print(rowString)

        # for i in board:
        #     print(i)
        return 'board shown'

    def getPositionValue(self, position, board = None):
        board = board if board else self.matrix

        try:
            value = deepcopy(board[position[1]][position[0]])
            # tt = deepcopy(board)
            # tt[position[1]][position[0]] = 1
            # self.renderBoard(tt)
            return value
        except IndexError as e:
            return f"position out of range, {e}"
    
    def changePositionValue(self, position, board = None):
        board = board if board else self.matrix

        board[position[1]][position[0]] = 1 if self.getPositionValue(position, board) == 0 else 0
        return board

    def getBoardWithPieces(self):
        pieces = self.activePieces
        renderBoard = deepcopy(self.matrix)
        for piece in pieces:
            positions = piece.getPiecePositions()
            for position in positions:
                # print(position)
                # print(renderBoard)
                renderBoard = self.changePositionValue(position, renderBoard)
                # print(renderBoard)
        return renderBoard


    def renderPieces(self):
        self.renderBoard(self.getBoardWithPieces())

    def spawnPiece(self, piece : Piece):
        xSpawnPos = math.floor(self.rowsLength/2) - 1
        self.activePieces.append(piece(position=[xSpawnPos,0]))

    def isPossiblePiecePosition(self, piece : Piece, board = None):
        board = board if board else self.matrix
        positions = piece.getPiecePositions()
        for position in positions:
            x = position[0]
            y = position[1]

            if x >= self.rowsLength or x < 0 or y >= self.columnsLength or y < 0 or self.getPositionValue((x, y), board) == 1:
                return False
        return True
            
    def runActionOnPiece(self, action, piece : Piece, board = None):
        board = board if board else self.matrix

        phantomPiece = deepcopy(piece)
        action = action.lower()
        movements = {'a' : [-1, 0], 'd' : [1, 0], 'l' : [0, 1]}
        

        if action in movements.keys():
            phantomPiece.position = sumLists(phantomPiece.position, movements[action])
            # print(action, movements[action])
            possible = self.isPossiblePiecePosition(phantomPiece, board)
            if not possible and action == "l":
                return True
            
            if possible:
                piece.position = phantomPiece.position
                return False
            
            return False
        
        if action == 'r' or action == 'q':
            if action == 'q':
                phantomPiece.unrotate()
            else:
                phantomPiece.rotate()
            possible = self.isPossiblePiecePosition(phantomPiece, board)
            while not possible:
                if action == 'q':
                    phantomPiece.unrotate()
                else:
                    phantomPiece.rotate()
                possible = self.isPossiblePiecePosition(phantomPiece, board)
            if possible:
                piece.formIndex = phantomPiece.formIndex
            return False
        
        if action == 'x':
            x = self.runActionOnPiece('l', piece, board)
            while x == False:
                x = self.runActionOnPiece('l', piece, board)
            return x
        return False

    def checkRows(self, board = None):
        board = board if board else self.matrix
        number = 0
        emptyRow = [0] * len(board[0])
        # print(emptyRow)
        completeRow = [1] * len(board[0])
        for row in board:
            complete = row == completeRow
            if complete:
                number +=1

        for i in range(number):
            board.remove(completeRow)
            board.insert(0, deepcopy(emptyRow))
            # print(board)
            # input()
        if number == 1:
            self.points += 40
            self.clearedLines += 1
        if number == 2:
            self.points += 100
            self.clearedLines += 2
        if number == 3:
            self.points += 300
            self.clearedLines += 3
        if number == 4:
            self.points += 1200
            self.clearedLines += 4
        return board



    def step(self, inp):
        # print(inp)
        if 's' in inp:
            inpList = list(inp)
            for i in range(len(inpList)):
                inpList[i] = 'l' if inpList[i] == 's' else inpList[i]
            inp = ''.join(inpList)
                    
        inp = inp
        if len(self.activePieces) == 0:
            self.spawnPiece(deepcopy(self.nextPiece))
            self.nextPiece = random.choice(self.pieces)
        actions = list(inp)
        lost = False
        for action in actions:
            for piece in self.activePieces:
                stop = self.runActionOnPiece(action, piece)
                if stop:
                    positions = piece.getPiecePositions()
                    for position in positions:
                        self.changePositionValue(position)
                    self.activePieces.remove(piece)
                    self.matrix = self.checkRows()
                    self.step('l')
                if 1 in self.matrix[3]:
                    lost = True
                    return lost
        return lost
    


if __name__ == "__main__":
    board = TetrisBoard(10)

    level = math.floor(board.clearedLines/10)

    inp = ''

    acts = ['a', 'd', 's', 'r', 'x']

    pressed = []
    lost =  False
    def on_press(key):
        global board, lost, level
        try:
            if key.char in acts:
                a = board.step(key.char)
                os.system('clear' if platform.system() == 'Darwin' else 'cls')
                print(f"""
--------------------------------
POINTS: {board.points}
--------------------------------
NEXT PIECE: {board.nextPiece().name}
""")
                board.renderBoard(board.nextPiece().forms[0])
                print(f"""
--------------------------------
LEVEL: {level}
""")
                board.renderPieces()
        except:
            pass



    listener = keyboard.Listener(on_press=on_press)

    listener.start()

    while inp != "hell":
        os.system('clear' if platform.system() == 'Darwin' else 'cls')
        if lost:
            print("YOU LOST BROOOOOO")
            break
        lost = board.step('l')
        print(f"""
--------------------------------
POINTS: {board.points}
--------------------------------
NEXT PIECE: {board.nextPiece().name}
    """)
        board.renderBoard(board.nextPiece().forms[0])
        print(f"""
--------------------------------
LEVEL: {level}
                    """)
        # inp = ''.join(pressed) if pressed else 'n'

        board.renderPieces()
        if lost:
            print("YOU LOST BROOOOOO")
            break
        time.sleep(1)
    
    
        