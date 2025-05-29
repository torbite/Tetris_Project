import copy, os, time, random
from copy import deepcopy

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

    def renderPieces(self):
        pieces = self.activePieces
        renderBoard = deepcopy(self.matrix)
        for piece in pieces:
            positions = piece.getPiecePositions()
            for position in positions:
                # print(position)
                # print(renderBoard)
                renderBoard = self.changePositionValue(position, renderBoard)
                # print(renderBoard)
            
        self.renderBoard(renderBoard)

    def spawnPiece(self, piece : Piece):
        self.activePieces.append(piece(position=[0,0]))

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
        movements = {'a' : [-1, 0], 'd' : [1, 0], 's' : [0, 1]}
        

        if action in movements.keys():
            phantomPiece.position = sumLists(phantomPiece.position, movements[action])
            # print(action, movements[action])
            possible = self.isPossiblePiecePosition(phantomPiece, board)
            if not possible and action == "s":
                return True
            
            if possible:
                piece.position = phantomPiece.position
                return False
            
            return False
        
        if action == 'r':
            phantomPiece.rotate()
            possible = self.isPossiblePiecePosition(phantomPiece, board)
            while not possible:
                phantomPiece.rotate()
                possible = self.isPossiblePiecePosition(phantomPiece, board)
            if possible:
                piece.formIndex = phantomPiece.formIndex
            return False
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
        if number == 2:
            self.points += 100
        if number == 3:
            self.points += 300
        if number == 4:
            self.points += 1200
        return board



    def step(self, inp):
        # print(inp)
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
                if 1 in self.matrix[3]:
                    lost = True
                    return lost
        return lost
    


    


board = TetrisBoard()
board.spawnPiece(random.choice(pieces))
# board.spawnPiece(LongbarPiece)
inp = ''
while inp != "hell":
    os.system('clear')
    print(f"""
--------------------------------
POINTS: {board.points}
--------------------------------
""")
    a = board.step(inp)
    b = board.step('s')
    board.renderPieces()
    if a or b:
        print("YOU LOST BROOOOOO")
        break
    inp = input()
    if len(board.activePieces) == 0:
        board.spawnPiece(random.choice(pieces))
        # board.spawnPiece(LongbarPiece)
#for r in range(board.rows):
#    for c in range(board.columns):
#        os.system('cls')
#        print(board.getPositionValue((c, r)))
#        time.sleep(0.1)


