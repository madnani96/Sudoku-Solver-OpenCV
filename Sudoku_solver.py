def check(g, r, c, num):
	for x in range(9):
		if g[r][x] == num:
			return False
	for x in range(9):
		if g[x][c] == num:
			return False
	sr = r - r % 3
	sc = c - c % 3
	for i in range(3):
		for j in range(3):
			if g[sr+i][sc+j] == num:
				return False
	return True

def solveSudoku(g, r, c):
	if (r == 8 and c == 9):
		return True
	if c == 9:
		r += 1
		c = 0
	if g[r][c] > 0:
		return solveSudoku(g, r, c + 1)
	for num in range(1, 10, 1):
		if check(g, r, c, num):
			g[r][c] = num
			if solveSudoku(g, r, c + 1):
				return True
		g[r][c] = 0
	return False

def print_grid(arr):
	for i in range(0,9):
		for j in range(0,9):
			print(arr[i][j], end = " ")
		print()
def sudoku(grid):
	if (solveSudoku(grid, 0, 0)):
		print_grid(grid)
		return
	else:
		print("No Solution Exists ")
		return

