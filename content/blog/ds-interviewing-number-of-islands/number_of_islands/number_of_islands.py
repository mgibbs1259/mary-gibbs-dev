import collections
from typing import List


class Solution:
    def get_num_islands(self, grid: List[List[str]], method: str = 'bfs') -> int:
        if not grid:
            return 0
        
        rows, cols = len(grid), len(grid[0])
        visited = set()
        num_islands = 0

        def perform_bfs(row: int, col: int) -> None:
            q = collections.deque()
            visited.add((row, col))
            q.append((row, col))

            while q:
                row, col = q.popleft()
                directions = [[1, 0], [-1, 0], [0, 1], [0, -1]]

                for direction_row, direction_col in directions:
                    r, c = row + direction_row, col + direction_col
                    if (r in range(rows) and c in range(cols) and
                            grid[r][c] == '1' and (r, c) not in visited):
                        q.append((r, c))
                        visited.add((r, c))
                        
        def perform_dfs(row: int, col: int) -> None:
            s = collections.deque()
            visited.add((row, col))
            s.append((row, col))

            while s:
                row, col = s.pop()
                directions = [[1, 0], [-1, 0], [0, 1], [0, -1]]

                for direction_row, direction_col in directions:
                    r, c = row + direction_row, col + direction_col
                    if (r in range(rows) and c in range(cols) and
                            grid[r][c] == '1' and (r, c) not in visited):
                        s.append((r, c))
                        visited.add((r, c))
        
        for row in range(rows):
            for col in range(cols):
                if grid[row][col] == '1' and (row, col) not in visited:
                    if method == 'bfs':
                        perform_bfs(row, col)
                    elif method == 'dfs':
                        perform_dfs(row, col)
                    else:
                        raise ValueError("Invalid method. Use 'bfs' or 'dfs'.")
                    num_islands += 1
                
        return num_islands


if __name__ == "__main__":
    # No grid
    grid_1 = []
    
    # Grid with single water cell
    grid_2 = [["0"]]
    
    # Grid with single land cell
    grid_3 = [["1"]]
    
    # Grid with alternating land and water cells, forming multiple small islands
    grid_4 = [["1", "0", "1"],
              ["0", "1", "0"],
              ["1", "0", "1"]]
    
    # Grid where a single island has a hole of water in the center
    grid_5 = [["1", "1", "1"],
              ["1", "0", "1"],
              ["1", "1", "1"]]
    
    # Random grid
    grid_6 = [["1","1","0","0","0"],
              ["1","1","0","0","0"],
              ["0","0","1","0","0"],
              ["0","0","0","1","1"]]
    
    soln = Solution()
    
    assert soln.get_num_islands(grid_1) == 0, "Error grid_1"
    assert soln.get_num_islands(grid_2) == 0, "Error grid_2"
    assert soln.get_num_islands(grid_3) == 1, "Error grid_3"
    assert soln.get_num_islands(grid_4) == 5, "Error grid_4"
    assert soln.get_num_islands(grid_5) == 1, "Error grid_5"
    assert soln.get_num_islands(grid_6) == 3, "Error grid_6"

    print("All test cases passed!")
    
    # BFS
    # Time complexity is O(M×N) where M is the number of rows and N is the number of columns
    # Space complexity is worst case O(min(M,N))
    
    # DFS
    # Time complexity is O(M×N) where M is the number of rows and N is the number of columns
    # Space complexity is worst case O(MxN)
    