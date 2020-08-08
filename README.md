# BiDirectionalSearch-MM
Implementation of Bidirectional Search That is Guaranteed to Meet in the Middle in Berkeley's Pacman Domain
AAAI Paper - http://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/12320/12109

My Report - https://drive.google.com/file/d/11NN1UU3cH3_hRsOi0FR80UEZ4ADCd0P3/view?usp=sharing

Instructions to Run Code from Scratch
To run Bidirectional Heuristic Search, just supply the algorithm name as bihs instead of other algorithms. The following list of commands is also saved as a text file “commands - BiDirectional Search.txt” in the project directory which should be helpful in running the commands directly. 
FIXED DOT SEARCH PROBLEM:
	Tiny Maze – python pacman.py -l tinyMaze -p SearchAgent -a fn=bihs,heuristic=manhattanHeuristic
	Medium Maze – python pacman.py -l mediumMaze -p SearchAgent -a fn=bihs,heuristic=manhattanHeuristic
	Big Maze – python pacman.py -l bigMaze -p SearchAgent -a fn=bihs,heuristic=manhattanHeuristic -z .5 --frameTime 0

CORNERS PROBLEM:
	Tiny Corners – python pacman.py -l tinyCorners -p SearchAgent -a fn=bihs,prob=CornersProblem, heuristic=cornersHeuristic
	Medium Corners – python pacman.py -l mediumCorners -p SearchAgent -a fn=bihs,prob=CornersProblem,heuristic=cornersHeuristic
	Big Corners – python pacman.py -l bigCorners -p SearchAgent -a fn=bihs,prob=CornersProblem,heuristic=cornersHeuristic -z .5 --frameTime 0

FOOD SEARCH PROBLEM:
	Tiny Safe Search (Food search) – python pacman.py -l tinySafeSearch -p SearchAgent -a fn=bihs,prob=FoodSearchProblem,heuristic=foodHeuristic
	Tricky Search – python pacman.py -l trickySearch -p SearchAgent -a fn=bihs,prob=FoodSearchProblem,heuristic=foodHeuristic
