
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/c2a4fe9c-74a5-46c2-bb58-5a20439e6efb">
    <img alt="sigil" src="https://github.com/dimenwarper/sigil/blob/400500a852b604cadb522dbc6ae5057d2daae908/logo.png" width=15%>
  </picture>
</p>


# Sigil: **A framework for auto-improving code**

Sigil is a framework for guiding code auto improvement through LLM-guided code optimization. 

Sigil allows you to:

- Auto-improve your code
    - Specify parts of your code that you want to improve via python decorators
    - Tie these pieces of code to evaluation functions that will guide the improvement
    - Automatically track code usage
    - Run LLM-guided self improvement using optimization techniques like alphaevolve
- Keep and serve the solutions you want
    - Sigil book-keeps all solutions explored throughout the optimization
    - You can serve the solutions that makes sense and keep track of usage of each
- Collaborate with others
    - Sigil enables others to contribute to your project by enabling them to launch optimization runs
    - You can similarly contribute optimization runs for other projects

## Main concepts

- **Codopt run:** An execution of LLM-guided code optimization through a number of iterations. In the end of such a run, one hopes to find a piece of code that is better than the current version.
- **Spec:** Specs are namespaces that track a scope of work. They can be thought of as sub-projects that define a context of a codopt run
- **Workspace:** A place where all codopt run solutions are stored, indexed by spec and function to improve.
- **Evaluation/eval function:** A function that evaluates a piece of code. One can potentially combine many evaluation functions and/or expose all of them to the LLM as it searches for better solutions

## Example flow

```python
## Write Specification

# File mylib.py
from sigil import improve, Spec

spec = Spec(
	name="myspec",
	...
	)
sigil.track(spec)

def myeval(x):
	return score

@improve(with_eval=myeval, serve_workspace="v1")
def myfun(x):
	return y

## Use without improvement will start tracking values and scores in spec

# File test.py
from mylib import myfun
if __name__ == '__main__':
	for v in values:
		myfun(v)
		
## Inspect the current values and scores

$ sigil inspect-samples myspec

-------------------------------------------
workspace | sample  | eval_function | score
--------------------------------------------
v1        |  x=3.   | myeval	      |   4
... 
	
## Execute self improvement

$ sigil run myspec --name v1 --optimizer alphaevolve --niter 1000

## Workspace after run

# File .sigil/ws/myspec/v1/mylib.py
...
def myfun(x):
	...
	return improved_y
...

## Inspect solutions

$ sigil inspect-solutions myspec

------------------------------------------------------------------
workspace | summary                      | eval_function | score
--------------------------------------------------------------------
v1        |  Solution uses X features    | myeval	      |   40 +/- 5
... 
	
```

## Technical anchors

- We will track solutions in using a tree of unified diffs, these will comprise a workspace of a spec
- We will have a library of codopt algorithms, tailored to LLM agents, these will include:
    - AlphaEvolve
    - Simulated annealing
    - Reinforcement learning
    - [MADS](https://community.wolfram.com/groups/-/m/t/2958734)
- At first, we will support inspection of codopt runs and solutions through the CLI, but we will consider building a UI
- We will support publishing sigil runs, maybe not just in github, but in another central repository to track usage stats/contributions


