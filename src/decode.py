import pywrapfst as fst
import sys

tm = fst.Fst.read(sys.argv[1])
lm = fst.Fst.read(sys.argv[2])

isym = {}
with open(sys.argv[3], "r") as isymfile:
  for line in isymfile:
    x, y = line.strip().split()
    isym[x] = int(y)

osym = {}
with open(sys.argv[4], "r") as osymfile:
  for line in osymfile:
    x, y = line.strip().split()
    osym[int(y)] = x

for line in sys.stdin:
  # Read input 
  compiler = fst.Compiler()
  arr = line.strip().split() + ["</s>"]
  for i, x in enumerate(arr):
    xsym = isym[x] if x in isym else isym["<unk>"]
    print >> compiler, "%d %d %s %s" % (i, i+1, xsym, xsym)
  print >> compiler, "%s" % (len(arr))
  ifst = compiler.compile()

  # Create the search graph and do search
  graph = fst.compose(ifst, tm)
  graph = fst.compose(graph, lm)
  graph = fst.shortestpath(graph)

  # Read off the output
  out = []
  for state in graph.states():
    for arc in graph.arcs(state):
      if arc.olabel != 0:
        out.append(osym[arc.olabel])
  print(" ".join(reversed(out[1:])))

