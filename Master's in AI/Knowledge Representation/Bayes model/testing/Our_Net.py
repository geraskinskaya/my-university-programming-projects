from pylab import *
import matplotlib.pyplot as plt
import os
import graphviz
import pydotplus
import pyAgrum.lib.notebook as bn
from IPython.core.display import HTML

import pyAgrum as gum

bn=gum.fastBN("c->r->w<-s<-c")
bn
bn.cpt(c).fillWith([0.4,0.6])
bn.cpt("c").fillWith([0.5,0.5])
bn.cpt("s").var_names
bn.cpt("s")[:]=[ [0.5,0.5],[0.9,0.1]]
print(bn.cpt("s")[1])
bn.cpt("s")[0,:]=0.5 # equivalent to [0.5,0.5]
bn.cpt("s")[1,:]=[0.9,0.1]
bn.cpt("s")[0,:]=0.5 # equivalent to [0.5,0.5]
bn.cpt("s")[1,:]=[0.9,0.1]
print(bn.cpt("w").var_names)
bn.cpt("w")
bn.cpt("w")[0,0,:] = [1, 0] # r=0,s=0
bn.cpt("w")[0,1,:] = [0.1, 0.9] # r=0,s=1
bn.cpt("w")[1,0,:] = [0.1, 0.9] # r=1,s=0
bn.cpt("w")[1,1,:] = [0.01, 0.99] # r=1,s=1
bn.cpt("w")[{'r': 0, 's': 0}] = [1, 0]
bn.cpt("w")[{'r': 0, 's': 1}] = [0.1, 0.9]
bn.cpt("w")[{'r': 1, 's': 0}] = [0.1, 0.9]
bn.cpt("w")[{'r': 1, 's': 1}] = [0.01, 0.99]
bn.cpt("w")
bn.cpt("r")[{'c':0}]=[0.8,0.2]
bn.cpt("r")[{'c':1}]=[0.2,0.8]
print(gum.availableBNExts())
gum.saveBN(bn,"out/WaterSprinkler.bif")
with open("out/WaterSprinkler.bif","r") as out:
    print(out.read())
ie=gum.LazyPropagation(bn)
ie.makeInference()
print (ie.posterior("w"))

HTML(f"In our BN, $P(W)=${ie.posterior('w')[:]}")
ie.posterior("w")[:]
ie.setEvidence({'s':0, 'c': 0})
ie.makeInference()
ie.posterior("w")
ie.setEvidence({'s': [0.5, 1], 'c': [1, 0]})
ie.makeInference()
ie.posterior("w") # using gnb's feature
gnb.showProba(ie.posterior("w"))
gnb.showPosterior(bn,{'s':1,'c':0},'w')
gnb.showInference(bn,evs={})
gnb.showInference(bn,evs={'s':1,'c':0})
gnb.showInference(bn,evs={'s':1,'c':[0.3,0.9]})
gnb.showInference(bn,evs={'c':[0.3,0.9]},targets={'c','w'})


# fast create a BN (random paramaters are chosen for the CPTs)
bn=gum.fastBN("A->B<-C->D->E<-F<-A;C->G<-H<-I->J")


def testIndep(bn, x, y, knowing):
    res = "" if bn.isIndependent(x, y, knowing) else " NOT"
    giv = "." if len(knowing) == 0 else f" given {knowing}."
    print(f"{x} and {y} are{res} independent{giv}")


testIndep(bn, "A", "C", [])
testIndep(bn, "A", "C", ["E"])
print()
testIndep(bn, "E", "C", [])
testIndep(bn, "E", "C", ["D"])
print()
testIndep(bn, "A", "I", [])
testIndep(bn, "A", "I", ["E"])
testIndep(bn, "A", "I", ["G"])
testIndep(bn, "A", "I", ["E", "G"])


gum.MarkovBlanket(bn,"C")
gum.MarkovBlanket(bn,"J")
[bn.variable(i).name() for i in bn.minimalCondSet("B",["A","H","J"])]
[bn.variable(i).name() for i in bn.minimalCondSet("B",["A","G","H","J"])]

ie=gum.LazyPropagation(bn)
ie.evidenceImpact("B",["A","C","H","G"])
ie.evidenceImpact("B",["A","G","H","J"])

bn=gum.fastBN("Cloudy?->Sprinkler?->Wet Grass?<-Rain?<-Cloudy?")

bn.cpt("Cloudy?").fillWith([0.5,0.5])

bn.cpt("Sprinkler?")[:]=[[0.5,0.5],
                         [0.9,0.1]]

bn.cpt("Rain?")[{'Cloudy?':0}]=[0.8,0.2]
bn.cpt("Rain?")[{'Cloudy?':1}]=[0.2,0.8]

bn.cpt("Wet Grass?")[{'Rain?': 0, 'Sprinkler?': 0}] = [1, 0]
bn.cpt("Wet Grass?")[{'Rain?': 0, 'Sprinkler?': 1}] = [0.1, 0.9]
bn.cpt("Wet Grass?")[{'Rain?': 1, 'Sprinkler?': 0}] = [0.1, 0.9]
bn.cpt("Wet Grass?")[{'Rain?': 1, 'Sprinkler?': 1}] = [0.01, 0.99]

gum.config['notebook','potential_visible_digits']=2
gnb.sideBySide(bn.cpt("Cloudy?"),captions=['$P(Cloudy)$'])
gnb.sideBySide(
  gnb.getSideBySide(bn.cpt("Sprinkler?"),captions=['$P(Sprinkler|Cloudy)$']),
  gnb.getBN(bn,size="3!"),
  gnb.getSideBySide(bn.cpt("Rain?"),captions=['$P(Rain|Cloudy)$']))
gnb.sideBySide(bn.cpt("Wet Grass?"),captions=['$P(WetGrass|Sprinkler,Rain)$'])

