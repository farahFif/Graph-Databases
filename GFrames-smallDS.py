#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pyspark.sql.functions import col, lit, when
from graphframes import *
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql import *
from IPython.display import display
import numpy as np

spark = SparkSession     \
  .builder     \
  .appName("Python Spark SQL basic example")     \
  .config("spark.some.config.option", "some-value")     \
  .getOrCreate()



vertex = spark.read.csv("file:///home/farah/Documents/v.txt",sep="	", inferSchema="true", header="false").toDF("id","title","year","venue","desc")


edges = spark.read.csv("file:///home/farah/Documents/e.txt",sep="	", inferSchema="true", header="false").toDF("src","dst","relationship")

#vertex.show()


#edges = edge.selectExpr("_c0 as src", "_c1 as dst","_c2 as relationship")
#edges.show()


from graphframes.examples import Graphs

g = GraphFrame(vertex, edges)

#g.edges.show()

#g.vertices.show()

motifs = g.find("(a)-[e]->(b)")

pi = g.vertices.filter(" year == 2001 ")
fl = np.array(pi.select(col('id')).collect()).ravel()

motifs = motifs.filter(" a.title like 'ARIES: A Transaction Recovery Method Supporting Fine-Granularity Locking and Partial Rollbacks Using Write-Ahead Logging' ")
motifs.show()

def parcours(dejavu, graph, next):
  tovisit = []
  temp = graph.find("(a)-[e]->(b)").filter(" a.id == '"+ str(next)+"' ")
  dejavu.append(next)
  for x in (np.array(temp.select(col('e.dst')).collect()).ravel()) :
    tovisit.append(x )
  return dejavu, tovisit

src = motifs.select(col('e.src')).collect()
dest = motifs.select(col('e.dst')).collect()
src = np.array(src).ravel()
dst = np.array(dest).ravel()
print(src)
print(dst)

dejavu = []
tovisit = dst
dejavu.append(src[0])
tovi =  []
i=0
k=1
while i < k :
  if(dejavu.__contains__(tovisit[i]) == 0):
    deja , tovi = parcours(dejavu, g,tovisit[i])
    dejavu +=deja
    tovisit = np.concatenate([tovisit, np.array(tovi)])
    dejavu = list(set(dejavu))
    k = len(tovisit)
  i += 1

print("dejavu books")
print(dejavu)

#Books = [x for x in fl if x in dejavu]
print("books")

#print(Books)
#================================== Year in which paper is the most referenced ============

ids = dejavu.toDF()
ids.show()


#================================= Most influencial ====================
print(" ---- Most Influencial Papers ---- \n ")
results2 = g.pageRank(resetProbability=0.15, maxIter=20)
k =results2.vertices.filter(" pagerank > 0.5 ")
#n = k.orderBy(k.pagerank.desc()).limit(10).show()

#================================= Five largest communities ====================
print(" ---- Largest communities ---- \n")

result = g.labelPropagation(maxIter=5)

new = result.groupBy('label').count().orderBy(col('count'), ascending=False)
new.show(5)
#result.join(new, new.label == result.label).orderBy( col('count'), ascending=False).show()

