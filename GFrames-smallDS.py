from pyspark.sql.functions import col, lit, when
from graphframes import *
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql import *
from IPython.display import display
import numpy as np
from pyspark.sql.types import IntegerType
from graphframes.examples import Graphs

spark = SparkSession     \
  .builder     \
  .appName("Python Spark SQL basic example")     \
  .config("spark.some.config.option", "some-value")     \
  .getOrCreate()



vertex = spark.read.csv("file:///home/farah/Documents/v.txt",sep="	", inferSchema="true", header="false").toDF("id","title","year","venue","desc")
edges = spark.read.csv("file:///home/farah/Documents/e.txt",sep="	", inferSchema="true", header="false").toDF("src","dst","relationship")


g = GraphFrame(vertex, edges)

verticez = g.vertices
motifs = g.find("(a)-[e]->(b)")

pi = g.vertices.filter(" year == 2001 ")
fl = np.array(pi.select(col('id')).collect()).ravel()

motifs = motifs.filter(" b.title == 'ARIES: A Transaction Recovery Method Supporting Fine-Granularity Locking and Partial Rollbacks Using Write-Ahead Logging' ")

def parcours(dejavu, graph, next):
  tovisit = []
  temp = graph.find("(a)-[e]->(b)").filter(" b.id == '"+ str(next)+"' ")
  dejavu.append(next)
  for x in (np.array(temp.select(col('e.src')).collect()).ravel()) :
    tovisit.append(x )
  return dejavu, tovisit

src = motifs.select(col('e.src')).collect()
dest = motifs.select(col('e.dst')).collect()
src = np.array(src).ravel()
dst = np.array(dest).ravel()


dejavu = []
tovisit = src
dejavu.append(dst[0])
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

#print("dejavu books")
#print(dejavu)

Books = [ int(x) for x in fl if x in dejavu]
print(Books)
print(" ===== Books that can be traced back ")
verticez.filter(~verticez.id.isin(*Books) == False).show()

#================================== Year in which paper is the most referenced ============

vall = [float(x) for x in dejavu]
ids = spark.sparkContext.parallelize(vall)

row_rdd = ids.map(lambda x: Row(x))
ids = spark.createDataFrame(row_rdd,['booksid'])

print("======= The year in which the book is the most referenced ========")
final_df = verticez.join(ids,[verticez.id == ids.booksid])
counted = final_df.groupBy('year').count().orderBy(col('count'), ascending=False).limit(1)
counted.show()

#================================= Most influencial ====================
print(" ---- Most Influencial Papers ---- \n ")
results2 = g.pageRank(resetProbability=0.15, maxIter=20)
k =results2.vertices.filter(" pagerank > 0.5 ")
k.orderBy(k.pagerank.desc()).limit(10).show()

#================================= Five largest communities ====================
print(" ---- Largest communities ---- \n")

result = g.labelPropagation(maxIter=5)

new = result.groupBy('label').count().orderBy(col('count'), ascending=False)
new.show(5)
#result.join(new, new.label == result.label).orderBy( col('count'), ascending=False).show()

