5.spark版本协同过滤
单机版本的协同过滤很多同学都会写，当数据量大到一定程度(单机没办法全部载入)的时候，
我们就要考虑用spark这种分布式的系统来处理了。我们先来看看user-based和item-based协同过滤怎么写

5.1：user-based协同过滤
#-*- coding:utf8 -*-
# pySpark实现的基于用户的协同过滤
# 使用的余弦相似度

import sys
from collections import defaultdict
from itertools import combinations
import random
import numpy as np
import pdb

from pyspark import SparkContext

# user item rating timestamp
def parseVectorOnUser(line):
    '''
        解析数据，key是user，后面是item和打分
    '''
    line = line.split("|")
    return line[0],(line[1],float(line[2]))

def parseVectorOnItem(line):
    '''
        解析数据，key是item，后面是user和打分
    '''
    line = line.split("|")
    return line[1],(line[0],float(line[2]))

def sampleInteractions(item_id,users_with_rating,n):
    '''
        如果某个商品上用户行为特别多，可以选择适当做点下采样
    '''
    if len(users_with_rating) > n:
        return item_id, random.sample(users_with_rating,n)
    else:
        return item_id, users_with_rating

def findUserPairs(item_id,users_with_rating):
    '''
        对每个item，找到共同打分的user对
    '''
    for user1,user2 in combinations(users_with_rating,2):
        return (user1[0],user2[0]),(user1[1],user2[1])

def calcSim(user_pair,rating_pairs):
    ''' 
        对每个user对，根据打分计算余弦距离，并返回共同打分的item个数
    '''
    sum_xx, sum_xy, sum_yy, sum_x, sum_y, n = (0.0, 0.0, 0.0, 0.0, 0.0, 0)
    
    for rating_pair in rating_pairs:
        sum_xx += np.float(rating_pair[0]) * np.float(rating_pair[0])
        sum_yy += np.float(rating_pair[1]) * np.float(rating_pair[1])
        sum_xy += np.float(rating_pair[0]) * np.float(rating_pair[1])
        # sum_y += rt[1]
        # sum_x += rt[0]
        n += 1

    cos_sim = cosine(sum_xy,np.sqrt(sum_xx),np.sqrt(sum_yy))
    return user_pair, (cos_sim,n)

def cosine(dot_product,rating_norm_squared,rating2_norm_squared):
    '''
        2个向量A和B的余弦相似度
       dotProduct(A, B) / (norm(A) * norm(B))
    '''
    numerator = dot_product
    denominator = rating_norm_squared * rating2_norm_squared

    return (numerator / (float(denominator))) if denominator else 0.0

def keyOnFirstUser(user_pair,item_sim_data):
    '''
        对于每个user-user对，用第一个user做key(好像有点粗暴...)
    '''
    (user1_id,user2_id) = user_pair
    return user1_id,(user2_id,item_sim_data)

def nearestNeighbors(user,users_and_sims,n):
    '''
        选出相似度最高的N个邻居
    '''
    users_and_sims.sort(key=lambda x: x[1][0],reverse=True)
    return user, users_and_sims[:n]

def topNRecommendations(user_id,user_sims,users_with_rating,n):
    '''
        根据最近的N个邻居进行推荐
    '''

    totals = defaultdict(int)
    sim_sums = defaultdict(int)

    for (neighbor,(sim,count)) in user_sims:

        # 遍历邻居的打分
        unscored_items = users_with_rating.get(neighbor,None)

        if unscored_items:
            for (item,rating) in unscored_items:
                if neighbor != item:

                    # 更新推荐度和相近度
                    totals[neighbor] += sim * rating
                    sim_sums[neighbor] += sim

    # 归一化
    scored_items = [(total/sim_sums[item],item) for item,total in totals.items()]

    # 按照推荐度降序排列
    scored_items.sort(reverse=True)

    # 推荐度的item
    ranked_items = [x[1] for x in scored_items]

    return user_id,ranked_items[:n]

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print >> sys.stderr, \
            "Usage: PythonUserCF <master> <file>"
        exit(-1)

    sc = SparkContext(sys.argv[1],"PythonUserCF")
    lines = sc.textFile(sys.argv[2])

    '''
        处理数据，获得稀疏item-user矩阵:
        item_id -> ((user_1,rating),(user2,rating))
    '''
    item_user_pairs = lines.map(parseVectorOnItem).groupByKey().map(
        lambda p: sampleInteractions(p[0],p[1],500)).cache()

    '''
        获得2个用户所有的item-item对得分组合:
        (user1_id,user2_id) -> [(rating1,rating2),
                                (rating1,rating2),
                                (rating1,rating2),
                                ...]
    '''
    pairwise_users = item_user_pairs.filter(
        lambda p: len(p[1]) > 1).map(
        lambda p: findUserPairs(p[0],p[1])).groupByKey()

    '''
        计算余弦相似度，找到最近的N个邻居:
        (user1,user2) ->    (similarity,co_raters_count)
    '''
    user_sims = pairwise_users.map(
        lambda p: calcSim(p[0],p[1])).map(
        lambda p: keyOnFirstUser(p[0],p[1])).groupByKey().map(
        lambda p: nearestNeighbors(p[0],p[1],50))

    ''' 
        对每个用户的打分记录整理成如下形式
        user_id -> [(item_id_1, rating_1),
                   [(item_id_2, rating_2),
                    ...]
    '''

    user_item_hist = lines.map(parseVectorOnUser).groupByKey().collect()

    ui_dict = {}
    for (user,items) in user_item_hist: 
        ui_dict[user] = items

    uib = sc.broadcast(ui_dict)

    '''
        为每个用户计算Top N的推荐
        user_id -> [item1,item2,item3,...]
    '''
    user_item_recs = user_sims.map(lambda p: topNRecommendations(p[0],p[1],uib.value,100)).collect()
	
	
5.2：item-based协同过滤

#-*- coding:utf8 -*-
# pySpark实现的基于物品的协同过滤

import sys
from collections import defaultdict
from itertools import combinations
import numpy as np
import random
import csv
import pdb

from pyspark import SparkContext

def parseVector(line):
    '''
        解析数据，key是item，后面是user和打分
    '''
    line = line.split("|")
    return line[0],(line[1],float(line[2]))

def sampleInteractions(user_id,items_with_rating,n):
    '''
        如果某个用户打分行为特别多，可以选择适当做点下采样
    '''
    if len(items_with_rating) > n:
        return user_id, random.sample(items_with_rating,n)
    else:
        return user_id, items_with_rating

def findItemPairs(user_id,items_with_rating):
    '''
        对每个用户的打分item，组对
    '''
    for item1,item2 in combinations(items_with_rating,2):
        return (item1[0],item2[0]),(item1[1],item2[1])

def calcSim(item_pair,rating_pairs):
    ''' 
        对每个item对，根据打分计算余弦距离，并返回共同打分的user个数
    '''
    sum_xx, sum_xy, sum_yy, sum_x, sum_y, n = (0.0, 0.0, 0.0, 0.0, 0.0, 0)
    
    for rating_pair in rating_pairs:
        sum_xx += np.float(rating_pair[0]) * np.float(rating_pair[0])
        sum_yy += np.float(rating_pair[1]) * np.float(rating_pair[1])
        sum_xy += np.float(rating_pair[0]) * np.float(rating_pair[1])
        # sum_y += rt[1]
        # sum_x += rt[0]
        n += 1

    cos_sim = cosine(sum_xy,np.sqrt(sum_xx),np.sqrt(sum_yy))
    return item_pair, (cos_sim,n)

def cosine(dot_product,rating_norm_squared,rating2_norm_squared):
    '''
    The cosine between two vectors A, B
       dotProduct(A, B) / (norm(A) * norm(B))
    '''
    numerator = dot_product
    denominator = rating_norm_squared * rating2_norm_squared
    return (numerator / (float(denominator))) if denominator else 0.0

def correlation(size, dot_product, rating_sum, \
            rating2sum, rating_norm_squared, rating2_norm_squared):
    '''
        2个向量A和B的相似度
        [n * dotProduct(A, B) - sum(A) * sum(B)] /
        sqrt{ [n * norm(A)^2 - sum(A)^2] [n * norm(B)^2 - sum(B)^2] }

    '''
    numerator = size * dot_product - rating_sum * rating2sum
    denominator = sqrt(size * rating_norm_squared - rating_sum * rating_sum) * \
                    sqrt(size * rating2_norm_squared - rating2sum * rating2sum)

    return (numerator / (float(denominator))) if denominator else 0.0

def keyOnFirstItem(item_pair,item_sim_data):
    '''
        对于每个item-item对，用第一个item做key(好像有点粗暴...)
    '''
    (item1_id,item2_id) = item_pair
    return item1_id,(item2_id,item_sim_data)

def nearestNeighbors(item_id,items_and_sims,n):
    '''
        排序选出相似度最高的N个邻居
    '''
    items_and_sims.sort(key=lambda x: x[1][0],reverse=True)
    return item_id, items_and_sims[:n]

def topNRecommendations(user_id,items_with_rating,item_sims,n):
    '''
        根据最近的N个邻居进行推荐
    '''
    
    totals = defaultdict(int)
    sim_sums = defaultdict(int)

    for (item,rating) in items_with_rating:

        # 遍历item的邻居
        nearest_neighbors = item_sims.get(item,None)

        if nearest_neighbors:
            for (neighbor,(sim,count)) in nearest_neighbors:
                if neighbor != item:

                    # 更新推荐度和相近度
                    totals[neighbor] += sim * rating
                    sim_sums[neighbor] += sim

    # 归一化
    scored_items = [(total/sim_sums[item],item) for item,total in totals.items()]

    # 按照推荐度降序排列
    scored_items.sort(reverse=True)

    ranked_items = [x[1] for x in scored_items]

    return user_id,ranked_items[:n]

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print >> sys.stderr, \
            "Usage: PythonItemCF <master> <file>"
        exit(-1)

    sc = SparkContext(sys.argv[1], "PythonItemCF")
    lines = sc.textFile(sys.argv[2])

    ''' 
        处理数据，获得稀疏user-item矩阵:
        user_id -> [(item_id_1, rating_1),
                   [(item_id_2, rating_2),
                    ...]
    '''
    user_item_pairs = lines.map(parseVector).groupByKey().map(
        lambda p: sampleInteractions(p[0],p[1],500)).cache()

    '''
        获取所有item-item组合对
        (item1,item2) ->    [(item1_rating,item2_rating),
                             (item1_rating,item2_rating),
                             ...]
    '''

    pairwise_items = user_item_pairs.filter(
        lambda p: len(p[1]) > 1).map(
        lambda p: findItemPairs(p[0],p[1])).groupByKey()

    '''
        计算余弦相似度，找到最近的N个邻居:
        (item1,item2) ->    (similarity,co_raters_count)
    '''

    item_sims = pairwise_items.map(
        lambda p: calcSim(p[0],p[1])).map(
        lambda p: keyOnFirstItem(p[0],p[1])).groupByKey().map(
        lambda p: nearestNeighbors(p[0],p[1],50)).collect()


    item_sim_dict = {}
    for (item,data) in item_sims: 
        item_sim_dict[item] = data

    isb = sc.broadcast(item_sim_dict)

    '''
        计算最佳的N个推荐结果
        user_id -> [item1,item2,item3,...]
    '''
    user_item_recs = user_item_pairs.map(lambda p: topNRecommendations(p[0],p[1],isb.value,500)).collect()
	
5.3 spark自带推荐算法

#!/usr/bin/env python
# 基于spark中ALS的推荐系统，针对movielens中电影打分数据做推荐
# Edit：寒小阳(hanxiaoyang.ml@gmail.com)

import sys
import itertools
from math import sqrt
from operator import add
from os.path import join, isfile, dirname

from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS

def parseRating(line):
    """
        MovieLens的打分格式是userId::movieId::rating::timestamp
        我们对格式做一个解析
    """
    fields = line.strip().split("::")
    return long(fields[3]) % 10, (int(fields[0]), int(fields[1]), float(fields[2]))

def parseMovie(line):
    """
        对应的电影文件的格式为movieId::movieTitle
        解析成int id, 文本
    """
    fields = line.strip().split("::")
    return int(fields[0]), fields[1]

def loadRatings(ratingsFile):
    """
        载入得分
    """
    if not isfile(ratingsFile):
        print "File %s does not exist." % ratingsFile
        sys.exit(1)
    f = open(ratingsFile, 'r')
    ratings = filter(lambda r: r[2] > 0, [parseRating(line)[1] for line in f])
    f.close()
    if not ratings:
        print "No ratings provided."
        sys.exit(1)
    else:
        return ratings

def computeRmse(model, data, n):
    """
        评估的时候要用的，计算均方根误差
    """
    predictions = model.predictAll(data.map(lambda x: (x[0], x[1])))
    predictionsAndRatings = predictions.map(lambda x: ((x[0], x[1]), x[2])) \
      .join(data.map(lambda x: ((x[0], x[1]), x[2]))) \
      .values()
    return sqrt(predictionsAndRatings.map(lambda x: (x[0] - x[1]) ** 2).reduce(add) / float(n))

if __name__ == "__main__":
    if (len(sys.argv) != 3):
        print "Usage: /path/to/spark/bin/spark-submit --driver-memory 2g " + \
          "MovieLensALS.py movieLensDataDir personalRatingsFile"
        sys.exit(1)

    # 设定环境
    conf = SparkConf() \
      .setAppName("MovieLensALS") \
      .set("spark.executor.memory", "2g")
    sc = SparkContext(conf=conf)

    # 载入打分数据
    myRatings = loadRatings(sys.argv[2])
    myRatingsRDD = sc.parallelize(myRatings, 1)

    movieLensHomeDir = sys.argv[1]

    # 得到的ratings为(时间戳最后一位整数, (userId, movieId, rating))格式的RDD
    ratings = sc.textFile(join(movieLensHomeDir, "ratings.dat")).map(parseRating)

    # 得到的movies为(movieId, movieTitle)格式的RDD
    movies = dict(sc.textFile(join(movieLensHomeDir, "movies.dat")).map(parseMovie).collect())

    numRatings = ratings.count()
    numUsers = ratings.values().map(lambda r: r[0]).distinct().count()
    numMovies = ratings.values().map(lambda r: r[1]).distinct().count()

    print "Got %d ratings from %d users on %d movies." % (numRatings, numUsers, numMovies)

    # 根据时间戳最后一位把整个数据集分成训练集(60%), 交叉验证集(20%), 和评估集(20%)

    # 训练, 交叉验证, 测试 集都是(userId, movieId, rating)格式的RDD

    numPartitions = 4
    training = ratings.filter(lambda x: x[0] < 6) \
      .values() \
      .union(myRatingsRDD) \
      .repartition(numPartitions) \
      .cache()

    validation = ratings.filter(lambda x: x[0] >= 6 and x[0] < 8) \
      .values() \
      .repartition(numPartitions) \
      .cache()

    test = ratings.filter(lambda x: x[0] >= 8).values().cache()

    numTraining = training.count()
    numValidation = validation.count()
    numTest = test.count()

    print "Training: %d, validation: %d, test: %d" % (numTraining, numValidation, numTest)

    # 训练模型，在交叉验证集上看效果

    ranks = [8, 12]
    lambdas = [0.1, 10.0]
    numIters = [10, 20]
    bestModel = None
    bestValidationRmse = float("inf")
    bestRank = 0
    bestLambda = -1.0
    bestNumIter = -1

    for rank, lmbda, numIter in itertools.product(ranks, lambdas, numIters):
        model = ALS.train(training, rank, numIter, lmbda)
        validationRmse = computeRmse(model, validation, numValidation)
        print "RMSE (validation) = %f for the model trained with " % validationRmse + \
              "rank = %d, lambda = %.1f, and numIter = %d." % (rank, lmbda, numIter)
        if (validationRmse < bestValidationRmse):
            bestModel = model
            bestValidationRmse = validationRmse
            bestRank = rank
            bestLambda = lmbda
            bestNumIter = numIter

    testRmse = computeRmse(bestModel, test, numTest)

    # 在测试集上评估 交叉验证集上最好的模型
    print "The best model was trained with rank = %d and lambda = %.1f, " % (bestRank, bestLambda) \
      + "and numIter = %d, and its RMSE on the test set is %f." % (bestNumIter, testRmse)

    # 我们把基线模型设定为每次都返回平均得分的模型
    meanRating = training.union(validation).map(lambda x: x[2]).mean()
    baselineRmse = sqrt(test.map(lambda x: (meanRating - x[2]) ** 2).reduce(add) / numTest)
    improvement = (baselineRmse - testRmse) / baselineRmse * 100
    print "The best model improves the baseline by %.2f" % (improvement) + "%."

    # 个性化的推荐(针对某个用户)

    myRatedMovieIds = set([x[1] for x in myRatings])
    candidates = sc.parallelize([m for m in movies if m not in myRatedMovieIds])
    predictions = bestModel.predictAll(candidates.map(lambda x: (0, x))).collect()
    recommendations = sorted(predictions, key=lambda x: x[2], reverse=True)[:50]

    print "Movies recommended for you:"
    for i in xrange(len(recommendations)):
        print ("%2d: %s" % (i + 1, movies[recommendations[i][1]])).encode('ascii', 'ignore')

    # clean up
    sc.stop()