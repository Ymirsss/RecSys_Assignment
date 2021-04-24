import numpy as np
import csv

def getData(file):
    li = []
    with open(file, 'rt') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        for row in reader:
            # print(row)
            (userid, movieid, rating, ts) = row
            li.append([int(userid), int(movieid),float(rating)])
        data = np.array(li)
    return data

